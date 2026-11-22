"""Tests for scripts/nat_encret: the rMatch driver and the nat_encret builder.

Every fixture is synthetic (subject 98) and built under tmp_path; nothing reads
the dataset, and nothing needs a GPU.
"""

import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

NAT = Path(__file__).resolve().parents[1] / "scripts" / "nat_encret"
if str(NAT) not in sys.path:
    sys.path.insert(0, str(NAT))

import build_nat_encret as B  # noqa: E402
import encret_io as io  # noqa: E402
import rmatch_run as R  # noqa: E402

# The prompt is copied verbatim from rMatch @ f86b543. Editing it changes every map built with it,
# so a change must be deliberate: update this hash and the provenance it is recorded in.
PROMPT_SHA256 = "154e667a53f0eb9006c02c7a7ec9f15ead1037d8e5b6c0ee834bc7f8e40a38a7"


def test_prompt_template_is_pinned():
    assert hashlib.sha256(R.DEFAULT_PROMPT.encode()).hexdigest() == PROMPT_SHA256 == R.prompt_sha256()


@pytest.mark.parametrize("raw, want", [
    ("reasoning... <NONE>", set()),
    ("it says <3, 7, 12>", {3, 7, 12}),
    ("<2>", {2}),
    ("no brackets at all", None),
])
def test_parse_output(raw, want):
    assert R._parse_output(raw) == want


def _events():
    return pd.DataFrame(dict(film=["f"] * 3 + ["g"] * 2, seg_number=[1, 2, 3, 1, 2],
                             onset=[0., 10., 20., 0., 5.], offset=[10., 20., 30., 5., 9.],
                             text=["A cat sits.", "A dog barks.", "They leave.", "Rain.", "Sun."]))


def _units(n=3, film="f", trial=1, session="20", rank=1):
    return pd.DataFrame(dict(subject="98", session=session, run="", trial_num=trial, unit_idx=range(n),
                             stimulus_id=film, recall_rank=rank, first_row=range(0, 3 * n, 3),
                             last_row=range(2, 3 * n, 3), n_words=3, onset=np.arange(n) * 2.0,
                             offset=np.arange(n) * 2.0 + 1.5, text=[f"unit {i} here." for i in range(n)]))


class TestPrompts:
    def test_story_is_numbered_in_time_order_and_target_marked(self):
        P = R.make_prompts(_units(), _events())
        assert len(P) == 3 and set(P.n_events) == {3}
        p = P.prompt.iloc[1]
        assert "1. A cat sits.\n2. A dog barks.\n3. They leave." in p
        assert "unit 0 here.\n>>> unit 1 here. <<<\nunit 2 here." in p
        assert "This is the original story:\nA cat sits. A dog barks. They leave." in p

    def test_window_is_five_each_side(self):
        P = R.make_prompts(_units(n=13), _events())
        window = P.prompt.iloc[6].split("Recall window:\n")[1].split("\n\nWhich")[0].splitlines()
        assert len(window) == 11 and window[5] == ">>> unit 6 here. <<<"

    def test_unknown_film_raises(self):
        with pytest.raises(ValueError, match="without SEG-C events"):
            R.make_prompts(_units(film="h"), _events())

    def test_pseudo_pairs_each_film_with_another_films_recall(self):
        U = pd.concat([_units(film="f", trial=1), _units(film="g", trial=2)])
        P = R.make_prompts(U, _events(), "pseudo")
        assert set(zip(P.film, P.source_film)) == {("f", "g"), ("g", "f")}


def _shard(work, P, raws, tag="real_00of01"):
    (work / "shards").mkdir(parents=True, exist_ok=True)
    S = P.drop(columns="prompt").assign(raw=raws, hit_max=False)
    S.to_parquet(work / "shards" / f"gen_{tag}.parquet")
    (work / "shards" / f"gen_{tag}.json").write_text(json.dumps(dict(
        model=R.MODEL, engine="vllm 0.0", rmatch_commit=R.RMATCH_COMMIT, gpu="test")))


class TestCollect:
    def test_rows_per_matched_event_and_unmatched(self, tmp_path):
        P = R.make_prompts(_units(), _events())
        P.to_parquet(tmp_path / "prompts_real.parquet")
        _shard(tmp_path, P, ["<1, 2>", "<NONE>", "<3, 9> and garbage"])
        M, info = R.collect(tmp_path, "real", _events())
        assert M.groupby("unit_idx").size().to_dict() == {0: 2, 1: 1, 2: 1}
        assert M[M.unit_idx == 0].event_seg_number.tolist() == [1, 2]
        assert not M[M.unit_idx == 1].mapped.iloc[0]
        assert info["out_of_range_ids"] == 1 and info["parse_fail"] == 0 and info["n_units"] == 3

    def test_missing_shard_output_refused(self, tmp_path):
        P = R.make_prompts(_units(), _events())
        P.to_parquet(tmp_path / "prompts_real.parquet")
        _shard(tmp_path, P.iloc[:2], ["<1>", "<2>"])
        with pytest.raises(SystemExit, match="no output"):
            R.collect(tmp_path, "real", _events())


def _derivative_with_map(tmp_path, raws=("<1, 2>", "<NONE>", "<3>")):
    out, work = tmp_path / "nat_encret", tmp_path / "work"
    U = _units()
    B.write_units(U.drop(columns="text"), out)
    P = R.make_prompts(U, _events())
    work.mkdir()
    P.to_parquet(work / "prompts_real.parquet")
    _shard(work, P, list(raws))
    M, info = R.collect(work, "real", _events())
    M.to_csv(work / "mapping_rmatch.tsv", sep="\t", index=False)
    (work / "mapping_rmatch.json").write_text(json.dumps(info))
    return out, work


class TestMapPackaging:
    def test_writes_text_free_map_with_provenance(self, tmp_path):
        out, work = _derivative_with_map(tmp_path)
        B.cmd_map(Namespace(out=out, work=work, matcher="rmatch", revision="abc123"))
        p = io.map_path(out, dict(sub="98", ses="20", run=""), "rmatch")
        T = io.read_tsv(p)
        assert list(T.columns) == B.MAP_COLS
        assert T.seg_number.fillna(-1).astype(int).tolist() == [1, 2, -1, 3]
        assert T.mapped.tolist() == [1, 1, 0, 1]
        prov = json.loads(p.with_suffix(".json").read_text())["Matcher"]
        assert prov["ModelRevision"] == "abc123" and prov["PromptSHA256"] == PROMPT_SHA256
        assert not any("here." in line for line in p.read_text().splitlines())  # no unit text

    def test_revision_required_when_shards_lack_it(self, tmp_path):
        out, work = _derivative_with_map(tmp_path)
        with pytest.raises(SystemExit, match="revision"):
            B.cmd_map(Namespace(out=out, work=work, matcher="rmatch", revision=None))

    def test_unit_missing_from_map_refused(self, tmp_path):
        out, work = _derivative_with_map(tmp_path)
        M = pd.read_csv(work / "mapping_rmatch.tsv", sep="\t")
        M[M.unit_idx != 2].to_csv(work / "mapping_rmatch.tsv", sep="\t", index=False)
        with pytest.raises(SystemExit, match="1 units unmapped"):
            B.cmd_map(Namespace(out=out, work=work, matcher="rmatch", revision="abc123"))


def test_event_recall_counts_first_recalls_only(tmp_path):
    out, work = _derivative_with_map(tmp_path)
    B.cmd_map(Namespace(out=out, work=work, matcher="rmatch", revision="abc123"))
    segb = pd.DataFrame(dict(film=["f", "f", "g"], seg_number=[1, 2, 1], onset=[0., 15., 0.],
                             offset=[15., 30., 9.]))
    grid = B.event_grid(_events(), segb)
    assert grid[grid.film == "f"].segb.tolist() == [1, 2, 2]  # midpoints 5, 15, 25
    P = B.event_recall(io.read_units(out), B.read_maps(out, "rmatch"), grid)
    assert P.n_units.tolist() == [1, 1, 1] and P.recalled.tolist() == [1, 1, 1]
    assert P.trial_units.iloc[0] == 3 and P.trial_words.iloc[0] == 9
    N = B.event_norms(P)
    assert N.p_recall.tolist() == [1.0, 1.0, 1.0] and N.n_subjects.tolist() == [1, 1, 1]
