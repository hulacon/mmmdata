"""Tests for scripts/nat_encret/build_film_events.py (the film event table).

Every fixture is synthetic and built under tmp_path. Nothing reads the dataset, and no
embedding model is loaded: a fake embedder stands in for mpnet.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
NAT = SCRIPTS / "nat_encret"
for _p in (NAT, SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_film_events as F  # noqa: E402
import stimfeat_preflight as pf  # noqa: E402

NONSTORY = {("f", 4)}


def _ann():
    """Two films. f: SEG-B 1 holds C 1-2, SEG-B 2 holds C 3-4 (C 4 non-story); g: one scene."""
    rows = [
        ("f", "B", 1, 0, 20, "Opening"), ("f", "B", 2, 20, 40, "Ending"),
        ("f", "C", 1, 0, 10, "A cat sits on a mat."), ("f", "C", 2, 10, 20, "The cat chases a dog."),
        ("f", "C", 3, 20, 30, "The dog runs home."), ("f", "C", 4, 30, 40, "A black screen."),
        ("g", "B", 1, 0, 9, "Weather"),
        ("g", "C", 1, 0, 5, "Rain falls."), ("g", "C", 2, 5, 9, "The sun comes out."),
    ]
    return pd.DataFrame(rows, columns=["stimulus_id", "level", "seg_number", "onset", "offset", "description"])


def _fake_embed(texts, device=None):
    rng = np.random.default_rng(len(texts))
    E = rng.normal(size=(len(texts), 8))
    return E / np.linalg.norm(E, axis=1, keepdims=True), dict(checkpoint="fake", n_texts=len(texts))


def _store(tmp_path, segc, model="minilm", dim=4, text_override=None):
    """A minimal stimuli_features root holding one word2psy SEG-C chunk table for ``model``."""
    sf = tmp_path / "sf"
    for d in ("movies", "_labels", "_inputs"):
        (sf / d).mkdir(parents=True, exist_ok=True)
    n = len(segc)
    rng = np.random.default_rng(0)
    ch = pd.DataFrame(dict(chunk_idx=range(n), chunk_label=segc.stimulus_id, n_words=segc.n_words,
                           stimulus_id=segc.stimulus_id, onset=segc.onset, offset=segc.offset))
    for k in range(dim):
        ch[f"{model}_{k:03d}"] = rng.normal(size=n)
    ch.to_csv(sf / "movies" / f"annot_{model}_chunks.csv", index=False)
    pd.DataFrame(dict(stimulus_id=segc.stimulus_id, chunk_idx=range(n), seg_number=segc.filmevent_seg_number)
                 ).to_csv(sf / "_labels" / "movies_annot_segc.csv", index=False)
    text = segc.filmevent_text.tolist() if text_override is None else text_override
    pd.DataFrame(dict(stimulus_id=segc.stimulus_id, onset=segc.onset, offset=segc.offset, description=text)
                 ).to_csv(sf / "_inputs" / "movies_annot_segc.csv", index=False)
    (sf / "movies" / f"annot_{model}.meta.json").write_text(json.dumps(
        dict(extractor="word2psy", extractor_version="0.0.0", models={model: dict(checkpoint=f"ckpt/{model}")})))
    return sf


class TestTables:
    def test_segc_rows_story_flag_and_scene(self):
        c = F.segc_table(_ann(), NONSTORY)
        assert list(c.columns) == F.SEGC_COLS
        assert len(c) == 6
        f = c[c.stimulus_id == "f"].set_index("filmevent_seg_number")
        assert f.filmevent_story.to_dict() == {1: 1, 2: 1, 3: 1, 4: 0}
        assert f.filmevent_segb.to_dict() == {1: 1, 2: 1, 3: 2, 4: 2}
        assert f.loc[2, "n_words"] == 5 and f.loc[2, "filmevent_duration"] == 10

    def test_unknown_nonstory_key_fails_loudly(self):
        with pytest.raises(ValueError, match="NONSTORY keys not in"):
            F.segc_table(_ann(), {("f", 99)})

    def test_segb_joins_story_text_only(self):
        c = F.segc_table(_ann(), NONSTORY)
        b = F.segb_table(_ann(), c).set_index(["stimulus_id", "filmevent_seg_number"])
        assert b.loc[("f", 1), "filmevent_story_text"] == "A cat sits on a mat. The cat chases a dog."
        assert b.loc[("f", 2), "filmevent_story_text"] == "The dog runs home."
        assert b.loc[("f", 2), "filmevent_n_events"] == 2 and b.loc[("f", 2), "filmevent_n_story_events"] == 1
        assert b.loc[("f", 2), "filmevent_story_duration"] == 10
        assert b.loc[("f", 1), "filmevent_label"] == "Opening"
        assert b.loc[("f", 1), "n_words"] == 11


class TestCentrality:
    def test_degree_excludes_self_and_non_story(self):
        v = np.array([[1, 0], [1, 0], [0, 1], [1, 1], [5, 5.]])
        film = ["a", "a", "a", "a", "b"]
        story = [1, 1, 1, 0, 1]
        deg, z = F.centrality(v, film, story)
        # a: cos(0,1) = 1, cos(0,2) = cos(1,2) = 0; row 3 is non-story and outside the graph
        np.testing.assert_allclose(deg[:3], [1, 1, 0])
        assert np.isnan(deg[3]) and np.isnan(z[3])
        assert np.isnan(z[4])  # a film with one story node has no centrality
        np.testing.assert_allclose([np.mean(z[:3]), np.std(z[:3])], [0, 1], atol=1e-12)

    def test_unnormalised_input_gives_cosine(self):
        v = np.array([[2, 0], [3, 0], [0, 7.]])
        deg, _ = F.centrality(v, ["a"] * 3, [1, 1, 1])
        np.testing.assert_allclose(deg, [1, 1, 0])


class TestStoredVectors:
    def test_aligned_to_segc_rows(self, tmp_path):
        c = F.segc_table(_ann(), NONSTORY)
        sf = _store(tmp_path, c)
        V, prov = F.stored_vectors(sf, "minilm", c.iloc[::-1].reset_index(drop=True))
        ch = pd.read_csv(sf / "movies" / "annot_minilm_chunks.csv")
        np.testing.assert_allclose(V[::-1], ch.filter(like="minilm_").to_numpy())
        assert prov["checkpoint"] == "ckpt/minilm" and prov["dim"] == 4

    def test_text_drift_refused(self, tmp_path):
        c = F.segc_table(_ann(), NONSTORY)
        sf = _store(tmp_path, c, text_override=["changed"] + c.filmevent_text.tolist()[1:])
        with pytest.raises(ValueError, match="differ from the stored chunks"):
            F.stored_vectors(sf, "minilm", c)

    def test_missing_cell_names_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="annot_minilm_chunks.csv"):
            F.stored_vectors(tmp_path, "minilm", F.segc_table(_ann(), NONSTORY))


class TestBuildAndSidecar:
    @pytest.fixture
    def built(self, tmp_path, monkeypatch):
        c0 = F.segc_table(_ann(), NONSTORY)
        sf = _store(tmp_path, c0)
        monkeypatch.setattr(F, "STORED", ("minilm",))
        segc, segb, prov = F.build(sf, embed=_fake_embed, ann=_ann(), nonstory=NONSTORY)
        meta = F.sidecar(segc, segb, tmp_path, prov["stored"], prov["mpnet"], prov["mpnet_b"], 0.0)
        return segc, segb, meta

    def test_nan_only_where_declared(self, built):
        segc, segb, meta = built
        assert F.undeclared_nans(segc, meta) == [] and F.undeclared_nans(segb, meta) == []
        assert segc.loc[segc.filmevent_story == 0, "centrality_mpnet_z"].isna().all()
        f = segc[(segc.stimulus_id == "f") & (segc.filmevent_story == 1)]
        assert f.centrality_mpnet_z.notna().all()
        g = segc[segc.stimulus_id == "g"]  # two story nodes: degree defined, z is not
        assert g.centrality_mpnet_degree.notna().all() and g.centrality_mpnet_z.isna().all()
        assert segb.centrality_mpnet_degree.notna().sum() == 2  # f's two scenes; g has one scene
        assert "centrality_minilm_z" not in segb.columns  # stored embeddings are of SEG-B labels

    def test_every_column_reserved_or_prefixed(self, built):
        segc, segb, meta = built
        reserved = {"stimulus_id", "onset", "offset", "n_words"}
        for t, df in (("segc", segc), ("segb", segb)):
            declared = {c for m in meta["models"].values() for c in m["features"]["tables"].get(t, [])}
            assert set(df.columns) - reserved == declared, t
            for m, entry in meta["models"].items():
                assert all(c.startswith(m + "_") for c in entry["features"]["columns"]), m

    def test_labels_resolve_for_every_column(self, built):
        _, _, meta = built
        for m, entry in meta["models"].items():
            lab = entry["labels"]
            assert set(lab["columns"]) <= set(entry["features"]["columns"]), m
            for col in entry["features"]["columns"]:
                merged = {**lab["default"], **lab["columns"].get(col, {})}
                assert pf._entry_problems(merged, f"{m}[{col}]", complete=True) == [], (m, col)

    def test_sidecar_core_keys(self, built):
        _, _, meta = built
        for k in ("schema_version", "extractor", "extractor_version", "created_at", "input", "output", "models",
                  "total_runtime_sec"):
            assert k in meta
        assert meta["models"]["centrality_minilm"]["labels"]["default"]["derived_from"] == "ckpt/minilm"
        assert meta["models"]["centrality_mpnet"]["checkpoint"] == F.MPNET


def test_table_diffs_round_trip_through_csv(tmp_path):
    fresh = pd.DataFrame(dict(stimulus_id=["f", "g"], centrality_mpnet_z=[0.123456789, np.nan],
                              filmevent_text=["A, quoted \"text\".", ""]))
    fresh.to_csv(tmp_path / "t.csv", index=False, float_format="%.6g")
    disk = pd.read_csv(tmp_path / "t.csv", dtype={"stimulus_id": str}, keep_default_na=False,
                       na_values={"centrality_mpnet_z": [""]})
    assert F.table_diffs(fresh, disk) == []
    disk.loc[0, "stimulus_id"] = "h"
    disk.loc[1, "filmevent_text"] = "x"
    assert F.table_diffs(fresh, disk) == ["stimulus_id differs from a fresh build",
                                          "filmevent_text differs from a fresh build"]
