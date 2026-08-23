"""Tests for the §4.1 compliance gate in scripts/stimfeat_campaign.py.

Each case reproduces a defect the gate *missed* during the extraction
campaign (workbench/stimfeat-campaign, 2026-08-22/23). Every one of them
reached the store and was found by something else — psytwill refusing a
group, or a hand count while sizing — never by `verify`, which reported the
store clean each time. They are regression cases in the strict sense: they
fail against the pre-2026-08-23 gate, which narrowed to declared columns
only.

Needs psytwill importable, so run under the stimfeat env:

    /gpfs/projects/hulacon/shared/envs/stimfeat/bin/python -m pytest \
        tests/test_stimfeat_verify.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("psytwill", reason="the gate calls psytwill's own resolver")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stimfeat_campaign.py"


@pytest.fixture(scope="module")
def campaign():
    """Import the campaign driver as a module, without running its CLI."""
    spec = importlib.util.spec_from_file_location("stimfeat_campaign", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # Registered before exec: the driver's dataclasses use PEP 563 string
    # annotations, which dataclasses resolves through sys.modules[__module__].
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_cell(tmp_path, sidecar, tables):
    """Write one §4.1 family: `<stem>.meta.json` + its CSVs. Returns the stem."""
    stem = tmp_path / "beats.csv"
    (tmp_path / "beats.meta.json").write_text(json.dumps(sidecar))
    for name, columns in tables.items():
        rows = ",".join(columns)
        values = ",".join("0" for _ in columns)
        (tmp_path / name).write_text(f"{rows}\n{values}\n")
    return stem


def kinds(violations):
    return {kind for kind, _, _ in violations}


# ---------------------------------------------------------------------------
# The three misses
# ---------------------------------------------------------------------------

def test_undeclared_unprefixed_column_is_caught(campaign, tmp_path):
    """aud2psy `beats` emitted a bare `is_downbeat` that no sidecar declared.

    Invisible to the old gate by construction: it flagged only columns the
    sidecar *claimed as features*, and this one was never declared at all.
    psytwill melts it anyway — everything outside INDEX_COLUMNS is a feature —
    so it landed as 8,392 rows attributed to no model.
    """
    stem = write_cell(
        tmp_path,
        {"extractor": "aud2psy", "models": {"beats": {"columns": ["time"]}}},
        {"beats_beats.csv": ["stimulus_id", "time", "is_downbeat"]},
    )
    violations = campaign.family_violations(stem)
    assert "unattributable" in kinds(violations)
    assert any("is_downbeat" in cols for _, _, cols in violations)


def test_features_without_stimulus_id_are_caught(campaign, tmp_path):
    """335 word-level tables carried features but no `stimulus_id`.

    The old gate checked column *attributability* and never that the key
    column exists, so this passed. psytwill falls back to `chunk_label` with
    a warning — which for `movies/caption` made the caption text the id.
    """
    stem = write_cell(
        tmp_path,
        {"extractor": "word2psy",
         "models": {"beats": {"features": {"columns": ["beats_tempo"]}}}},
        {"beats_words.csv": ["word_idx", "chunk_label", "beats_tempo"]},
    )
    violations = campaign.family_violations(stem)
    assert "no_stimulus_id" in kinds(violations)


def test_aud2psy_flat_declaration_form_is_read(campaign, tmp_path):
    """aud2psy declares `models.<m>.columns`; viz2psy nests `features.columns`.

    Reading only the nested form left `declared` empty for every aud2psy
    cell, so even the forward (declared-but-unattributable) check was a no-op
    on a third of the store.
    """
    sidecar = {"extractor": "aud2psy",
               "models": {"beats": {"columns": ["beats_tempo", "beats_strength"]}}}
    declared, listed = campaign._declared_columns(sidecar)
    assert declared == {"beats_tempo", "beats_strength"}
    assert listed == {"beats"}


# ---------------------------------------------------------------------------
# The reverse direction, and what must NOT fire
# ---------------------------------------------------------------------------

def test_declared_but_never_emitted_is_caught(campaign, tmp_path):
    """A sidecar promising a column the family does not contain is as wrong
    as one omitting a column it does — the other side of the same filter."""
    stem = write_cell(
        tmp_path,
        {"extractor": "aud2psy",
         "models": {"beats": {"columns": ["beats_tempo", "beats_ghost"]}}},
        {"beats_beats.csv": ["stimulus_id", "beats_tempo"]},
    )
    violations = campaign.family_violations(stem)
    assert "not_emitted" in kinds(violations)
    assert any("beats_ghost" in cols for k, _, cols in violations
               if k == "not_emitted")


def test_embedding_models_do_not_trip_the_declaration_checks(campaign, tmp_path):
    """Embedding models declare a `pattern` + `count`, not a column list.

    Treating "no list" as "declares nothing" would fire `undeclared` on all
    1,024 EBind dimensions of every cell in the store.
    """
    stem = write_cell(
        tmp_path,
        {"extractor": "viz2psy",
         "models": {"beats": {"features": {"pattern": "beats_###", "count": 3}}}},
        {"beats.csv": ["stimulus_id", "beats_000", "beats_001", "beats_002"]},
    )
    assert campaign.family_violations(stem) == []


def test_aggregate_suffixes_count_as_emitted(campaign, tmp_path):
    """§4.1 lets a chunks table rename `x` to `x_mean` / `x_sd` / ..., which
    must not read as "declared column x was never written"."""
    stem = write_cell(
        tmp_path,
        {"extractor": "word2psy",
         "models": {"beats": {"features": {"columns": ["beats_tempo"]}}}},
        {"beats_chunks.csv": ["stimulus_id", "beats_tempo_mean", "beats_tempo_sd"]},
    )
    assert campaign.family_violations(stem) == []


def test_compliant_family_is_clean(campaign, tmp_path):
    stem = write_cell(
        tmp_path,
        {"extractor": "aud2psy",
         "models": {"beats": {"columns": ["time", "beats_is_downbeat"]}}},
        {"beats_beats.csv": ["stimulus_id", "time", "beats_is_downbeat"]},
    )
    assert campaign.family_violations(stem) == []


def test_family_glob_does_not_reach_a_sibling_cell(campaign, tmp_path):
    """`clip` must not pick up `clip_text`'s tables and check them against the
    wrong sidecar. Its own `.meta.json` is what marks it as a cell rather
    than a table. No such shadowing exists in the store today; the bare
    prefix glob is what would let it in unnoticed."""
    (tmp_path / "clip.meta.json").write_text(
        json.dumps({"extractor": "viz2psy",
                    "models": {"clip": {"features": {"pattern": "clip_###",
                                                     "count": 2}}}}))
    (tmp_path / "clip.csv").write_text("stimulus_id,clip_000,clip_001\na,0,0\n")
    (tmp_path / "clip_text.csv").write_text("stimulus_id,clip_text_000\na,0\n")
    (tmp_path / "clip_text.meta.json").write_text(
        json.dumps({"extractor": "word2psy", "models": {"clip_text": {}}}))
    tables = campaign._family_tables(tmp_path / "clip.csv")
    assert [t.name for t in tables] == ["clip.csv"]
