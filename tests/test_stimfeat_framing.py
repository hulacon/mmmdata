"""The campaign passes the reviewed framing table to every viz2psy video cell,
and counts such a cell done only if it was framed with that table.

Offline: no extractor runs. The driver is imported as a module, as in
test_stimfeat_verify.py.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stimfeat_campaign.py"


@pytest.fixture
def campaign(monkeypatch):
    spec = importlib.util.spec_from_file_location("stimfeat_campaign_framing", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def table(tmp_path, monkeypatch):
    t = tmp_path / "framing_table.csv"
    t.write_text("path\n")
    (tmp_path / "framing_table.csv.json").write_text(json.dumps({"sha256": "c" * 64}))
    monkeypatch.setenv("STIMFEAT_FRAMING_TABLE", str(t))
    return t


def _cells(campaign, tmp_path):
    video = campaign.Unit(id="film-a", out_dir=tmp_path / "movies" / "film-a",
                          inputs=[str(tmp_path / "Film_A.mov")], extra=["--stimulus-id", "film-a"])
    frames = campaign.Source("movies", "frames", "viz2psy", "", True, "films", lambda: [video])
    images = campaign.Unit(id="all", out_dir=tmp_path / "movie_cues", inputs=["a.png", "b.png"])
    cues = campaign.Source("movies", "cue", "viz2psy", "", True, "cues", lambda: [images])
    return frames, video, cues, images


def test_video_cells_pass_the_table(campaign, table, tmp_path):
    frames, video, cues, images = _cells(campaign, tmp_path)
    cmd = campaign.command_for(frames, video, "clip")
    assert cmd[cmd.index("--framing-table") + 1] == str(table)
    assert "--framing-table" not in campaign.command_for(cues, images, "clip")


def test_empty_override_runs_without_a_table(campaign, tmp_path, monkeypatch):
    monkeypatch.setenv("STIMFEAT_FRAMING_TABLE", "")
    frames, video, _, _ = _cells(campaign, tmp_path)
    assert "--framing-table" not in campaign.command_for(frames, video, "clip")


def test_default_table_is_configured(campaign, monkeypatch):
    monkeypatch.delenv("STIMFEAT_FRAMING_TABLE", raising=False)
    assert campaign.framing_table() is not None and campaign.framing_table().name.endswith(".csv")


def _sidecar(stem, extractor="viz2psy", type_="video", sha=None):
    framing = {"frame_source": "raw decode"}
    if sha:
        framing["framing_table_sha256"] = sha
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".meta.json").write_text(json.dumps(
        {"extractor": extractor, "input": {"type": type_, "framing": framing}}))


def test_video_cell_done_only_under_the_current_table(campaign, table, tmp_path):
    stem = tmp_path / "movies" / "film-a" / "clip.csv"
    assert not campaign.is_done(stem)
    _sidecar(stem)  # extracted before the framing table
    assert not campaign.is_done(stem)
    _sidecar(stem, sha="d" * 64)  # an older table
    assert not campaign.is_done(stem)
    _sidecar(stem, sha="c" * 64)
    assert campaign.is_done(stem)


def test_other_cells_need_only_a_sidecar(campaign, table, tmp_path):
    stem = tmp_path / "movies" / "film-a" / "egemaps.csv"
    _sidecar(stem, extractor="aud2psy")  # a soundtrack: no framing
    assert campaign.is_done(stem)
    stem = tmp_path / "movie_cues" / "clip.csv"
    _sidecar(stem, type_="image_folder")
    assert campaign.is_done(stem)
