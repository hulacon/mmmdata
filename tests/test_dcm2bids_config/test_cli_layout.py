"""The duckbrain layout is the same config at the path duckbrain reads.

duckbrain converts a session from ``<dcm2bids_config_dir>/sub-XX/ses-YY/
dcm2bids_config.json`` and reuses that file as-is, so the bridge must write
exactly what ``scripts/run_dcm2bids.py`` would have converted with. These pin
the path, the byte-equality of the two layouts, and that committed duckbrain
configs never drift from their mmmdata twins.
"""

import json
from pathlib import Path

import pytest

from src.python.core.config import load_config
from src.python.dcm2bids_config.cli import _resolve_output_path, generate_one


def test_duckbrain_layout_path():
    got = _resolve_output_path(Path("/cfg"), "sub-00", "ses-01", "duckbrain")
    assert got == Path("/cfg/sub-00/ses-01/dcm2bids_config.json")


def test_mmmdata_layout_path_unchanged():
    got = _resolve_output_path(Path("/cfg"), "sub-00", "ses-01")
    assert got == Path("/cfg/sub-00/ses-01_conf.json")


def test_unknown_layout_refuses():
    with pytest.raises(ValueError, match="unknown layout"):
        _resolve_output_path(Path("/cfg"), "sub-00", "ses-01", "heudiconv")


def test_both_layouts_write_identical_bytes(tmp_path):
    """ses-01 (anatomy, no fieldmaps by default) needs no DICOMs to generate."""
    src, ovr, db = tmp_path / "src", tmp_path / "ovr", tmp_path / "db"
    a = generate_one("sub-00", "ses-01", src, ovr)
    b = generate_one("sub-00", "ses-01", src, ovr, layout="duckbrain", out_dir=db)
    assert a["status"] == b["status"] == "written"
    assert Path(b["output_path"]) == db / "sub-00" / "ses-01" / "dcm2bids_config.json"
    assert Path(a["output_path"]).read_bytes() == Path(b["output_path"]).read_bytes()


def test_duckbrain_layout_reads_overrides_from_config_dir(tmp_path):
    """--out-dir moves the output only; overrides still come from config_dir."""
    ovr = tmp_path / "ovr"
    (ovr / "sub-00").mkdir(parents=True)
    (ovr / "sub-00" / "overrides.toml").write_text(
        '[ses-01]\nfmap_strategy = "series_number"\nfmap_groups = ["encoding"]\n'
        '[ses-01.fmap_series.encoding]\nap = 5\npa = 7\n'
    )
    r = generate_one("sub-00", "ses-01", tmp_path / "src", ovr,
                     layout="duckbrain", out_dir=tmp_path / "db")
    fmaps = [d for d in r["config"]["descriptions"] if d["datatype"] == "fmap"]
    assert len(fmaps) == 2


@pytest.mark.requires_dataset
def test_committed_duckbrain_configs_match_their_mmmdata_twins():
    """A duckbrain config is a generated copy, never a hand edit.

    Regenerate with ``--layout duckbrain --force`` after changing an override;
    this fails if only one layout was regenerated.
    """
    code_root = Path(load_config()["paths"]["code_root"])
    db_root = code_root / "config" / "dcm2bids_duckbrain"
    ovr_root = code_root / "config" / "dcm2bids_overrides"
    pairs = sorted(db_root.glob("sub-*/ses-*/dcm2bids_config.json"))
    for db_cfg in pairs:
        sub, ses = db_cfg.parent.parent.name, db_cfg.parent.name
        twin = ovr_root / sub / f"{ses}_conf.json"
        assert twin.exists(), f"{db_cfg} has no mmmdata-layout twin at {twin}"
        assert json.loads(db_cfg.read_text()) == json.loads(twin.read_text()), (
            f"{db_cfg} differs from {twin}; regenerate both layouts"
        )
