"""release.stage: the stage guard, private-output rules, and safe replacement."""

import datetime as dt
import os
from pathlib import Path

import pytest

from release.stage import (
    Stage, StageError, atomic_write_text, check_private_output, check_stage,
    load_release_config, parse_epoch,
)
from tests.release_fixtures import make_env, write_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_guard_accepts_a_staged_copy(tmp_path):
    cfg, _, stage, _ = make_env(tmp_path)
    assert check_stage(stage, cfg) == stage.resolve()


def test_guard_refuses_the_bids_root(tmp_path):
    cfg, work, _, _ = make_env(tmp_path)
    with pytest.raises(StageError, match="BIDS root") as exc:
        check_stage(work, cfg)
    assert str(work) in str(exc.value)


def test_guard_refuses_a_path_inside_the_bids_root(tmp_path):
    cfg, work, _, _ = make_env(tmp_path)
    with pytest.raises(StageError, match="BIDS root"):
        check_stage(work / "derivatives" / "fmriprep", cfg)


def test_guard_refuses_the_source_dir(tmp_path):
    cfg, _, _, _ = make_env(tmp_path)
    (cfg.source_dir / "dataset_description.json").write_text("{}")
    with pytest.raises(StageError, match="source-data") as exc:
        check_stage(cfg.source_dir, cfg)
    assert str(cfg.source_dir) in str(exc.value)


def test_guard_refuses_a_directory_containing_the_bids_root(tmp_path):
    cfg, _, _, _ = make_env(tmp_path)
    (tmp_path / "dataset_description.json").write_text("{}")
    with pytest.raises(StageError, match="contains"):
        check_stage(tmp_path, cfg)


def test_guard_refuses_a_symlink_to_the_bids_root(tmp_path):
    cfg, work, _, _ = make_env(tmp_path)
    link = tmp_path / "looks_like_a_stage"
    link.symlink_to(work)
    with pytest.raises(StageError, match="BIDS root"):
        check_stage(link, cfg)


def test_guard_refuses_a_non_bids_directory(tmp_path):
    cfg, _, _, _ = make_env(tmp_path)
    other = tmp_path / "elsewhere"
    other.mkdir()
    with pytest.raises(StageError, match="dataset_description"):
        check_stage(other, cfg)


def test_guard_reads_roots_from_config(tmp_path):
    cfg, work, stage, _ = make_env(tmp_path)
    loaded = load_release_config(write_config(tmp_path, cfg))
    assert loaded.bids_root == work and loaded.source_dir == cfg.source_dir
    with pytest.raises(StageError):
        Stage(work, loaded)
    with pytest.raises(StageError):
        Stage(cfg.source_dir, loaded)
    Stage(stage, loaded)                                  # accepted


def test_repo_config_has_release_epoch_and_guard_refuses_configured_root():
    cfg = load_release_config(REPO_ROOT / "config")
    assert cfg.epoch == dt.date(1900, 1, 1)
    # path relations are checked before anything touches the filesystem, so
    # this holds off-cluster too
    with pytest.raises(StageError, match="BIDS root"):
        check_stage(cfg.bids_root, cfg)
    with pytest.raises(StageError, match="source-data"):
        check_stage(cfg.source_dir, cfg)


def test_epoch_defaults_when_release_section_absent(tmp_path):
    cfg, _, _, _ = make_env(tmp_path)
    loaded = load_release_config(write_config(tmp_path, cfg, epoch=None))
    assert loaded.epoch == dt.date(1900, 1, 1)


@pytest.mark.parametrize("bad", ["1990-01-01", "1900/01/01", "soon"])
def test_epoch_must_be_iso_and_at_most_1925(bad):
    with pytest.raises(ValueError):
        parse_epoch(bad)


def test_private_output_refused_inside_stage_or_bids_root_or_if_present(tmp_path):
    cfg, work, stage, private = make_env(tmp_path)
    with pytest.raises(StageError, match="inside the staged copy"):
        check_private_output(stage / "offsets.tsv", stage, cfg)
    with pytest.raises(StageError, match="inside the BIDS root"):
        check_private_output(work / "offsets.tsv", stage, cfg)
    (private / "old.tsv").write_text("x")
    with pytest.raises(StageError, match="already exists"):
        check_private_output(private / "old.tsv", stage, cfg)
    assert check_private_output(private / "new.tsv", stage, cfg) == (private / "new.tsv").resolve()


def test_atomic_write_leaves_a_hardlinked_twin_untouched(tmp_path):
    original = tmp_path / "work.tsv"
    original.write_text("a\tb\n1\t2\n")
    os.chmod(original, 0o640)
    twin = tmp_path / "stage.tsv"
    os.link(original, twin)
    atomic_write_text(twin, "a\tb\n9\t9\n")
    assert original.read_text() == "a\tb\n1\t2\n"
    assert twin.read_text() == "a\tb\n9\t9\n"
    assert original.stat().st_ino != twin.stat().st_ino
    assert twin.stat().st_mode & 0o777 == 0o640
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_stage_refuses_writes_through_a_symlinked_directory(tmp_path):
    cfg, work, stage, _ = make_env(tmp_path, hardlink=False)
    os.symlink(work / "phenotype", stage / "linked")
    s = Stage(stage, cfg)
    with pytest.raises(StageError, match="outside the stage"):
        s.queue_write("linked/final_debriefing.tsv", "x\n")
    with pytest.raises(StageError, match="outside the stage"):
        s.queue_write("../escape.tsv", "x\n")


def test_read_only_stage_cannot_queue_writes(tmp_path):
    cfg, work, _, _ = make_env(tmp_path)
    s = Stage(work, cfg, writable=False)          # the working tree, read-only
    assert s.exists("README")
    with pytest.raises(StageError, match="read-only"):
        s.queue_write("README", "x")
