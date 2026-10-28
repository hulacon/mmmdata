"""glmsingle_tb.py --space: run discovery and BOLD paths follow the space, one voxel grid is enforced, and a non-default space cannot land in the default tree."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import glmsingle_tb as tb  # noqa: E402

SUB, SES, NVOL = "sub-99", "ses-04", 20


def _bold(path, shape=(3, 3, 3), affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(shape + (NVOL,), dtype=np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), str(path))


@pytest.fixture
def world(tmp_path):
    """Two TBencoding runs; both spaces for run-01, T1w only for run-02."""
    fp, bids = tmp_path / "fmriprep", tmp_path / "bids"
    for run in ("run-01", "run-02"):
        _bold(tb.bold_path(fp, SUB, SES, "TBencoding", run, "T1w"))
        ev = bids / SUB / SES / "func" / f"{SUB}_{SES}_task-TBencoding_{run}_events.tsv"
        ev.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"onset": [0.0, 6.0], "duration": 3.0, "trial_type": "image",
                      "mmmId": [1, 2]}).to_csv(ev, sep="\t", index=False)
    _bold(tb.bold_path(fp, SUB, SES, "TBencoding", "run-01"))
    return {"fp": fp, "bids": bids, "out": tmp_path / "out"}


def test_discovery_and_bold_paths_follow_the_space(world):
    assert "space-MNI152NLin2009cAsym_res-2_" in tb.bold_path(world["fp"], SUB, SES, "TBencoding", "run-01").name
    assert "space-T1w_desc-preproc" in tb.bold_path(world["fp"], SUB, SES, "TBencoding", "run-01", "T1w").name

    default = tb.discover_sessions(world["bids"], world["fp"], SUB, "enc", [SES])
    t1w = tb.discover_sessions(world["bids"], world["fp"], SUB, "enc", [SES], space="T1w")
    assert [r for _, typed in default for _, r, _ in typed] == ["run-01"]
    assert [r for _, typed in t1w for _, r, _ in typed] == ["run-01", "run-02"]

    m = tb.run(world["bids"], world["fp"], SUB, t1w, world["out"], "enc", 3.0,
               dry_run=True, space="T1w")
    assert m["space"] == "T1w" and m["n_volumes_per_run"] == [NVOL, NVOL]


def test_off_grid_run_stops_the_fit(world):
    shifted = np.eye(4)
    shifted[0, 3] = 1.8
    _bold(tb.bold_path(world["fp"], SUB, SES, "TBencoding", "run-02", "T1w"), affine=shifted)
    t1w = tb.discover_sessions(world["bids"], world["fp"], SUB, "enc", [SES], space="T1w")
    with pytest.raises(SystemExit):
        tb.run(world["bids"], world["fp"], SUB, t1w, world["out"], "enc", 3.0,
               dry_run=True, space="T1w")
    assert len(tb.off_grid_runs([tb.bold_path(world["fp"], SUB, SES, "TBencoding", r, "T1w")
                                 for r in ("run-01", "run-02")])) == 1


def test_non_default_space_needs_its_own_tree(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["glmsingle_tb.py", "--subject", SUB, "--arm", "enc",
                                      "--space", "T1w"])
    with pytest.raises(SystemExit, match="needs --output-base"):
        tb.main()
