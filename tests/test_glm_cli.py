"""The runner end to end on a synthetic two-run fMRIPrep tree.

Dry run needs only the tree; the full fit needs nilearn and writes maps whose
names carry Contract A keys, plus the tree's dataset_description.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")

from neuroimaging.constants import MOTION_6  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import glm_contrast_maps  # noqa: E402

SPACE = "MNI152NLin2009cAsym_res-2"
TR = 1.5
N_SCANS = 200
SHAPE = (5, 5, 5)


def _events():
    rows, t = [], 0.0
    for _ in range(3):
        for c in ("hand", "foot", "mouth", "saccade", "rest"):
            rows.append({"onset": t, "duration": 20.0, "trial_type": c, "run_idx": 1})
            t += 20.0
    return pd.DataFrame(rows)


def _seed_run(root: Path, sub: str, ses: str, run: str, seed: int, events=True):
    prefix = f"sub-{sub}_ses-{ses}_task-motor_run-{run}"
    raw = root / f"sub-{sub}" / f"ses-{ses}" / "func"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{prefix}_bold.nii.gz").write_text("")
    (raw / f"{prefix}_bold.json").write_text(json.dumps({"RepetitionTime": TR, "TaskName": "motor"}))
    if events:
        _events().to_csv(raw / f"{prefix}_events.tsv", sep="\t", index=False)
    fp = root / "derivatives" / "fmriprep" / f"sub-{sub}" / f"ses-{ses}" / "func"
    fp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=100.0, scale=1.0, size=SHAPE + (N_SCANS,)).astype(np.float32)
    block = np.zeros(N_SCANS)
    for start in (0.0, 100.0, 200.0):  # hand blocks at 20 s each
        block[int(start / TR) : int((start + 20) / TR)] = 1.0
    data[0:2, 0:2, 0:2, :] += 3.0 * block[None, None, None, :]
    nib.Nifti1Image(data, np.eye(4)).to_filename(str(fp / f"{prefix}_space-{SPACE}_desc-preproc_bold.nii.gz"))
    nib.Nifti1Image(np.ones(SHAPE, dtype=np.uint8), np.eye(4)).to_filename(
        str(fp / f"{prefix}_space-{SPACE}_desc-brain_mask.nii.gz")
    )
    conf = pd.DataFrame(rng.normal(scale=0.05, size=(N_SCANS, 7)), columns=MOTION_6 + ["cosine00"])
    conf.to_csv(fp / f"{prefix}_desc-confounds_timeseries.tsv", sep="\t", index=False)


@pytest.fixture
def tree(tmp_path):
    _seed_run(tmp_path, "aa", "30", "01", 0)
    _seed_run(tmp_path, "aa", "30", "02", 1)
    return tmp_path


def test_dry_run_builds_designs_and_writes_nothing(tree, capsys):
    pytest.importorskip("nilearn")
    rc = glm_contrast_maps.main(["--subject", "sub-aa", "--model", "motor", "--dry-run",
                                 "--bids-root", str(tree)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "runs (2)" in out and "dry run" in out
    assert "design columns" in out
    assert not (tree / "derivatives" / "glm_localizer").exists()


def test_run_without_events_is_refused_before_any_fit(tree):
    _seed_run(tree, "aa", "30", "03", 2, events=False)
    with pytest.raises(SystemExit, match="without an events.tsv.*run-03"):
        glm_contrast_maps.main(["--subject", "aa", "--model", "motor", "--dry-run", "--bids-root", str(tree)])


def test_no_runs_is_a_named_error(tree):
    with pytest.raises(SystemExit, match="no fMRIPrep runs"):
        glm_contrast_maps.main(["--subject", "zz", "--model", "motor", "--dry-run", "--bids-root", str(tree)])


def test_full_fit_writes_contract_a_named_maps_and_description(tree):
    pytest.importorskip("nilearn")
    rc = glm_contrast_maps.main(["--subject", "sub-aa", "--model", "motor", "--bids-root", str(tree),
                                 "--smoothing-fwhm", "0", "--per-run-maps"])
    assert rc == 0
    base = tree / "derivatives" / "glm_localizer"
    desc = json.loads((base / "dataset_description.json").read_text())
    assert desc["DatasetType"] == "derivative"
    func = base / "sub-aa" / "ses-30" / "func"  # one session selected -> ses- kept
    fx = func / f"sub-aa_ses-30_task-motor_space-{SPACE}_contrast-handVsRest_stat-z_statmap.nii.gz"
    assert fx.exists(), sorted(p.name for p in func.iterdir())
    per_run = func / f"sub-aa_ses-30_task-motor_run-01_space-{SPACE}_contrast-handVsRest_stat-effect_statmap.nii.gz"
    assert per_run.exists()
    z = nib.load(str(fx)).get_fdata()
    assert z[0:2, 0:2, 0:2].mean() > 3.0
    assert abs(z[3:, 3:, 3:].mean()) < 1.5
    meta = json.loads((func / "sub-aa_task-motor_model-motor_run_metadata.json").read_text())
    assert meta["estimator"] == "nilearn" and len(meta["runs"]) == 2
    assert meta["config"]["space"] == SPACE


def test_split_design_guard_reaches_the_cli(tree):
    """task-motor across both session groups is refused unless opted in."""
    _seed_run(tree, "bb", "02", "01", 5)
    from neuroimaging.constants import MixedLocalizerDesignError

    # Subject bb has ses-02 only, subject aa ses-30 only; a subject filter never
    # spans both, so exercise the guard through find_fmriprep_runs directly.
    from neuroimaging.io import find_fmriprep_runs

    with pytest.raises(MixedLocalizerDesignError):
        find_fmriprep_runs(task="motor", bids_root=tree)
    assert glm_contrast_maps.main(["--subject", "bb", "--model", "motor", "--dry-run",
                                   "--bids-root", str(tree)]) == 0
