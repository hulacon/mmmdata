"""glmsingle_export_hrf.py on a toy TYPEB pickle: eight NIfTIs, per-run margin and runner-up correct."""

import sys
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import glmsingle_export_hrf as ex  # noqa: E402

SHAPE, NRUNS, NH = (2, 3, 2), 3, 20


def _typeb(rng):
    fit_run = rng.uniform(0, 50, size=SHAPE + (NRUNS, NH)).astype(np.float32)
    fit_run[0, 0, 0, 1, :] = np.nan  # one voxel-run GLMsingle never fitted
    fit = fit_run.mean(axis=3)
    return {
        "FitHRFR2run": fit_run, "FitHRFR2": fit, "HRFindex": np.argmax(fit, axis=-1),
        "HRFindexrun": np.argmax(np.nan_to_num(fit_run, nan=-1), axis=-1),
        "R2": fit.max(axis=-1), "R2run": np.nanmax(fit_run, axis=-1),
        "meanvol": rng.uniform(100, 200, size=SHAPE).astype(np.float32),
        "betasmd": np.zeros(SHAPE + (5,), dtype=np.float32),
    }


def test_per_run_margin_is_best_minus_runner_up_and_nan_where_unfitted():
    rng = np.random.default_rng(0)
    fr = rng.uniform(0, 50, size=(4, NRUNS, NH))
    fr[0, 0, :] = np.nan
    best, second, margin = ex.per_run_margin(fr)
    srt = np.sort(fr, axis=-1)
    np.testing.assert_allclose(margin[1:], (srt[..., -1] - srt[..., -2])[1:], rtol=1e-6)
    assert np.isnan(margin[0, 0]) and best[0, 0] == 0
    assert (best != second)[1:].all()
    assert (np.take_along_axis(fr, second[..., None], -1)[..., 0] <= np.take_along_axis(fr, best[..., None], -1)[..., 0])[1:].all()


def test_export_writes_all_eight_and_only_the_missing_ones(tmp_path, capsys):
    rng = np.random.default_rng(1)
    d = _typeb(rng)
    arm = tmp_path / "glmsingle_tb" / "sub-zz" / "enc"
    (arm / "glmsingle_outputs").mkdir(parents=True)
    np.save(arm / "glmsingle_outputs" / "TYPEB_FITHRF.npy", d, allow_pickle=True)
    ref = tmp_path / "ref.nii.gz"
    nib.Nifti1Image(np.zeros(SHAPE, dtype=np.float32), np.diag([2, 2, 2, 1])).to_filename(str(ref))
    argv = ["--subject", "zz", "--arm", "enc", "--reference", str(ref), "--tree", str(tmp_path / "glmsingle_tb")]
    assert ex.main(argv) == 0
    out = arm / "hrf"
    stem = f"sub-zz_arm-enc_space-{ex.SPACE}"
    files = sorted(p.name for p in out.glob("*.nii.gz"))
    assert len(files) == 8
    idx_run = nib.load(str(out / f"{stem}_desc-hrfindexrun_dseg.nii.gz"))
    assert idx_run.shape == SHAPE + (NRUNS,) and idx_run.get_data_dtype() == np.int16
    np.testing.assert_array_equal(np.asarray(idx_run.dataobj), d["HRFindexrun"])
    margin = np.asarray(nib.load(str(out / f"{stem}_desc-hrfmarginrun_stat.nii.gz")).dataobj)
    second = np.asarray(nib.load(str(out / f"{stem}_desc-hrfsecondrun_dseg.nii.gz")).dataobj)
    srt = np.sort(d["FitHRFR2run"], axis=-1)
    np.testing.assert_allclose(margin[0, 1], (srt[..., -1] - srt[..., -2])[0, 1], rtol=1e-5)
    assert np.isnan(margin[0, 0, 0, 1])
    assert (second[0, 1] == np.argsort(d["FitHRFR2run"][0, 1], axis=-1)[..., -2]).all()
    assert np.allclose(nib.load(str(out / f"{stem}_desc-fithrfr2_stat.nii.gz")).affine, np.diag([2, 2, 2, 1]))
    # second call: nothing to do; after deleting one, only that one is rewritten
    assert ex.main(argv) == 0
    assert "all 8 exports exist" in capsys.readouterr().out
    (out / f"{stem}_desc-hrfr2run_stat.nii.gz").unlink()
    assert ex.main(argv) == 0
    assert "for ['R2run']" in capsys.readouterr().out
    assert len(list(out.glob("*.nii.gz"))) == 8


def test_grid_mismatch_is_refused(tmp_path):
    d = _typeb(np.random.default_rng(2))
    arm = tmp_path / "t" / "sub-zz" / "enc"
    (arm / "glmsingle_outputs").mkdir(parents=True)
    np.save(arm / "glmsingle_outputs" / "TYPEB_FITHRF.npy", d, allow_pickle=True)
    ref = tmp_path / "ref.nii.gz"
    nib.Nifti1Image(np.zeros((3, 3, 3), dtype=np.float32), np.eye(4)).to_filename(str(ref))
    with pytest.raises(SystemExit, match="does not match the reference"):
        ex.main(["--subject", "zz", "--arm", "enc", "--reference", str(ref), "--tree", str(tmp_path / "t")])


def test_mask_enables_the_masked_fithrfr2run_export(tmp_path):
    d = _typeb(np.random.default_rng(3))
    arm = tmp_path / "t" / "sub-zz" / "enc"
    (arm / "glmsingle_outputs").mkdir(parents=True)
    np.save(arm / "glmsingle_outputs" / "TYPEB_FITHRF.npy", d, allow_pickle=True)
    (arm / "run_metadata.json").write_text('{"run_labels": ["ses-01/run-01[enc]", "ses-01/run-02[enc]", "ses-02/run-01[enc]"]}')
    ref = tmp_path / "ref.nii.gz"
    nib.Nifti1Image(np.zeros(SHAPE, dtype=np.float32), np.eye(4)).to_filename(str(ref))
    m1 = np.ones(SHAPE, dtype=np.uint8); m1[0, 0, :] = 0
    m2 = np.ones(SHAPE, dtype=np.uint8); m2[1, 0, 0] = 0
    for i, m in enumerate((m1, m2)):
        nib.Nifti1Image(m, np.eye(4)).to_filename(str(tmp_path / f"m{i}.nii.gz"))
    argv = ["--subject", "zz", "--arm", "enc", "--reference", str(ref), "--tree", str(tmp_path / "t"),
            "--mask", str(tmp_path / "m0.nii.gz"), str(tmp_path / "m1.nii.gz")]
    assert ex.main(argv) == 0
    npz = np.load(arm / "hrf" / f"sub-zz_arm-enc_space-{ex.SPACE}_desc-fithrfr2run_masked.npz")
    inter = (m1 & m2).astype(bool)
    assert npz["data"].shape == (int(inter.sum()), NRUNS, NH) and npz["data"].dtype == np.float32
    np.testing.assert_array_equal(npz["index"], np.flatnonzero(inter))
    np.testing.assert_allclose(npz["data"], d["FitHRFR2run"][inter], rtol=0, atol=0)
    assert list(npz["run_labels"]) == ["ses-01/run-01[enc]", "ses-01/run-02[enc]", "ses-02/run-01[enc]"]
    # NIfTIs exist already, so a second call writes nothing; --force rewrites the npz too
    assert ex.main(argv) == 0
