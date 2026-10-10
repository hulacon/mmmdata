"""glmsingle_operator_report.py end to end on toy GLMsingle pickles.

Type D is built here independently of the module under test — with the fracridge package and
GLMsingle's autoscale rule — so the validation gate is a real check: it must pass on a faithful fit and
refuse one whose type-D betas were tampered with.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("glmsingle")
fracridge = pytest.importorskip("fracridge").fracridge

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import glmsingle_operator_report as rep  # noqa: E402

from neuroimaging.glm import glmsingle_operator as go  # noqa: E402

TR, STIMDUR, N_RUNS, N_TIMES, PER_RUN = 1.5, 3.0, 3, 70, 9
SHAPE = (4, 3, 2)


def _toy_tree(tmp_path: Path, tamper: bool = False) -> tuple[Path, Path]:
    from glmsingle.design.convolve_design import convolve_design

    rng = np.random.default_rng(0)
    k = N_RUNS * PER_RUN
    items = np.concatenate([rng.permutation(PER_RUN) for _ in range(N_RUNS)])  # each item once per run
    designs = []
    for p in range(N_RUNS):
        d = np.zeros((N_TIMES, k))
        d[4 + 3 * np.arange(PER_RUN), np.arange(p * PER_RUN, (p + 1) * PER_RUN)] = 1
        designs.append(d)
    nv = int(np.prod(SHAPE))
    hrf_idx = np.where(np.arange(nv) % 2, 4, 9)
    pcreg = [rng.normal(size=(N_TIMES, 2)) for _ in range(N_RUNS)]
    pcnum = 1
    frac = np.where(np.arange(nv) % 3 == 0, 0.05, 0.4)
    meanvol = rng.uniform(800, 1200, size=nv)
    lib = go.hrf_library(STIMDUR, TR)
    proj = go.nuisance_projectors([N_TIMES] * N_RUNS, [2] * N_RUNS, [p[:, :pcnum] for p in pcreg])
    item_pat = rng.normal(size=(PER_RUN, nv))
    b_ols = np.zeros((nv, k))
    b_d = np.zeros((nv, k))
    scaleoffset = np.zeros((nv, 2))
    for v in range(nv):
        X = np.concatenate([P @ convolve_design(d, lib[:, hrf_idx[v]], {"n_times": N_TIMES, "tr": TR})
                            for d, P in zip(designs, proj)])
        y = X @ item_pat[items, v] + rng.normal(size=X.shape[0])
        ols = np.linalg.lstsq(X, y, rcond=None)[0]
        rr = fracridge(X, y[:, None], frac[v])[0][:, 0]
        A = np.c_[rr, np.ones(k)]
        h = np.linalg.lstsq(A, ols, rcond=None)[0]  # GLMsingle autoscale: match the unregularized fit
        b_ols[v], b_d[v], scaleoffset[v] = ols, A @ h, h
    if tamper:
        b_d = b_d[:, rng.permutation(k)]
    psc = lambda b: (b / meanvol[:, None] * 100).reshape(SHAPE + (k,)).astype(np.float32)  # noqa: E731
    gout = tmp_path / "tree" / "sub-01" / "enc" / "glmsingle_outputs"
    gout.mkdir(parents=True)
    vol = lambda x: np.asarray(x).reshape(SHAPE)  # noqa: E731
    np.save(gout / "DESIGNINFO.npy", {"designSINGLE": designs, "stimorder": items, "tr": TR, "stimdur": STIMDUR,
                                      "params": {"maxpolydeg": [2] * N_RUNS, "extra_regressors": [False]}})
    np.save(gout / "TYPEB_FITHRF.npy", {"HRFindex": vol(hrf_idx)})
    np.save(gout / "TYPEC_FITHRF_GLMDENOISE.npy", {"HRFindex": vol(hrf_idx), "betasmd": psc(b_ols)})
    np.save(gout / "TYPED_FITHRF_GLMDENOISE_RR.npy", {
        "HRFindex": vol(hrf_idx), "betasmd": psc(b_d), "FRACvalue": vol(frac).astype(np.float32),
        "R2": vol(rng.uniform(0, 30, nv)).astype(np.float32), "scaleoffset": scaleoffset.reshape(SHAPE + (2,)),
        "meanvol": vol(meanvol).astype(np.float32), "pcnum": pcnum, "pcregressors": pcreg})
    img = nib.Nifti1Image(np.ones(SHAPE, dtype=np.uint8), np.eye(4))
    ref = tmp_path / "ref.nii.gz"
    nib.save(img, ref)
    return tmp_path / "tree", ref


def test_report_passes_the_gate_and_writes_everything(tmp_path):
    tree, ref = _toy_tree(tmp_path)
    assert rep.main(["--subject", "01", "--arm", "enc", "--tree", str(tree), "--reference", str(ref),
                     "--mask", str(ref), "--rhos", "0,0.4"]) == 0
    out = tree / "sub-01" / "enc" / "operator"
    stem = "sub-01_arm-enc_space-MNI152NLin2009cAsym_res-2"
    report = json.loads((out / f"{stem}_desc-operator.json").read_text())
    assert report["validation"]["passed"] and report["validation"]["r_p01"] > 0.999
    assert abs(report["validation"]["frac_err_median"]) < 0.02
    prof = pd.read_csv(out / f"{stem}_desc-operatornull_profile.tsv", sep="\t")
    assert set(prof.type) == {"B", "C", "D"} and set(prof.rho) == {0.0, 0.4}
    lag1 = nib.load(out / f"{stem}_desc-leakagelag1_stat.nii.gz").get_fdata().ravel()
    frac = np.load(tree / "sub-01/enc/glmsingle_outputs/TYPED_FITHRF_GLMDENOISE_RR.npy",
                   allow_pickle=True).item()["FRACvalue"].ravel()
    assert np.nanmean(lag1[frac == 0.05]) > np.nanmean(lag1[frac == 0.4])  # more shrinkage, more mixing


def test_report_refuses_a_fit_its_operator_does_not_reproduce(tmp_path):
    tree, ref = _toy_tree(tmp_path, tamper=True)
    with pytest.raises(SystemExit, match="does not reproduce TYPED"):
        rep.main(["--subject", "01", "--arm", "enc", "--tree", str(tree), "--reference", str(ref),
                  "--mask", str(ref)])
    out = tree / "sub-01" / "enc" / "operator"
    assert (out / "sub-01_arm-enc_space-MNI152NLin2009cAsym_res-2_desc-operator.json").exists()
    assert not list(out.glob("*_stat.nii.gz"))
