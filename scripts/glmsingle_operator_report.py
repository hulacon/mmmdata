#!/usr/bin/env python3
"""Rebuild a GLMsingle fit's per-voxel operators and write what they imply, beside the fit.

For one subject x arm of a GLMsingle tree, using only the fit's own outputs (no refitting; see
``neuroimaging.glm.glmsingle_operator``), writes under ``<arm>/operator/``:

    sub-##_arm-<arm>_space-<space>_desc-ridgelambda_stat.nii.gz   type-D ridge penalty per voxel
    sub-##_arm-<arm>_space-<space>_desc-leakagelag1_stat.nii.gz   type-D mean R[i,i+1] / mean R[i,i]
    sub-##_arm-<arm>_space-<space>_desc-leakagelag2_stat.nii.gz   the same two trials apart
    sub-##_arm-<arm>_space-<space>_desc-operatornull_profile.tsv  predicted pattern r by lag, per type
    sub-##_arm-<arm>_space-<space>_desc-operator.json             provenance, validation, summaries

The null profile is what a null with lag-independent item signal plus AR(1) temporal noise predicts
for pooled pattern similarity between trials ``l`` apart (within run, each run's mean pattern removed),
for types B, C and D, at each ``--rhos`` value. Signal and noise variances come from the type-C betas:
items shown in more than one run give the signal variance from their cross-run covariance. It is
computed for two voxel sets: the top decile of TYPED R2, and a fixed random sample of the mask.

**Validation gate.** The rebuilt type-D operator, applied to the type-C betas, must reproduce the stored
type-D betas once their autoscale is undone (1st-percentile r >= ``--min-r`` over a voxel sample).
Otherwise nothing but the JSON is written and the exit status is non-zero.

Loads the type B, C and D pickles in turn (each several GB for a long fit): run under sbatch
(``glmsingle_operator_report.sbatch``). Existing outputs are kept unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SPACE = "MNI152NLin2009cAsym_res-2"
MEANVOL_FRACTION = 0.25
NULL_SAMPLE = 25000
VALIDATE_N = 2000


def _code_version() -> dict:
    repo = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        head, dirty = "unknown", None
    return {"head": head, "dirty": dirty}


def _load(path: Path) -> dict:
    return np.load(path, allow_pickle=True).item()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--reference", required=True, help="NIfTI on the fit's grid, supplies shape + affine")
    ap.add_argument("--mask", nargs="+", required=True, help="brain mask NIfTI(s) on the fit grid, intersected")
    ap.add_argument("--tree", default=None, help="GLMsingle tree; default <output_dir>/glmsingle_tb")
    ap.add_argument("--space", default=SPACE)
    ap.add_argument("--rhos", default="0,0.3,0.6", help="AR(1) coefficients bracketing the noise model")
    ap.add_argument("--min-r", type=float, default=0.99)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    import nibabel as nib
    import pandas as pd

    from neuroimaging.glm import glmsingle_operator as go

    if args.tree is None:
        from core.config import load_config
        tree = Path(load_config()["paths"]["output_dir"]) / "glmsingle_tb"
    else:
        tree = Path(args.tree)
    sub = args.subject if args.subject.startswith("sub-") else f"sub-{args.subject}"
    arm_dir = tree / sub / args.arm
    gout = arm_dir / "glmsingle_outputs"
    out_dir = arm_dir / "operator"
    stem = f"{sub}_arm-{args.arm}_space-{args.space}"
    paths = {k: out_dir / f"{stem}_desc-{k}_stat.nii.gz" for k in ("ridgelambda", "leakagelag1", "leakagelag2")}
    profile_path = out_dir / f"{stem}_desc-operatornull_profile.tsv"
    json_path = out_dir / f"{stem}_desc-operator.json"
    if not args.force and json_path.exists() and profile_path.exists() and all(p.exists() for p in paths.values()):
        print(f"{sub}/{args.arm}: operator report exists; --force to rewrite")
        return 0
    for name in ("DESIGNINFO.npy", "TYPEB_FITHRF.npy", "TYPEC_FITHRF_GLMDENOISE.npy",
                 "TYPED_FITHRF_GLMDENOISE_RR.npy"):
        if not (gout / name).exists():
            sys.exit(f"ERROR: {gout / name} missing")
    rhos = tuple(float(x) for x in args.rhos.split(","))

    ref = nib.load(args.reference)
    shape = tuple(ref.shape[:3])
    mask = np.ones(shape, dtype=bool)
    for mp in args.mask:
        m = np.asarray(nib.load(mp).dataobj).astype(bool)
        if m.shape != shape:
            sys.exit(f"ERROR: mask {mp} shape {m.shape} does not match the reference grid {shape}")
        mask &= m

    # -- design ------------------------------------------------------------------------------
    info = _load(gout / "DESIGNINFO.npy")
    tr, stimdur, designs = float(info["tr"]), float(info["stimdur"]), info["designSINGLE"]
    params = info["params"]
    maxpoly = params.get("maxpolydeg")
    if maxpoly is None or np.isscalar(maxpoly):
        maxpoly = [int(round(d.shape[0] * tr / 60 / 2)) if maxpoly is None else int(maxpoly) for d in designs]
    extra = params.get("extra_regressors", False)
    if not (extra is None or extra is False
            or all(e is None or e is False or np.size(e) == 0 for e in extra)):
        sys.exit("ERROR: the fit used extra_regressors; the operator rebuild does not include them yet")
    items = np.asarray(info["stimorder"], dtype=int)
    n_trials = items.size
    run_of = np.full(n_trials, -1)
    for p, d in enumerate(designs):
        run_of[go.trial_order(d)] = p
    if np.any(run_of < 0):
        sys.exit("ERROR: some trials are in no run's design")

    # -- type D selections, plus a validation sample of its betas ------------------------------------
    td = _load(gout / "TYPED_FITHRF_GLMDENOISE_RR.npy")
    if td["meanvol"].shape != shape:
        sys.exit(f"ERROR: fit grid {td['meanvol'].shape} does not match the reference grid {shape}")
    mv = td["meanvol"].astype(float)
    in_mask = mask & np.isfinite(mv)
    floor = MEANVOL_FRACTION * np.median(mv[in_mask])
    idx = np.flatnonzero(in_mask.ravel())
    frac = td["FRACvalue"].ravel()[idx].astype(float)
    hrf_d = td["HRFindex"].ravel()[idx].astype(int)
    r2 = td["R2"].ravel()[idx].astype(float)
    so = td["scaleoffset"].reshape(-1, 2)[idx].astype(float)
    above_floor = mv.ravel()[idx] >= floor
    pcnum = int(td["pcnum"])
    pcreg = [p[:, :pcnum] for p in td["pcregressors"]]
    rng = np.random.default_rng(0)
    val = np.sort(rng.choice(idx.size, size=min(VALIDATE_N, idx.size), replace=False))
    typed_val = td["betasmd"].reshape(-1, n_trials)[idx[val]].astype(float)
    del td
    tb = _load(gout / "TYPEB_FITHRF.npy")
    hrf_agree_b = float(np.mean(tb["HRFindex"].ravel()[idx] == hrf_d))
    del tb
    tc = _load(gout / "TYPEC_FITHRF_GLMDENOISE.npy")
    hrf_agree_c = float(np.mean(tc["HRFindex"].ravel()[idx] == hrf_d))
    bc = tc["betasmd"].reshape(-1, n_trials)[idx].astype(np.float64)
    del tc
    good = np.all(np.isfinite(bc), axis=1) & (np.median(np.abs(bc), axis=1) <= 100) & (so[:, 0] > 0)

    # -- per-HRF operators ------------------------------------------------------------------------
    lib = go.hrf_library(stimdur, tr)
    proj_b = go.nuisance_projectors([d.shape[0] for d in designs], maxpoly)
    proj_c = go.nuisance_projectors([d.shape[0] for d in designs], maxpoly, pcreg)
    lam = np.full(idx.size, np.nan)
    leak = np.full((idx.size, 2), np.nan)
    rebuilt = np.full((val.size, n_trials), np.nan)
    top = good & above_floor & (r2 >= np.nanpercentile(r2[good & above_floor], 90))
    sample_pool = np.flatnonzero(good & above_floor)
    sample = np.zeros(idx.size, dtype=bool)
    sample[rng.choice(sample_pool, size=min(NULL_SAMPLE, sample_pool.size), replace=False)] = True
    null_sets = {"top_r2_decile": top, f"mask_sample_{NULL_SAMPLE}": sample}
    need = top | sample
    lags = go.DEFAULT_LAGS
    sums = {T: {"signal": np.zeros((idx.size, len(lags))),
                "noise": {r: np.zeros((idx.size, len(lags))) for r in rhos}} for T in "BCD"}
    n_pairs, n_tr = None, None
    for hh in np.unique(hrf_d[good]):
        sel = np.flatnonzero((hrf_d == hh) & good)
        bases_c = go.run_bases(designs, lib[:, hh], proj_c, tr)
        lam[sel] = go.solve_lambda(bases_c, bc[sel], frac[sel])
        leak[sel] = go.leakage(bases_c, lam[sel], lags=(1, 2))
        hit = np.flatnonzero(np.isin(sel, val))
        if hit.size:
            rebuilt[np.searchsorted(val, sel[hit])] = go.ridge_betas(bases_c, bc[sel[hit]], lam[sel[hit]])
        nsel = sel[need[sel]]
        if nsel.size:
            bases_b = go.run_bases(designs, lib[:, hh], proj_b, tr)
            for T, bases, lam_t in (("B", bases_b, None), ("C", bases_c, None), ("D", bases_c, lam[nsel])):
                s = go.lag_sums(bases, lam_t, rhos=rhos, lags=lags)
                sums[T]["signal"][nsel] = s.signal
                for r in rhos:
                    sums[T]["noise"][r][nsel] = s.noise[r]
                n_pairs, n_tr = s.n_pairs, s.n_trials
        print(f"{sub}/{args.arm} hrf {hh}: {sel.size} voxels", flush=True)

    # -- validation gate ----------------------------------------------------------------------------
    mvv = np.abs(mv.ravel()[idx[val]])
    raw_c = bc[val] * (mvv / 100.0)[:, None]
    unscaled = (typed_val * (mvv / 100.0)[:, None] - so[val, 1:2]) / so[val, 0:1]
    ok = good[val] & np.all(np.isfinite(rebuilt), axis=1)
    rebuilt_raw = rebuilt * (mvv / 100.0)[:, None]
    r = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(rebuilt_raw[ok], unscaled[ok])])
    frac_err = np.linalg.norm(unscaled[ok], axis=1) / np.linalg.norm(raw_c[ok], axis=1) - frac[val][ok]
    validation = {"n": int(ok.sum()), "r_p01": float(np.percentile(r, 1)), "r_median": float(np.median(r)),
                  "frac_err_median": float(np.median(frac_err)),
                  "frac_err_abs_p99": float(np.percentile(np.abs(frac_err), 99)),
                  "passed": bool(np.percentile(r, 1) >= args.min_r)}

    # -- null profiles ------------------------------------------------------------------------------
    centred = go.run_center(bc, run_of)
    counts = np.bincount(items)
    def cross_run_pairs(keep_item):
        pairs = []
        for it in np.flatnonzero(keep_item):
            ix = np.flatnonzero(items == it)
            pairs += [(a, b) for k, a in enumerate(ix) for b in ix[k + 1:] if run_of[a] != run_of[b]]
        return np.array(pairs, dtype=int).reshape(-1, 2)
    tau2_source = "items with 2-3 exposures"
    pairs = cross_run_pairs((counts >= 2) & (counts <= 3))
    if len(pairs) < 200:
        tau2_source = "all repeated items, including many-exposure items (fewer than 200 pairs otherwise)"
        pairs = cross_run_pairs(counts >= 2)
    rows = []
    for set_name, vs in null_sets.items():
        v = np.flatnonzero(vs)
        for rho in rhos:
            unit = sums["C"]["noise"][rho][v, lags.index(0)] / n_tr
            sdiag = sums["C"]["signal"][v, lags.index(0)] / n_tr
            tau2, sigma2 = go.signal_and_noise_variance(centred[v], pairs, unit, sdiag)
            for T in "BCD":
                s = go.LagSums(lags=lags, signal=sums[T]["signal"][v],
                               noise={rho: sums[T]["noise"][rho][v]}, n_pairs=n_pairs, n_trials=n_tr)
                weight = so[v, 0] ** 2 if T == "D" else None
                pred, noise_only = go.null_profile(s, rho, tau2, sigma2, weight)
                for l, pr, no in zip([l for l in lags if l > 0], pred, noise_only):
                    rows.append(dict(voxel_set=set_name, type=T, rho=rho, lag=l, predicted_r=pr,
                                     noise_part=no, n_voxels=int(v.size)))

    def summarize(m):
        x = leak[m, 0]
        return {"n": int(m.sum()), "lag1_median": float(np.median(x)), "lag1_p25": float(np.percentile(x, 25)),
                "lag1_p75": float(np.percentile(x, 75)), "lag2_median": float(np.median(leak[m, 1])),
                "frac_median": float(np.median(frac[m])), "share_frac_at_min": float(np.mean(frac[m] <= frac[m].min()))}
    report = {
        "subject": sub, "arm": args.arm, "space": args.space, "tree": str(tree), "reference": args.reference,
        "masks": args.mask, "meanvol_floor": {"fraction_of_median": MEANVOL_FRACTION, "value": float(floor)},
        "code_version": _code_version(), "glmsingle_params": {"pcnum": pcnum, "maxpolydeg": [int(m) for m in maxpoly],
                                                               "n_trials": int(n_trials), "n_runs": len(designs)},
        "hrf_agreement_with_typed": {"B": hrf_agree_b, "C": hrf_agree_c},
        "validation": validation, "tau2_source": tau2_source, "n_tau2_pairs": int(len(pairs)), "rhos": list(rhos),
        "leakage": {"mask_above_floor": summarize(good & above_floor), "top_r2_decile": summarize(top)},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2))
    if not validation["passed"]:
        print(json.dumps(validation, indent=2))
        sys.exit(f"ERROR: rebuilt operator does not reproduce TYPED (r p01 {validation['r_p01']:.4f} < "
                 f"{args.min_r}); maps not written")
    pd.DataFrame(rows).to_csv(profile_path, sep="\t", index=False, float_format="%.6g")
    for key, values in (("ridgelambda", lam), ("leakagelag1", leak[:, 0]), ("leakagelag2", leak[:, 1])):
        vol = np.full(int(np.prod(shape)), np.nan, dtype=np.float32)
        vol[idx] = values
        nib.save(nib.Nifti1Image(vol.reshape(shape), ref.affine), paths[key])
    print(json.dumps({k: report[k] for k in ("validation", "leakage")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
