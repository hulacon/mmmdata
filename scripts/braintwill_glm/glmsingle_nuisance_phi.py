#!/usr/bin/env python3
"""Per-run braintwill AR coefficients under a GLMsingle fit's own nuisance set.

``run_consistency.py --save-phi`` fits the noise model on residuals after the frozen reference
nuisance (motion, aCompCor, fMRIPrep cosines, non-steady-state spikes, intercept). A GLMsingle
fit's operator projects out something else: per-run polynomials of degree ``maxpolydeg`` and,
from type C on, the GLMdenoise noise components it selected (``pcregressors[:, :pcnum]``).
This driver fits the same noise model with the task side unchanged, so the nuisance set is
the only difference:

- **Runs:** the GLMsingle fit's runs, in its order (``run_metadata.json``), each checked
  against the fit's design length.
- **S:** the model's condition columns from ``adapter`` (the stand-in's design), as in
  ``run_consistency``.
- **N:** GLMsingle's polynomial matrix for the run plus its first ``pcnum`` noise components.
  No motion, aCompCor, cosine or spike columns.
- **Data:** the run masked by the runs' brain-mask intersection and mean-scaled, as in
  ``adapter``. A per-voxel affine rescaling does not change the fitted AR coefficients.

Writes ``<stem>_phi.npz`` with the keys ``run_consistency --save-phi`` writes (``phi``,
``runs``, ``mask_flat``, ``mask_shape``, ``affine``), plus ``pcnum`` and ``maxpolydeg``, and
``<stem>.json`` with per-run diagnostics. The output is a finding: write it outside this
public repository.

Usage (compute node; loading the type-D output needs ~12 GB per subject):
    .venv/bin/python scripts/braintwill_glm/glmsingle_nuisance_phi.py --subject sub-## \\
        --arm enc --model tbrepetition --out-dir /path/outside/this/repo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter import adapt, config_bids_root  # noqa: E402
from equivalence import derivatives_dir, parse_noise  # noqa: E402

from braintwill import RunBlockGLM, RunDesign  # noqa: E402

import mmmget  # noqa: E402

TYPED = "TYPED_FITHRF_GLMDENOISE_RR.npy"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True)
    p.add_argument("--arm", default="enc", help="GLMsingle fit arm under the tree")
    p.add_argument("--model", required=True, help="stats model whose condition columns form S")
    p.add_argument("--tree", default="glmsingle_tb", help="GLMsingle derivatives tree")
    p.add_argument("--space", default="MNI152NLin2009cAsym_res-2", help="must be the space the GLMsingle fit used")
    p.add_argument("--noise", default="ar3-bc")
    p.add_argument("--no-pcs", action="store_true", help="polynomials only (the type-B nuisance set)")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args(argv)


def glmsingle_nuisance(fit_dir: Path, with_pcs: bool) -> tuple[dict, list[int], list[int], list[np.ndarray | None], int]:
    """The fit's manifest, per-run lengths and polynomial degrees, and noise components (``None`` per
    run when ``with_pcs`` is false)."""
    gout = fit_dir / "glmsingle_outputs"
    meta_path = fit_dir / "run_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no GLMsingle manifest at {meta_path}")
    meta = json.loads(meta_path.read_text())
    info = np.load(gout / "DESIGNINFO.npy", allow_pickle=True).item()
    n_times = [int(d.shape[0]) for d in info["designSINGLE"]]
    maxpoly = [int(m) for m in np.ravel(info["params"]["maxpolydeg"])]
    if len(maxpoly) == 1:
        maxpoly = maxpoly * len(n_times)
    if not (len(n_times) == len(maxpoly) == meta["n_total_runs"]):
        raise ValueError(f"{fit_dir}: {len(n_times)} designs, {len(maxpoly)} poly degrees, "
                         f"{meta['n_total_runs']} runs in the manifest")
    pcnum, pcs = 0, [None] * len(n_times)
    if with_pcs:
        td = np.load(gout / TYPED, allow_pickle=True).item()
        pcnum = int(td["pcnum"])
        pcs = [np.asarray(p[:, :pcnum], dtype=float) for p in td["pcregressors"]]
        del td
        if len(pcs) != len(n_times) or any(p.shape[0] != n for p, n in zip(pcs, n_times)):
            raise ValueError(f"{fit_dir}: noise components do not match the run lengths")
    return meta, n_times, maxpoly, pcs, pcnum


def main(argv=None) -> int:
    from glmsingle.ols.make_poly_matrix import make_polynomial_matrix

    args = parse_args(argv)
    subject = args.subject.removeprefix("sub-")
    fit_dir = derivatives_dir() / args.tree / f"sub-{subject}" / args.arm
    t0 = time.time()
    meta, n_times, maxpoly, pcs, pcnum = glmsingle_nuisance(fit_dir, with_pcs=not args.no_pcs)
    if len(meta["tasks"]) != 1:
        raise SystemExit(f"ERROR: {fit_dir} pools tasks {meta['tasks']}; one task per fit is supported")
    task = meta["tasks"][0]
    # run_labels look like "ses-##/run-##[subgroup]"
    wanted = [tuple(x.removeprefix("ses-").removeprefix("run-") for x in lab.split("[")[0].split("/"))
              for lab in meta["run_labels"]]
    # include_excluded: the GLMsingle fit's recorded run list decides, so a run QC
    # has excluded since must still be found, not reported missing
    found = {(r.session, r.run): r for r in mmmget.runs(subject=subject, task=task, space=args.space,
                                                        bids_root=config_bids_root(), include_excluded=True)}
    missing = [w for w in wanted if w not in found]
    if missing:
        raise SystemExit(f"ERROR: GLMsingle runs not found now: {missing}")
    cell = adapt(subject, args.model, space=args.space, runs=[found[w] for w in wanted])
    noise = parse_noise(args.noise)
    nuis = "poly" if args.no_pcs else "poly+pcs"
    print(f"sub-{subject} {args.tree}/{args.arm} → {args.model} {args.space}: {len(cell.runs)} runs, "
          f"{int(cell.mask.sum())} mask voxels, {noise.name}, N = {nuis} (pcnum {pcnum})", flush=True)

    runs_out, phis = [], []
    for i, ar in enumerate(cell.runs):
        t1 = time.time()
        Y = cell.load_data(i)
        if Y.shape[0] != n_times[i]:
            raise ValueError(f"{ar.run.entity_prefix}: {Y.shape[0]} volumes, GLMsingle design has {n_times[i]}")
        N = make_polynomial_matrix(n_times[i], maxpoly[i])
        names = [f"poly{d}" for d in range(N.shape[1])]
        if pcs[i] is not None and pcs[i].size:
            N = np.c_[N, pcs[i]]
            names += [f"pc{j}" for j in range(pcs[i].shape[1])]
        design = RunDesign(S=ar.design.S, N=N, nuisance=tuple(names))
        fit = RunBlockGLM(cell.axis, [design], noise=noise).fit([Y])
        phi = np.array(fit.fits[0].phi, dtype=np.float32)
        phi[~fit.fits[0].ok] = np.nan
        phis.append(phi)
        rec = {"run": ar.run.entity_prefix, "n_scans": int(Y.shape[0]), "nuisance_columns": len(names),
               "ok_fraction": float(fit.ok.mean()), "masked": int(fit.masked_per_run()[0]),
               "phi_median": np.nanmedian(fit.fits[0].phi, axis=0).round(4).tolist(),
               "seconds": round(time.time() - t1, 1)}
        runs_out.append(rec)
        print(f"  {ar.run.entity_prefix}: N {len(names)} cols, ok {rec['ok_fraction']:.4f} "
              f"({rec['masked']} masked), φ̃ {rec['phi_median']}, {rec['seconds']:.0f} s", flush=True)
        del Y

    stem = f"sub-{subject}_model-{args.model}_space-{args.space}_noise-{noise.name}_nuisance-glmsingle-{args.arm}"
    if args.no_pcs:
        stem += "-poly"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / f"{stem}_phi.npz", phi=np.stack(phis),
             runs=np.array([ar.run.entity_prefix for ar in cell.runs]),
             mask_flat=np.flatnonzero(cell.mask.ravel()), mask_shape=np.array(cell.mask.shape),
             affine=cell.mask_img.affine, pcnum=pcnum, maxpolydeg=np.array(maxpoly))
    record = {"subject": subject, "model": args.model, "space": args.space, "noise": noise.name,
              "glmsingle_fit": str(fit_dir), "nuisance": nuis, "pcnum": pcnum, "maxpolydeg": sorted(set(maxpoly)),
              "mask_voxels": int(cell.mask.sum()), "runs": runs_out, "seconds": round(time.time() - t0, 1)}
    (args.out_dir / f"{stem}.json").write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out_dir / stem}.json (+ _phi.npz), {time.time() - t0:.0f} s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
