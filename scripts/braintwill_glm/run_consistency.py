#!/usr/bin/env python3
"""Per-run contrast effects and variances on task data: the input to Part R's between-run test.

The glm-strategy program's Part R test R-2 asks whether a contrast's per-run estimates scatter
around their fixed effect by more than their own variances allow (Cochran's Q). This script does
the fitting half. For one stand-in cell (subject × model × space × session pool, read from its
``*_run_metadata.json`` exactly as ``equivalence.py`` does), every run is designed through
``adapter`` and fitted on its own with braintwill under ``--noise`` (default ``ar3-bc``, the
P-Stage 1 default). Each contrast the run can estimate gets its effect and variance, variance NaN
where that voxel's noise fit was masked.

Writes ``<stem>_runs.npz`` (per contrast: ``<name>_effect`` and ``<name>_variance``, each
(runs, voxels) float32, and ``<name>_runs``, the run labels in the stand-in's order) and
``<stem>.json`` (run list, mask size, per-run noise-fit diagnostics). Scoring is
``score_consistency.py``. The output is a finding; write it outside this public repository.

``--save-phi`` also writes ``<stem>_phi.npz``: each run's fitted AR coefficients ``phi``
(runs, voxels, order) float32, NaN where the fit was masked, with ``runs`` (labels),
``mask_flat`` (the mask's C-order flat voxel indices, i.e. the columns' order), ``mask_shape``
and ``affine``. That is the noise model itself, for consumers that need Σ rather than a
contrast's variance (Part R's R-3).

``--censor-fd MM`` censors volumes with framewise displacement above MM through braintwill's
``retain``; ``--confounds [COL ...]`` replaces the regime's named confound columns (none when
given bare: aCompCor, cosines and non-steady-state spikes stay). Both are recorded in the JSON,
with each run's censored count. Default: the stand-in's design, unchanged.

Usage (compute node — a T1w cell loads every run's BOLD):
    .venv/bin/python scripts/braintwill_glm/run_consistency.py --subject sub-## --model floc \\
        --sessions ses-## --out-dir /path/outside/this/repo
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
from equivalence import derivatives_dir, parse_noise, stand_in  # noqa: E402

from braintwill import RunBlockGLM  # noqa: E402

import mmmget  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--sessions", nargs="*", default=None,
                   help="the stand-in's session pool: one session, or omit for the cross-session pool")
    p.add_argument("--space", default="T1w")
    p.add_argument("--tree", default="nilearn_glm")
    p.add_argument("--desc", default="referenceOLS")
    p.add_argument("--regime", default="reference")
    p.add_argument("--noise", default="ar3-bc")
    p.add_argument("--save-phi", action="store_true", help="also write each run's fitted AR coefficients")
    p.add_argument("--censor-fd", type=float, default=None, help="censor volumes with FD above this (mm)")
    p.add_argument("--confounds", nargs="*", default=None,
                   help="replace the regime's named confound columns (bare flag: none)")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    from neuroimaging.glm.models import load_model

    args = parse_args(argv)
    subject = args.subject.removeprefix("sub-")
    sessions = [s.removeprefix("ses-") for s in args.sessions] if args.sessions else None
    if sessions and len(sessions) > 1:
        raise SystemExit("ERROR: a stand-in session pool is one session, or the whole subject")
    model = load_model(args.model)
    tree = derivatives_dir() / args.tree
    meta, _ = stand_in(tree, subject, model, args.space, args.desc, sessions[0] if sessions else None)
    wanted = [(r["session"], r["run"]) for r in meta["runs"]]
    # include_excluded: the stand-in's recorded run list decides, so a run QC has
    # excluded since must still be found, not reported missing
    found = {(r.session, r.run): r for r in mmmget.runs(subject=subject, task=model.task,
                                                        variant=meta["config"]["variant"],
                                                        space=args.space, bids_root=config_bids_root(),
                                                        include_excluded=True)}
    missing = [w for w in wanted if w not in found]
    if missing:
        raise SystemExit(f"ERROR: stand-in runs not found now: {missing}")
    t0 = time.time()
    cell = adapt(subject, args.model, space=args.space, regime=args.regime, runs=[found[w] for w in wanted],
                 confounds=args.confounds, censor_fd=args.censor_fd)
    noise = parse_noise(args.noise)
    n_vox = int(cell.mask.sum())
    print(f"sub-{subject} {model.name} {args.space}: {len(cell.runs)} runs, {n_vox} mask voxels, {noise.name}")

    per_contrast: dict[str, dict[str, list]] = {c: {"effect": [], "variance": [], "runs": []}
                                                for c in cell.contrast_names()}
    runs_out, phis = [], []
    for i, ar in enumerate(cell.runs):
        t1 = time.time()
        Y = cell.load_data(i)
        fit = RunBlockGLM(cell.axis, [ar.design], noise=noise).fit([Y])
        if args.save_phi:
            phi = np.array(fit.fits[0].phi, dtype=np.float32)
            phi[~fit.fits[0].ok] = np.nan
            phis.append(phi)
        for name, c in ar.contrasts.items():
            m = fit.contrast(c)
            per_contrast[name]["effect"].append(m.effect.astype(np.float32))
            per_contrast[name]["variance"].append(m.variance.astype(np.float32))
            per_contrast[name]["runs"].append(ar.run.entity_prefix)
        n_cens = 0 if ar.design.retain is None else int((~ar.design.retain).sum())
        rec = {"run": ar.run.entity_prefix, "n_scans": int(Y.shape[0]), "n_censored": n_cens,
               "contrasts": sorted(ar.contrasts),
               "ok_fraction": float(fit.ok.mean()), "masked": int(fit.masked_per_run()[0]),
               "phi_median": np.nanmedian(fit.fits[0].phi, axis=0).round(4).tolist(),
               "seconds": round(time.time() - t1, 1)}
        runs_out.append(rec)
        print(f"  {ar.run.entity_prefix}: n {Y.shape[0]} ({n_cens} censored), ok {rec['ok_fraction']:.4f} ({rec['masked']} masked), "
              f"{rec['seconds']:.0f} s")
        del Y

    arrays = {}
    for name, acc in per_contrast.items():
        if not acc["effect"]:
            continue
        arrays[f"{name}_effect"] = np.stack(acc["effect"])
        arrays[f"{name}_variance"] = np.stack(acc["variance"])
        arrays[f"{name}_runs"] = np.array(acc["runs"])
    pool = f"ses-{sessions[0]}_" if sessions else ""
    stem = f"sub-{subject}_{pool}model-{model.name}_space-{args.space}_noise-{noise.name}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / f"{stem}_runs.npz", **arrays)
    if args.save_phi:
        np.savez(args.out_dir / f"{stem}_phi.npz", phi=np.stack(phis),
                 runs=np.array([ar.run.entity_prefix for ar in cell.runs]),
                 mask_flat=np.flatnonzero(cell.mask.ravel()), mask_shape=np.array(cell.mask.shape),
                 affine=cell.mask_img.affine)
    record = {"subject": subject, "model": model.name, "space": args.space, "sessions": sessions,
              "noise": noise.name, "censor_fd": args.censor_fd, "confounds_override": args.confounds,
              "stand_in_desc": args.desc, "stand_in_metadata_runs": wanted,
              "mask_voxels": n_vox, "config": cell.cfg.to_dict(),
              "runs_per_contrast": {k: len(v["runs"]) for k, v in per_contrast.items()},
              "runs": runs_out, "seconds": round(time.time() - t0, 1)}
    (args.out_dir / f"{stem}.json").write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out_dir / stem}.json (+ _runs.npz), {time.time() - t0:.0f} s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
