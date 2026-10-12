#!/usr/bin/env python3
"""Fake task designs fitted to resting runs: braintwill's null-design false-positive rate.

The glm-strategy program's Part R test R-1 (Eklund et al. 2016-style): on resting data the
true effect of any design is zero, so the fraction of voxels a calibrated fit calls
significant at α = 0.05 two-sided should be about 5 %. Two design shapes, matching the
program's simulation calibration (Part S) so the real-data number reads against it:

- **block**: 12 s blocks of three conditions (5 each) and 6 rest blocks, shuffled — Part S's
  shape (6 each + 8 rest on 312 s) scaled to fit a 300 s resting run;
- **event**: 2 s events of four conditions at SOA ~ U(3, 6) s.

Each draw shuffles the order and shifts the first onset by a random phase within one block
(block) or one mean SOA (event). Every design is built through the frozen reference
specification's path (``build_design_matrix`` under ``reference_config``): SPM canonical HRF,
the reference confounds, fMRIPrep's cosines and non-steady-state spikes, intercept. Two
contrasts per design: A vs baseline and A vs B.

Per run, per draw, per design, per contrast, it writes the boolean |z| > z_α map packed to
bits, plus per-run noise-fit diagnostics (stationary fraction, masked count, wall time). The
pass/fail reading is not here: it belongs to the record that pre-registered it.

``--save-maps DIR`` also writes each contrast's effect and variance (float32, variance NaN where
the noise fit was masked) to ``DIR``: the rest-run null for Part R's between-run consistency
test (R-2). The draws are seeded per run, so the designs and fits are the same as without it.

``--draws 1 --runs <one per subject>`` is the pre-registered feasibility check's resting half.

Usage (compute node):
    .venv/bin/python scripts/braintwill_glm/null_design.py --subject sub-## --task <rest task> \\
        --run-index 0 --draws 1 --out-dir /path/outside/this/repo
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter import coefficient_axis, config_bids_root, load_scaled, split_design  # noqa: E402

from braintwill import NoiseModel, RunBlockGLM  # noqa: E402

from neuroimaging.glm.config import repetition_time  # noqa: E402
from neuroimaging.glm.design import build_design_matrix  # noqa: E402
from neuroimaging.glm.models import StatsModel  # noqa: E402
from neuroimaging.glm.reference import reference_config  # noqa: E402
from neuroimaging.io import find_fmriprep_runs, load_confounds, mask_intersection  # noqa: E402

DESIGNS = {
    "block": ("A", "B", "C"),
    "event": ("A", "B", "C", "D"),
}
CONTRASTS = {"AvsBaseline": {"A": 1.0}, "AvsB": {"A": 1.0, "B": -1.0}}
BLOCK_S, LEAD_S, TAIL_S = 12.0, 6.0, 15.0
BLOCKS_PER_CONDITION, REST_BLOCKS = 5, 6


def fake_events(kind: str, n_scans: int, t_r: float, rng: np.random.Generator) -> pd.DataFrame:
    """One random draw of a design shape on a run of ``n_scans``."""
    span = n_scans * t_r
    if kind == "block":
        slots = rng.permutation(np.r_[np.repeat([0, 1, 2], BLOCKS_PER_CONDITION), np.full(REST_BLOCKS, -1)])
        start = LEAD_S + rng.uniform(0.0, BLOCK_S)
        if start + BLOCK_S * len(slots) > span - TAIL_S:
            raise ValueError(f"a {span:.0f} s run cannot hold {len(slots)} blocks of {BLOCK_S:.0f} s")
        rows = [(start + BLOCK_S * i, BLOCK_S, "ABC"[c]) for i, c in enumerate(slots) if c >= 0]
    elif kind == "event":
        onsets, t = [], LEAD_S + rng.uniform(0.0, 4.5)
        while t + 2.0 < span - TAIL_S:
            onsets.append(t)
            t += rng.uniform(3.0, 6.0)
        cond = rng.permutation(np.arange(len(onsets)) % 4)
        rows = [(o, 2.0, "ABCD"[c]) for o, c in zip(onsets, cond)]
    else:
        raise ValueError(f"design kind {kind!r}; known: {sorted(DESIGNS)}")
    return pd.DataFrame(rows, columns=["onset", "duration", "trial_type"])


def null_model(kind: str, task: str) -> StatsModel:
    return StatsModel(name=f"null-{kind}", description="null design on resting data", tasks=(task,),
                      factor="trial_type", conditions=DESIGNS[kind], hrf_model="spm", contrasts=(),
                      fixed_effects=True, path=Path(f"null-{kind}"))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True)
    p.add_argument("--task", required=True, nargs="+", help="resting task label(s); runs are pooled in this order")
    p.add_argument("--run-index", type=int, required=True, help="index into the subject's sorted rest runs")
    p.add_argument("--draws", type=int, default=20)
    p.add_argument("--seed", type=int, default=20261011)
    p.add_argument("--noise", default="ar3-bc")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--space", default="T1w")
    p.add_argument("--regime", default="reference")
    p.add_argument("--list", action="store_true", help="print the run list and mask size, fit nothing")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--save-maps", type=Path, default=None,
                   help="also write per-draw effect + variance maps here (large: ~180 MB per run)")
    return p.parse_args(argv)


def parse_noise(name: str) -> NoiseModel:
    if name == "white":
        return NoiseModel(order=0)
    head, _, kind = name.partition("-")
    return NoiseModel(order=int(head[2:]), corrected=kind == "bc")


def main(argv=None) -> int:
    from scipy import stats

    args = parse_args(argv)
    subject = args.subject.removeprefix("sub-")
    bids_root = config_bids_root()
    cfg = dataclasses.replace(reference_config(args.regime), space=args.space, hrf_model="spm")
    runs = []
    for task in args.task:
        runs += sorted(find_fmriprep_runs(subject=subject, task=task, variant=cfg.variant, space=args.space,
                                          bids_root=bids_root), key=lambda r: (r.session, r.run or ""))
    if not runs:
        raise SystemExit(f"ERROR: no {args.task} runs for sub-{subject} in {args.space}")
    # One mask for every rest run of the subject, so voxel FPRs aggregate across runs.
    mask_img, mask = mask_intersection(runs)
    if args.list:
        for i, r in enumerate(runs):
            print(i, r.entity_prefix)
        print(f"{len(runs)} runs, {int(mask.sum())} voxels in the intersection")
        return 0
    run = runs[args.run_index]
    noise = parse_noise(args.noise)
    z_crit = stats.norm.isf(args.alpha / 2)
    rng = np.random.default_rng([args.seed, int(subject), args.run_index])
    t0 = time.time()
    Y = load_scaled(run.bold, mask_img)
    n_scans, t_r = Y.shape[0], repetition_time(run, bids_root)
    confounds = load_confounds(run)
    t_load = time.time() - t0
    out_dir = args.out_dir / f"sub-{subject}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sub-{subject}_{run.entity_prefix.split('_', 1)[1]}_noise-{noise.name}"
    record = {"subject": subject, "run": run.entity_prefix, "run_index": args.run_index, "n_scans": n_scans,
              "t_r": t_r, "mask_voxels": int(mask.sum()), "noise": noise.name, "alpha": args.alpha,
              "seed": [args.seed, int(subject), args.run_index], "load_s": round(t_load, 1), "draws": []}
    hits, maps = {}, {}
    for d in range(args.draws):
        for kind in DESIGNS:
            model = null_model(kind, run.task)
            events = fake_events(kind, n_scans, t_r, rng)
            dm = build_design_matrix(events, confounds, t_r, n_scans, model, cfg, strict=True)
            axis = coefficient_axis(model, cfg)
            design = split_design(dm, model.conditions)
            t1 = time.time()
            fit = RunBlockGLM(axis, [design], noise=noise).fit([Y])
            entry = {"draw": d, "design": kind, "n_events": int(len(events)), "ok_fraction": float(fit.ok.mean()),
                     "masked": int(fit.masked_per_run()[0]), "fit_s": round(time.time() - t1, 1),
                     "phi_median": np.nanmedian(fit.fits[0].phi, axis=0).round(4).tolist(), "fpr": {}}
            for cname, weights in CONTRASTS.items():
                c = np.array([weights.get(cond, 0.0) for cond in model.conditions])
                cmap = fit.contrast(c)
                z = cmap.z
                if args.save_maps is not None:
                    maps[f"{kind}_{cname}_d{d:02d}_effect"] = cmap.effect.astype(np.float32)
                    maps[f"{kind}_{cname}_d{d:02d}_variance"] = cmap.variance.astype(np.float32)
                sig = np.abs(z) > z_crit                      # NaN (masked) compares False
                entry["fpr"][cname] = float(sig[fit.ok].mean())
                hits[f"{kind}_{cname}_d{d:02d}"] = np.packbits(sig)
                hits[f"{kind}_{cname}_d{d:02d}_ok"] = np.packbits(fit.ok)
            record["draws"].append(entry)
            print(f"  draw {d} {kind}: ok {entry['ok_fraction']:.4f} ({entry['masked']} masked), "
                  f"{entry['fit_s']:.0f} s, FPR " + ", ".join(f"{k} {v:.4f}" for k, v in entry["fpr"].items()))
    np.savez_compressed(out_dir / f"{stem}_hits.npz", **hits)
    (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=1))
    if args.save_maps is not None:
        maps_dir = args.save_maps / f"sub-{subject}"
        maps_dir.mkdir(parents=True, exist_ok=True)
        np.savez(maps_dir / f"{stem}_maps.npz", **maps)
    print(f"wrote {out_dir / stem}.json (+ _hits.npz), {time.time() - t0:.0f} s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
