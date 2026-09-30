#!/usr/bin/env python3
"""SRM comparator and its PCA control, on the films the subjects share.

Pre-registration §6 (SRM row, controls) and §8 (k grid); the scope and
implementation were DECIDED 2026-09-30 in mmmdata-agents
``docs/workbench/functional-space/``:

- **Piecewise**, on the same pieces and columns as every other route. Per
  piece ``k_eff = min(k, smallest subject's column count in the piece)``, so
  k = 100 is close to full rank in the smaller pieces; the sidecar counts
  them.
- **BrainIAK** ``SRM`` (probabilistic; Chen 2015). The template is fitted on
  the two template subjects and frozen. The target enters through BrainIAK's
  ``transform_subject``, against the shared response restricted to the rows
  the target block covers.
- **PCA control at matched k (Nastase 2020):** per piece, the top-k principal
  axes of the template subjects' anatomically matched shared-film rows
  (stacked over subjects). Every subject, the target included, uses that one
  basis, i.e. SRM with every ``W_i`` forced equal. It isolates what
  subject-specific ``W_i`` add over plain dimension reduction.

Rows are the response route's (``response_route.prepare``): measured
shared-film series, paired by nearest film time, z-scored per paired slice.
The route is undefined where the target shares no film (primary 0 %, the
secondary scenario).

Outputs, per k: ``w_sub-<s>_k-<k>.npz`` and ``pca_sub-<s>_k-<k>.npz`` in
``cha.save_cross`` form (``{label: (cols, W)}`` with ``W`` columns x k_eff,
orthonormal columns). Template subject ``s`` is carried into the target's
space by ``W_T W_s'`` per piece.

MPI. BrainIAK imports mpi4py; libmpi comes from conda-forge mpich in the
env, and Slurm's PMIX_* variables are cleared before the import (MPICH aborts
on the PMIx runtime inside an srun step).

Verbs:

  fit    one job: writes <derivatives>/functional_space/routes/srm/<scenario>/
         pct-<pct>/draw-<draw>/target-<sub>/

Usage:
    python srm_route.py fit --pct 50 --draw 0 --target 03
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "src" / "python", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

K_GRID = (10, 20, 50, 100)  # §8
SRM_ITERATIONS = 10  # BrainIAK's default


def _brainiak_srm():
    for k in [k for k in os.environ if k.startswith(("PMIX_", "PMI_"))]:
        del os.environ[k]  # MPICH aborts on Slurm's PMIx runtime inside an srun step
    from brainiak.funcalign.srm import SRM

    return SRM


def out_dir(derivatives: Path, scenario: str, pct: int, draw: int, target: str) -> Path:
    return (Path(derivatives) / "functional_space" / "routes" / "srm" / scenario
            / f"pct-{pct:03d}" / f"draw-{draw:02d}" / f"target-{target}")


def fit_piece(tpl: list[np.ndarray], tgt: np.ndarray, tgt_rows: np.ndarray, k: int, seed: int = 0
              ) -> tuple[list[np.ndarray], np.ndarray, int]:
    """SRM on one piece: template subjects' ``W`` (in ``tpl`` order), the target's ``W``, and ``k_eff``.

    ``tpl`` are the template subjects' (rows, columns) data; ``tgt`` the
    target's (target rows, columns); ``tgt_rows`` maps each target row to its
    template row.
    """
    k_eff = int(min(k, tgt.shape[1], *(x.shape[1] for x in tpl)))
    model = _brainiak_srm()(n_iter=SRM_ITERATIONS, features=k_eff, rand_seed=seed)
    model.fit([x.T for x in tpl])
    shared = model.s_
    model.s_ = shared[:, tgt_rows]  # the target block covers a sub-range of the template rows
    try:
        w_t = model.transform_subject(tgt.T)
    finally:
        model.s_ = shared
    return list(model.w_), w_t, k_eff


def pca_piece(tpl: list[np.ndarray], k: int) -> tuple[np.ndarray, int]:
    """The top-k principal axes (columns x k_eff) of the template subjects' rows, stacked."""
    x = np.vstack(tpl).astype(np.float64)
    x = x - x.mean(axis=0)
    k_eff = int(min(k, x.shape[1], x.shape[0]))
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return vt[:k_eff].T, k_eff


def fit_piece_all_k(tpl: list[np.ndarray], tgt: np.ndarray, tgt_rows: np.ndarray, target_pos: np.ndarray,
                    ks: list[int]) -> dict[int, tuple]:
    """One piece over the k grid: ``{k: (template W's, target W, PCA basis, target's PCA rows, k_eff)}``.

    Runs in a worker with BLAS pinned to one thread: at these sizes (a few
    hundred columns) threaded BLAS is slower than serial (2026-09-30: an SVD
    took 37 ms on 1 thread, 119 ms on 8), so the job parallelises over pieces.
    """
    from threadpoolctl import threadpool_limits

    out = {}
    with threadpool_limits(1):
        for k in ks:
            ws, w_t, ke = fit_piece(tpl, tgt, tgt_rows, k)
            v, _ = pca_piece(tpl, ke)
            out[k] = (ws, w_t, v, v[target_pos], ke)
    return out


def run_job(args: argparse.Namespace, log=print) -> Path | None:
    import cha
    import encoding as enc
    import response_route as rr
    import stimulus_route as sr

    t0 = time.time()
    job = rr.prepare(args, log)
    if job is None:
        return None
    from joblib import Parallel, delayed

    ks = [int(k) for k in args.k.split(",")]
    w = {k: {s: {} for s in job.subs} for k in ks}
    pca = {k: {s: {} for s in job.subs} for k in ks}
    k_eff = {k: [] for k in ks}
    labs = [lab for lab in job.pcols.template if lab in job.pcols.target]
    t1 = time.time()
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(fit_piece_all_k)([job.tpl[s][:, job.pcols.template[lab]] for s in job.template_subs],
                                 job.tgt[job.target][:, job.pcols.template[lab][job.pcols.target[lab]]],
                                 job.tgt_rows_in_tpl, job.pcols.target[lab], ks)
        for lab in labs)
    for lab, per_k in zip(labs, results):
        cols = job.pcols.template[lab]
        tcols = cols[job.pcols.target[lab]]
        for k, (ws, w_t, v, v_t, ke) in per_k.items():
            for s, wi in zip(job.template_subs, ws):
                w[k][s][lab] = (cols, wi)
                pca[k][s][lab] = (cols, v)
            w[k][job.target][lab] = (tcols, w_t)
            pca[k][job.target][lab] = (tcols, v_t)
            k_eff[k].append(ke)
    seconds_fit = round(time.time() - t1, 1)
    log(f"{len(labs)} pieces x k {ks}: {seconds_fit:.0f} s on {args.n_jobs} workers")

    dest = out_dir(enc.Paths().derivatives, args.scenario, args.pct, args.draw, args.target)
    dest.mkdir(parents=True, exist_ok=True)
    for k in ks:
        for s in job.subs:
            cha.save_cross(w[k][s], dest / f"w_sub-{s}_k-{k:03d}.npz")
            cha.save_cross(pca[k][s], dest / f"pca_sub-{s}_k-{k:03d}.npz")
    side = {
        "description": "SRM comparator (BrainIAK probabilistic SRM, piecewise) and its shared-PCA control, one "
                       "partition job. Per k and subject, {piece: (cols, W)} in cha.save_cross form; template "
                       "subject s enters the target's space by W_T W_s' per piece.",
        "job": {"scenario": args.scenario, "pct": args.pct, "draw": args.draw, "target": args.target},
        "template_subjects": job.template_subs, "reference_grid": f"sub-{job.ref}", "shared_films": job.films,
        "rows": {"template": int(next(iter(job.tpl.values())).shape[0]),
                 "target": int(job.tgt[job.target].shape[0])},
        "k_grid": ks,
        "k_eff": {str(k): {"n_pieces": len(v), "capped": int(sum(e < k for e in v)),
                           "median": float(np.median(v)) if v else None} for k, v in k_eff.items()},
        "srm": {"implementation": "brainiak.funcalign.srm.SRM", "n_iter": SRM_ITERATIONS, "rand_seed": 0,
                "target_entry": "transform_subject against the shared response on the target block's rows"},
        "pca_control": "per piece, top-k_eff principal axes of the template subjects' rows stacked; one basis "
                       "for every subject (target: its columns' rows of the basis)",
        "n_grayordinates": job.n_grayordinates,
        "code_version": sr._code_version(),
        "seconds": {"fit": seconds_fit, "total": round(time.time() - t0, 1)}, "n_jobs": args.n_jobs,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (dest / "srm.json").write_text(json.dumps(side, indent=2) + "\n")
    log(f"wrote {dest} in {side['seconds']['total']:.0f} s")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--scenario", default="primary", choices=("primary", "secondary"))
    p.add_argument("--pct", type=int, required=True)
    p.add_argument("--draw", type=int, required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--k", default=",".join(str(k) for k in K_GRID), help="comma-separated k grid")
    p.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                   help="worker processes over pieces (default: the job's CPUs)")
    args = ap.parse_args()
    {"fit": run_job}[args.verb](args)


if __name__ == "__main__":
    main()
