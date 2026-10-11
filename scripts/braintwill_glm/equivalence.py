#!/usr/bin/env python3
"""Does braintwill's run-block GLM reproduce the stand-in maps on the same input?

The glm-strategy program's P-Stage 3 exit criterion ("equivalent supported behaviour") for the
condition-level OLS fit. For one stand-in cell (subject × model × space × session pool, read
from its ``*_run_metadata.json``), this rebuilds every run's design through ``adapter``, fits
each run with braintwill under **white noise** (the OLS variance: RSS over residual dof), pools
the runs by precision-weighted fixed effects with nilearn's own formula, and compares effect,
variance, t and z with the maps on disk in ``derivatives/<tree>/`` voxel for voxel.

``--noise ar3-bc`` (or any ``ar<p>-bc``/``ar<p>-raw``) also fits that noise model per run and
records, per run, the fraction of mask voxels whose fit is stationary and converged, the masked
count and the wall time: the pre-registered feasibility check's task-run half. Those numbers
compare nothing with the stand-in.

Prints a summary and writes everything to ``--out`` (JSON). The output is a finding; write it
outside this public repository.

Usage (a compute node — a T1w cell loads every run's BOLD):
    .venv/bin/python scripts/braintwill_glm/equivalence.py --subject sub-## --model floc \\
        --sessions ses-## --out /path/to/equivalence.json [--noise ar3-bc]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter import adapt, config_bids_root  # noqa: E402

from braintwill import NoiseModel, RunBlockGLM  # noqa: E402

from neuroimaging.io import find_fmriprep_runs  # noqa: E402

STATS = ("effect", "variance", "t", "z")


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
    p.add_argument("--noise", default=None, help="also fit this noise model per run (feasibility)")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def parse_noise(name: str) -> NoiseModel:
    if name == "white":
        return NoiseModel(order=0)
    head, _, kind = name.partition("-")
    if not head.startswith("ar") or kind not in ("bc", "raw"):
        raise SystemExit(f"--noise takes white, ar<p>-bc or ar<p>-raw; got {name!r}")
    return NoiseModel(order=int(head[2:]), corrected=kind == "bc")


def derivatives_dir() -> Path:
    from core.config import load_config

    paths = load_config()["paths"]
    return Path(paths.get("output_dir", Path(paths["bids_project_dir"]) / "derivatives"))


def stand_in(tree: Path, subject: str, model, space: str, desc: str, session: str | None):
    """The stand-in cell's metadata and its maps, from the tree's own index."""
    d = tree / f"sub-{subject}" / (f"ses-{session}" if session else "") / "func"
    meta_path = d / f"sub-{subject}_task-{model.task}_space-{space}_desc-{desc}_model-{model.name}_run_metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"ERROR: no stand-in fit at {meta_path}")
    meta = json.loads(meta_path.read_text())
    maps = pd.read_csv(tree / "maps.tsv", sep="\t", dtype=str, keep_default_na=False)
    sel = maps[(maps.subject == subject) & (maps.task == model.task) & (maps.space == space)
               & (maps.desc == desc) & (maps.session == (session or "")) & (maps.run == "")]
    return meta, {(r.contrast, r.stat): tree / r.path for r in sel.itertuples()}


def fixed_effects(effects: np.ndarray, variances: np.ndarray, dofs: list[float]):
    """nilearn ``compute_fixed_effects(precision_weighted=True)``, on arrays."""
    from nilearn.glm.contrasts import Contrast

    variances = np.maximum(variances, 1.0e-16)
    w = 1.0 / variances
    var = 1.0 / w.sum(0)
    eff = (effects * w).sum(0) * var
    con = Contrast(effect=eff, variance=var, dim=1, dof=float(np.sum(dofs)), stat_type="t")
    z = con.z_score()
    return {"effect": eff, "variance": var, "t": con.stat_, "z": z}


def compare(ours: np.ndarray, disk: np.ndarray) -> dict:
    d = ours - disk
    scale = float(np.max(np.abs(disk))) or 1.0
    finite = np.isfinite(d)
    return {
        "max_abs": float(np.max(np.abs(d[finite]))),
        "median_abs": float(np.median(np.abs(d[finite]))),
        "max_rel_to_range": float(np.max(np.abs(d[finite])) / scale),
        "r": float(np.corrcoef(ours[finite], disk[finite])[0, 1]),
        "n_nonfinite": int((~finite).sum()),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    from neuroimaging.glm.models import load_model
    from nilearn.masking import apply_mask

    subject = args.subject.removeprefix("sub-")
    sessions = [s.removeprefix("ses-") for s in args.sessions] if args.sessions else None
    if sessions and len(sessions) > 1:
        raise SystemExit("ERROR: a stand-in session pool is one session, or the whole subject")
    model = load_model(args.model)
    tree = derivatives_dir() / args.tree
    meta, maps = stand_in(tree, subject, model, args.space, args.desc, sessions[0] if sessions else None)
    wanted = [(r["session"], r["run"]) for r in meta["runs"]]
    found = {(r.session, r.run): r for r in find_fmriprep_runs(subject=subject, task=model.task,
                                                               variant=meta["config"]["variant"],
                                                               space=args.space, bids_root=config_bids_root())}
    missing = [w for w in wanted if w not in found]
    if missing:
        raise SystemExit(f"ERROR: stand-in runs not found now: {missing}")
    t0 = time.time()
    cell = adapt(subject, args.model, space=args.space, regime=args.regime, runs=[found[w] for w in wanted])
    for key in ("confounds", "acompcor_n", "include_cosine", "include_non_steady_state", "hrf_model",
                "smoothing_fwhm", "drift_model"):
        if json.dumps(meta["config"].get(key)) != json.dumps(cell.cfg.to_dict().get(key)):
            raise SystemExit(f"ERROR: stand-in config {key}={meta['config'].get(key)!r} but the adapter "
                             f"built {cell.cfg.to_dict().get(key)!r}; not the same input")
    print(f"sub-{subject} {model.name} {args.space}: {len(cell.runs)} runs, {int(cell.mask.sum())} mask voxels, "
          f"axis {len(cell.axis)} conditions ({time.time() - t0:.0f} s to design)")

    noise = parse_noise(args.noise) if args.noise else None
    per_contrast: dict[str, dict[str, list]] = {c: {"effect": [], "variance": [], "dof": []}
                                                for c in cell.contrast_names()}
    runs_out = []
    for i, ar in enumerate(cell.runs):
        t1 = time.time()
        Y = cell.load_data(i)
        t_load = time.time() - t1
        glm = RunBlockGLM(cell.axis, [ar.design], noise=NoiseModel(order=0))
        t2 = time.time()
        fit = glm.fit([Y])
        dof = glm.kernels[0].dof
        for name, c in ar.contrasts.items():
            m = fit.contrast(c)
            per_contrast[name]["effect"].append(m.effect)
            per_contrast[name]["variance"].append(m.variance)
            per_contrast[name]["dof"].append(dof)
        rec = {"run": ar.run.entity_prefix, "n_scans": int(Y.shape[0]), "dof": dof,
               "contrasts": sorted(ar.contrasts), "load_s": round(t_load, 1),
               "white_fit_s": round(time.time() - t2, 1)}
        if noise is not None:
            t3 = time.time()
            nfit = RunBlockGLM(cell.axis, [ar.design], noise=noise).fit([Y])
            phi = nfit.fits[0].phi
            rec["noise"] = {
                "model": noise.name,
                "ok_fraction": float(nfit.ok.mean()),
                "masked": int(nfit.masked_per_run()[0]),
                "fit_s": round(time.time() - t3, 1),
                "phi_median": np.nanmedian(phi, axis=0).round(4).tolist(),
                "phi_q05": np.nanquantile(phi, 0.05, axis=0).round(4).tolist(),
                "phi_q95": np.nanquantile(phi, 0.95, axis=0).round(4).tolist(),
            }
        runs_out.append(rec)
        print(f"  {ar.run.entity_prefix}: n {Y.shape[0]}, dof {dof:.1f}, load {t_load:.0f} s"
              + (f", {noise.name} ok {rec['noise']['ok_fraction']:.4f} ({rec['noise']['masked']} masked, "
                 f"{rec['noise']['fit_s']:.0f} s)" if noise is not None else ""))
        del Y

    results = {}
    for name, acc in per_contrast.items():
        if not acc["effect"]:
            continue
        fx = fixed_effects(np.stack(acc["effect"]), np.stack(acc["variance"]), acc["dof"])
        results[name] = {"n_runs": len(acc["effect"])}
        for stat in STATS:
            path = maps.get((name, stat))
            if path is None or not path.exists():
                results[name][stat] = {"missing": str(path)}
                continue
            disk = np.asarray(apply_mask(str(path), cell.mask_img), dtype=float)
            results[name][stat] = compare(np.asarray(fx[stat], dtype=float), disk)
    worst = {stat: max((r[stat]["max_rel_to_range"] for r in results.values() if "max_abs" in r.get(stat, {})),
                       default=None) for stat in STATS}
    print("worst max |Δ| / max |stand-in|, over contrasts: "
          + ", ".join(f"{s} {v:.2e}" for s, v in worst.items() if v is not None))
    out = {
        "subject": subject, "model": model.name, "space": args.space, "sessions": sessions,
        "stand_in_desc": args.desc, "stand_in_metadata_runs": wanted, "mask_voxels": int(cell.mask.sum()),
        "config": cell.cfg.to_dict(), "runs": runs_out, "contrasts": results, "worst_rel": worst,
        "seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
