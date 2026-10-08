#!/usr/bin/env python3
"""Response route: classic hyperalignment on the films the subjects share.

Pre-registration §6 (response row), handoff 2026-09-29 in mmmdata-agents
``docs/archive/workbench/functional-space/``. One job per (scenario, level, draw,
target). Rows are **measured** series on the job's shared alignment films
(``use == "align_shared"``); nothing is predicted. The template is built from
the two template subjects and frozen before the target enters; the target's
transform comes from its own shared-film series only (§2).

Defined only where the target shares films with the template: primary
25/50/100 %. At primary 0 % there are no shared films, and in the secondary
scenario the target shares none, so the route is undefined there (a job
exits without writing).

Pairing (DECIDED 2026-09-30): subjects' window grids sit at different film
times, so each shared film is paired by nearest film time
(``films.paired_slices``), to the first template subject's grid:

  template  the two template subjects, paired to each other
  target    all three subjects, paired together (so the target's rows line up
            with template-subject rows; the template's rotations are fixed
            column maps, so pairing rows for the entry does not touch them)

Each paired slice is z-scored per column, so every row block is centred and
on one scale. A column that is constant (or non-finite) on a subject's rows is
invalid for that subject; nothing is filled.

Solver and outputs are the stimulus route's: per-piece Grams, iterative
Procrustes averaging (``stimulus_route.gram_template``), the target's entry
through ``G[T, s] R_s``, and per-subject crosses in ``cha.save_cross`` form,
so ``procrustes.transform_from_cross`` gives any λ. ``--save-grams`` keeps the
Grams for the jointly built combined template.

Verbs:

  plan   report a job's shared films and sizes; fits nothing
  fit    one job: writes <derivatives>/functional_space/routes/response/
         <scenario>/pct-<pct>/draw-<draw>/target-<sub>/ (crosses + sidecar)

Usage:
    python response_route.py plan --pct 50 --draw 0 --target 03
    python response_route.py fit --pct 50 --draw 0 --target 03
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "src" / "python", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import films as fm  # noqa: E402
import stimulus_route as sr  # noqa: E402


def out_dir(derivatives: Path, scenario: str, pct: int, draw: int, target: str) -> Path:
    return (Path(derivatives) / "functional_space" / "routes" / "response" / scenario
            / f"pct-{pct:03d}" / f"draw-{draw:02d}" / f"target-{target}")


def shared_films(parts: pd.DataFrame, windows: pd.DataFrame, scenario: str, pct: int, draw: int, target: str
                 ) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """The job's films shared by all three subjects, and each subject's window rows for them (by film).

    Returns ``([], {})`` when the target shares no film (the route is undefined).
    """
    job = parts[(parts["scenario"] == scenario) & (parts["pct"] == pct) & (parts["draw"] == draw)
                & (parts["target"] == target)]
    if job.empty:
        raise KeyError(f"no partition job {scenario} {pct}% draw {draw} target {target}")
    shared = job[job["use"] == "align_shared"]
    per_sub = {str(s): set(g["stimulus_id"]) for s, g in shared.groupby("subject")}
    if target not in per_sub:
        return [], {}
    films = sorted(set.intersection(*per_sub.values())) if len(per_sub) == 3 else []
    if not films:
        return [], {}
    out = {}
    for sub in per_sub:
        rows = windows[(windows["sub"] == sub) & (windows["role"] == "alignment")
                       & windows["stimulus_id"].isin(films)].set_index("stimulus_id")
        if len(rows) != len(films):
            raise ValueError(f"sub-{sub}: {len(rows)} alignment windows for {len(films)} shared films")
        out[sub] = rows
    return films, out


def zscore_columns(y: np.ndarray) -> np.ndarray:
    """Per-column z-score; a constant column comes out NaN (invalid), never filled."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (y - y.mean(0)) / y.std(0)


def paired_block(series: dict[str, dict[str, np.ndarray]], rows: dict[str, pd.DataFrame], subs: list[str],
                 ref: str, films: list[str]) -> dict[str, np.ndarray]:
    """Each subject's paired, z-scored rows over all films, concatenated in film order."""
    out = {s: [] for s in subs}
    for sid in films:
        sl = fm.paired_slices({s: rows[s].loc[sid] for s in subs}, ref)
        for s in subs:
            out[s].append(zscore_columns(series[s][sid][sl[s]]))
    return {s: np.concatenate(v) for s, v in out.items()}


def target_row_map(rows: dict[str, pd.DataFrame], template_subs: list[str], target: str, ref: str,
                   films: list[str]) -> np.ndarray:
    """For each target-block row, its row in the template block (both lie on ``ref``'s grid).

    The three-way pairing covers a sub-range of the two-way one on the
    reference grid, so every target-block row has exactly one template row.
    """
    out, at = [], 0
    for sid in films:
        s2 = fm.paired_slices({s: rows[s].loc[sid] for s in template_subs}, ref)[ref]
        s3 = fm.paired_slices({s: rows[s].loc[sid] for s in [*template_subs, target]}, ref)[ref]
        if s3.start < s2.start or s3.stop > s2.stop:
            raise ValueError(f"{sid}: three-way pairing leaves the two-way range on sub-{ref}'s grid")
        out.append(at + np.arange(s3.start - s2.start, s3.stop - s2.start))
        at += s2.stop - s2.start
    return np.concatenate(out)


@dataclass
class SharedFilmJob:
    """One job's paired shared-film rows, pieces and columns (the response and SRM routes' common input)."""

    films: list[str]
    subs: list[str]
    template_subs: list[str]
    target: str
    ref: str
    tpl: dict[str, np.ndarray]  # template block: template subjects, paired to each other
    tgt: dict[str, np.ndarray]  # target block: all three, paired together
    tgt_rows_in_tpl: np.ndarray
    pcols: "sr.PieceColumns"
    n_grayordinates: int


def prepare(args: argparse.Namespace, log=print) -> SharedFilmJob | None:
    """Load and pair a job's shared films; ``None`` (logged) where the route is undefined."""
    import encoding as enc
    import grayordinates as go
    import partitions as pt
    import cha
    import pieces as pc

    paths = enc.Paths()
    cha_paths = cha.Paths()
    windows = fm.load_windows(paths.derivatives)
    parts = pt.load_partitions(paths.derivatives)
    films, rows = shared_films(parts, windows, args.scenario, args.pct, args.draw, args.target)
    if not films:
        log(f"undefined: target sub-{args.target} shares no film in {args.scenario} "
            f"{args.pct}% draw {args.draw}; nothing written")
        return None
    subs = sorted(rows)
    template_subs = [s for s in subs if s != args.target]
    ref = template_subs[0]
    table = pd.read_csv(go.grayordinates_path(cha_paths.cleaned), sep="\t")
    parcels, names = pc.load_cortex_parcels(cha_paths.atlases)
    labels = pc.piece_labels(table, parcels, names, pc.load_hipp_unfold_x(go.hipp_template_path(cha_paths.cleaned)))

    cache: dict = {}
    series = {s: {sid: fm.film_series(rows[s].loc[sid], cha_paths.cleaned, cache) for sid in films} for s in subs}
    cache.clear()
    tpl = paired_block(series, rows, template_subs, ref, films)
    tgt = paired_block(series, rows, subs, ref, films)
    del series
    valid = {s: np.isfinite(tpl[s]).all(0) for s in template_subs}
    valid[args.target] = np.isfinite(tgt[args.target]).all(0)
    return SharedFilmJob(films, subs, template_subs, args.target, ref, tpl, tgt,
                         target_row_map(rows, template_subs, args.target, ref, films),
                         sr.piece_columns(labels, valid, template_subs, args.target), int(len(table)))


def run_job(args: argparse.Namespace, log=print) -> Path | None:
    import encoding as enc
    import cha

    t0 = time.time()
    job = prepare(args, log)
    if job is None:
        return None
    paths = enc.Paths()
    films, subs, template_subs, ref = job.films, job.subs, job.template_subs, job.ref
    tpl, tgt, pcols = job.tpl, job.tgt, job.pcols

    tpl_pairs = [(a, b) for i, a in enumerate(template_subs) for b in template_subs[i:]]
    tgt_pairs = [(args.target, s) for s in template_subs]
    acc_tpl = sr.GramAccumulator(pcols, tpl_pairs, args.target)
    acc_tgt = sr.GramAccumulator(pcols, tgt_pairs, args.target)
    acc_tpl.add(tpl)
    acc_tgt.add(tgt)
    lab_list = list(pcols.template)
    rot, cross = sr.gram_template(acc_tpl.g, template_subs, lab_list)
    cross[args.target] = sr.target_cross(acc_tgt.g, args.target, template_subs, rot, pcols)
    tcols = {lab: pcols.template[lab][pos] for lab, pos in pcols.target.items()}

    dest = out_dir(paths.derivatives, args.scenario, args.pct, args.draw, args.target)
    dest.mkdir(parents=True, exist_ok=True)
    for s in subs:
        cha.save_cross(sr.as_cross(cross[s], tcols if s == args.target else pcols.template),
                       dest / f"cross_sub-{s}.npz")
    if args.save_grams:
        sr.save_grams(acc_tpl.g, dest / "grams_template_film.npz", pcols)
        sr.save_grams(acc_tgt.g, dest / "grams_target_film.npz", pcols)
    side = {
        "description": "Response route, one partition job: per-piece cross-products (X_s' template) for every "
                       "subject; procrustes.transform_from_cross(cha.load_cross(path), n, lam) gives the "
                       "transform into the template for any lam.",
        "job": {"scenario": args.scenario, "pct": args.pct, "draw": args.draw, "target": args.target},
        "template_subjects": template_subs, "reference_grid": f"sub-{ref}",
        "shared_films": films,
        "rows": {"template": acc_tpl.n_rows, "target": acc_tgt.n_rows},
        "pairing": "nearest film time (films.paired_slices) to the reference grid; template = template subjects "
                   "paired to each other, target = all three paired together; each paired slice z-scored "
                   "per column",
        "n_pieces": {"template": len(pcols.template), "target": len(pcols.target)},
        "n_columns": {"template": int(sum(c.size for c in pcols.template.values())),
                      "target": int(sum(p.size for p in pcols.target.values()))},
        "template_iterations": sr.TEMPLATE_ITERATIONS,
        "alignment_diagnostics": {f"sub-{s}": sr.alignment_diagnostics(cross[s]) for s in subs},
        "alignment_diagnostics_note": "alignment data only, not a score: 'captured' = aligned over anatomical "
                                      "cross-covariance with the template at lam 0; 'tr_over_p' = mean tr(R)/p "
                                      "per lam",
        "n_grayordinates": job.n_grayordinates, "grams_saved": bool(args.save_grams),
        "code_version": sr._code_version(),
        "seconds": {"total": round(time.time() - t0, 1)},
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (dest / "response.json").write_text(json.dumps(side, indent=2) + "\n")
    log(f"wrote {dest} in {side['seconds']['total']:.0f} s")
    return dest


def cmd_plan(args: argparse.Namespace) -> None:
    import encoding as enc
    import partitions as pt

    paths = enc.Paths()
    windows = fm.load_windows(paths.derivatives)
    films, rows = shared_films(pt.load_partitions(paths.derivatives), windows, args.scenario, args.pct,
                               args.draw, args.target)
    if not films:
        print(f"undefined: target sub-{args.target} shares no film in {args.scenario} {args.pct}% draw {args.draw}")
        return
    for s, r in sorted(rows.items()):
        role = "target" if s == args.target else "template"
        print(f"sub-{s} ({role}): {len(films)} shared films, {int(r['n'].sum())} window volumes")
    print(f"out: {out_dir(paths.derivatives, args.scenario, args.pct, args.draw, args.target)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    for verb in ("plan", "fit"):
        p = sub.add_parser(verb)
        p.add_argument("--scenario", default="primary", choices=("primary", "secondary"))
        p.add_argument("--pct", type=int, required=True)
        p.add_argument("--draw", type=int, required=True)
        p.add_argument("--target", required=True)
        if verb == "fit":
            p.add_argument("--save-grams", action="store_true",
                           help="also keep the per-block Grams (for a jointly built combined template)")
    args = ap.parse_args()
    {"plan": cmd_plan, "fit": run_job}[args.verb](args)


if __name__ == "__main__":
    main()
