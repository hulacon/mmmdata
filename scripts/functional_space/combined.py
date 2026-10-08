#!/usr/bin/env python3
"""Combined model: stacked route blocks, one jointly built template, (λ, w) tuned per job.

Pre-registration §6 (combined row), §8 (block weights on a 0.25 simplex, λ
grid), §3.4 (tuning); design DECIDED 2026-09-30 in mmmdata-agents
``docs/workbench/functional-space/``. One job per (space, scenario, level,
draw, target). Nothing is refitted: the job reads the per-piece Grams each
route saved (``--save-grams``) and stacks them.

Stacking. With block b's rows weighted by ``w_b`` and scaled by ``c_b``, the
stacked profile's Grams are ``X_a' X_b = sum_b w_b c_b G_b[a, b]``, so the
jointly built template is ``stimulus_route.gram_template`` on the summed
template Grams, and the target enters through ``target_cross`` on the summed
target Grams. Blocks:

  cha        CHA's own final-level (ico5) profiles; draw-independent, one Gram set per target
  stimulus   the stimulus route in one frozen feature space (EBind; psytwill in its later arm)
  response   the response route, only where the target shares films (primary 25/50/100 %)

Columns. A piece's columns are those every block keeps (template columns,
and the target's subset of them); each block's Grams are cut down to them.

Scale (energy norm). ``c_b = 1 / mean_s(sum_pieces tr G_b[s, s] / sum_pieces p)``
over the two template subjects: each block has equal self-energy, and its
cross-subject signal keeps its natural size. ``c_b`` is frozen with the
template and the target uses the same value.

Tuning (§3.4). One λ and one weight vector per job, over λ ∈ §8's grid × the
0.25-step simplex (faces and vertices included, so the same table gives the
tuned single routes and the tuned leave-one-route-out models). The objective
is between-template-subject prediction on the job's tuning films: subject b's
series, mapped into a's space by ``R_b(λ) R_a(λ)'`` per piece, correlated per
column with a's measured series, averaged over columns, both directions.
Films are paired by nearest film time and z-scored per column per film. A
piece with a non-finite tuning column in either subject is left out of the
objective (counted, never filled). It is a fit diagnostic, not a score
(DECIDED 2026-09-29).

Verbs:

  plan   report the job's blocks, Gram files and tuning films; fits nothing
  fit    one job: writes <derivatives>/functional_space/routes/combined-<space>/
         <scenario>/pct-<pct>/draw-<draw>/target-<sub>/ (tuning.tsv, the
         combined model's crosses, combined.json)

Usage:
    python combined.py plan --pct 0 --draw 0 --target 03
    python combined.py fit --pct 0 --draw 0 --target 03 --n-jobs 16
"""

from __future__ import annotations

import argparse
import datetime as _dt
import itertools
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

import procrustes as pr  # noqa: E402
import stimulus_route as sr  # noqa: E402
from scoring import column_r  # noqa: E402

WEIGHT_STEP = 0.25  # §8


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------

@dataclass
class Block:
    """One route's Grams over its own columns."""

    name: str
    tpl: dict  # (a, b) -> {label: G}, template pairs
    tgt: dict  # (target, s) -> {label: G}, target rows x template columns
    pcols: sr.PieceColumns


def _positions(have: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Positions of ``want`` in the sorted ``have``; every one must be present."""
    pos = np.searchsorted(have, want)
    if pos.size and (pos.max() >= have.size or not np.array_equal(have[pos], want)):
        raise ValueError("requested columns are not a subset of the block's columns")
    return pos


def columns_from_crosses(route_dir: Path, template_subs: list[str], target: str) -> sr.PieceColumns:
    """A route's columns, read from its cross files (for Grams saved before they carried columns)."""
    def cols(sub):
        z = np.load(route_dir / f"cross_sub-{sub}.npz", allow_pickle=True)
        return {str(lab): z[f"p{i}_cols"].astype(np.int64) for i, lab in enumerate(z["labels"])}

    tpl = cols(template_subs[0])
    if cols(template_subs[1]).keys() != tpl.keys():
        raise ValueError(f"{route_dir}: template subjects' crosses cover different pieces")
    tgt = {lab: _positions(tpl[lab], c) for lab, c in cols(target).items()}
    return sr.PieceColumns(tpl, tgt)


def load_block(name: str, route_dir: Path, parts: tuple[str, ...], template_subs: list[str], target: str) -> Block:
    """Sum a route's Gram parts (plain concatenation of their rows) into one block."""
    tpl = tgt = None
    pcols = None
    for kind in ("template", "target"):
        gs = []
        for part in parts:
            path = route_dir / f"grams_{kind}_{part}.npz"
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing; rerun the {name} route with --save-grams")
            g, pc_ = sr.load_grams(path)
            pcols = pcols or pc_
            gs.append(g)
        merged = sr.merge_grams(*gs)
        if kind == "template":
            tpl = merged
        else:
            tgt = merged
    if pcols is None:
        pcols = columns_from_crosses(route_dir, template_subs, target)
    want_tpl = {(a, b) for i, a in enumerate(template_subs) for b in template_subs[i:]}
    if set(tpl) != want_tpl or set(tgt) != {(target, s) for s in template_subs}:
        raise ValueError(f"{name}: Gram pairs {sorted(tpl)} / {sorted(tgt)} do not match the job's subjects")
    return Block(name, tpl, tgt, pcols)


def common_columns(blocks: list[Block]) -> sr.PieceColumns:
    """Per piece, the template columns every block keeps, and the target's subset every block keeps."""
    labs = set.intersection(*(set(b.pcols.template) for b in blocks))
    tpl, tgt = {}, {}
    for lab in sorted(labs):
        cols = blocks[0].pcols.template[lab]
        for b in blocks[1:]:
            cols = np.intersect1d(cols, b.pcols.template[lab])
        if cols.size == 0:
            continue
        tpl[lab] = cols
        tcols = cols
        for b in blocks:
            if lab not in b.pcols.target:
                tcols = np.array([], np.int64)
                break
            tcols = np.intersect1d(tcols, b.pcols.template[lab][b.pcols.target[lab]])
        if tcols.size:
            tgt[lab] = _positions(cols, tcols)
    return sr.PieceColumns(tpl, tgt)


def restrict(block: Block, common: sr.PieceColumns) -> Block:
    """The block's Grams cut down to the common columns."""
    tpl = {pair: {} for pair in block.tpl}
    tgt = {pair: {} for pair in block.tgt}
    for lab, cols in common.template.items():
        pos = _positions(block.pcols.template[lab], cols)
        for pair, g in block.tpl.items():
            tpl[pair][lab] = g[lab][np.ix_(pos, pos)]
        if lab in common.target:
            have = block.pcols.template[lab][block.pcols.target[lab]]
            tpos = _positions(have, cols[common.target[lab]])
            for pair, g in block.tgt.items():
                tgt[pair][lab] = g[lab][np.ix_(tpos, pos)]
    return Block(block.name, tpl, tgt, common)


def energy_scale(block: Block, template_subs: list[str]) -> float:
    """``c_b``: one over the template subjects' mean self-energy per column."""
    per_sub = []
    for s in template_subs:
        g = block.tpl[(s, s)]
        per_sub.append(sum(float(np.trace(m)) for m in g.values()) / sum(m.shape[0] for m in g.values()))
    e = float(np.mean(per_sub))
    if not e > 0:
        raise ValueError(f"{block.name}: non-positive self-energy {e}")
    return 1.0 / e


def simplex(n: int, step: float = WEIGHT_STEP) -> list[tuple[float, ...]]:
    """Every weight vector on the ``step`` grid of the (n-1)-simplex, faces and vertices included."""
    k = round(1 / step)
    if not np.isclose(k * step, 1.0):
        raise ValueError(f"step {step} does not divide 1")
    return [tuple(i / k for i in c) for c in itertools.product(range(k + 1), repeat=n) if sum(c) == k]


def stack(grams: list[dict], coefs: list[float]) -> dict:
    """``sum_b coef_b G_b`` for one pair set, skipping zero coefficients."""
    keep = [(g, c) for g, c in zip(grams, coefs) if c != 0]
    if not keep:
        raise ValueError("every block weight is zero")
    return sr.merge_grams(*[g for g, _ in keep], weights=[c for _, c in keep])


# ---------------------------------------------------------------------------
# tuning
# ---------------------------------------------------------------------------

def _mean_sv(m: np.ndarray) -> float:
    return float(np.linalg.svd(m, compute_uv=False).mean())


def _rotation(m: np.ndarray, lam: float, mean_sv: float) -> np.ndarray:
    """``procrustes.procrustes_from_cross(m, lam)`` with the piece's mean singular value precomputed."""
    if np.isinf(lam):
        return np.eye(m.shape[0])
    return pr.procrustes_from_cross(m + lam * mean_sv * np.eye(m.shape[0]) if lam > 0 else m, 0.0)


def tune_piece(tpl_grams: list[dict], coefs: list[float], weights: list[tuple], template_subs: list[str],
               y: dict[str, np.ndarray], lam_grid=pr.LAMBDA_GRID) -> np.ndarray | None:
    """One piece over the (w, λ) grid: ``(n_w, n_lam, 3)`` of (sum r a<-b, sum r b<-a, n columns).

    ``tpl_grams`` are the blocks' template Grams for this piece (``{pair: G}``),
    ``y`` the template subjects' tuning rows on its columns. ``None`` when a
    tuning column is non-finite (the piece is left out, never filled).
    """
    from threadpoolctl import threadpool_limits

    a, b = template_subs
    if not all(np.isfinite(v).all() for v in y.values()):
        return None
    out = np.zeros((len(weights), len(lam_grid), 3))
    with threadpool_limits(1):
        for i, w in enumerate(weights):
            g = stack([{pair: {"p": m} for pair, m in blk.items()} for blk in tpl_grams],
                      [wb * cb for wb, cb in zip(w, coefs)])
            _, cross = sr.gram_template(g, template_subs, ["p"])
            msv = {s: _mean_sv(cross[s]["p"]) for s in template_subs}
            for j, lam in enumerate(lam_grid):
                r = {s: _rotation(cross[s]["p"], lam, msv[s]) for s in template_subs}
                r_ab = column_r(y[b] @ (r[b] @ r[a].T), y[a])  # b carried into a's space
                r_ba = column_r(y[a] @ (r[a] @ r[b].T), y[b])
                ok = np.isfinite(r_ab) & np.isfinite(r_ba)
                out[i, j] = (r_ab[ok].sum(), r_ba[ok].sum(), ok.sum())
    return out


def tuning_rows(parts: pd.DataFrame, windows: pd.DataFrame, scenario: str, pct: int, draw: int, target: str,
                template_subs: list[str], cleaned: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """The job's tuning films, paired by nearest film time to the first template subject and z-scored per film."""
    import films as fm
    import response_route as rr

    job = parts[(parts["scenario"] == scenario) & (parts["pct"] == pct) & (parts["draw"] == draw)
                & (parts["target"] == target) & (parts["use"] == "tuning")]
    per_sub = {s: sorted(job.loc[job["subject"] == s, "stimulus_id"]) for s in template_subs}
    films = per_sub[template_subs[0]]
    if not films or any(v != films for v in per_sub.values()):
        raise ValueError(f"tuning films missing or differ between the template subjects: "
                         f"{ {s: len(v) for s, v in per_sub.items()} }")
    rows = {}
    for s in template_subs:
        r = windows[(windows["sub"] == s) & (windows["role"] == "alignment")
                    & windows["stimulus_id"].isin(films)].set_index("stimulus_id")
        if len(r) != len(films) or r.index.duplicated().any():
            raise ValueError(f"sub-{s}: {len(r)} alignment windows for {len(films)} tuning films")
        rows[s] = r
    cache: dict = {}
    series = {s: {sid: fm.film_series(rows[s].loc[sid], cleaned, cache) for sid in films} for s in template_subs}
    cache.clear()
    y = rr.paired_block(series, rows, template_subs, template_subs[0], films)
    return films, {s: v.astype(np.float32) for s, v in y.items()}


def tuning_table(results: dict, weights: list[tuple], block_names: list[str], lam_grid=pr.LAMBDA_GRID
                 ) -> pd.DataFrame:
    """Sum the pieces' (w, λ) results into the objective table."""
    tot = sum(results.values())
    rows = []
    for i, w in enumerate(weights):
        for j, lam in enumerate(lam_grid):
            s_ab, s_ba, n = tot[i, j]
            rows.append({**{f"w_{b}": wb for b, wb in zip(block_names, w)}, "lam": lam,
                         "r_a_from_b": s_ab / n, "r_b_from_a": s_ba / n, "objective": (s_ab + s_ba) / (2 * n),
                         "n_columns": int(n)})
    return pd.DataFrame(rows)


def select(table: pd.DataFrame, block_names: list[str]) -> dict[str, dict]:
    """The tuned (w, λ) on every closed face of the simplex: the full model, each leave-one-out, each single route.

    Keyed by the blocks in the face (``"cha+stimulus"``). A closed face keeps
    its sub-faces (the full model may land on a vertex), so each model is at
    least as good on the objective as every model nested in it. Ties go to the
    first row in grid order. At λ = ∞ the weights are irrelevant (anatomical).
    """
    out = {}
    for k in range(len(block_names), 0, -1):
        for face in itertools.combinations(block_names, k):
            off = [b for b in block_names if b not in face]
            sub = table
            for b in off:
                sub = sub[sub[f"w_{b}"] == 0]
            if sub.empty:
                continue
            best = sub.loc[sub["objective"].idxmax()]
            out["+".join(face)] = {"weights": {b: float(best[f"w_{b}"]) for b in block_names},
                                   "lam": float(best["lam"]), "objective": float(best["objective"])}
    return out


# ---------------------------------------------------------------------------
# the chosen model
# ---------------------------------------------------------------------------

def fit_piece(tpl_grams: list[dict], tgt_grams: list[dict], coefs: list[float], template_subs: list[str],
              target: str, target_pos: np.ndarray | None) -> dict[str, np.ndarray]:
    """One piece of the stacked model at fixed weights: each subject's cross against the joint template."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        g = stack([{pair: {"p": m} for pair, m in blk.items()} for blk in tpl_grams], coefs)
        rot, cross = sr.gram_template(g, template_subs, ["p"])
        out = {s: cross[s]["p"] for s in template_subs}
        if target_pos is not None:
            gt = stack([{pair: {"p": m} for pair, m in blk.items()} for blk in tgt_grams], coefs)
            out[target] = sr.target_cross(gt, target, template_subs, rot, sr.PieceColumns({}, {"p": target_pos}))["p"]
    return out


# ---------------------------------------------------------------------------
# job
# ---------------------------------------------------------------------------

def out_dir(derivatives: Path, space: str, scenario: str, pct: int, draw: int, target: str) -> Path:
    return (Path(derivatives) / "functional_space" / "routes" / f"combined-{space}" / scenario
            / f"pct-{pct:03d}" / f"draw-{draw:02d}" / f"target-{target}")


def job_blocks(derivatives: Path, parts: pd.DataFrame, windows: pd.DataFrame, space: str, scenario: str,
               pct: int, draw: int, target: str) -> list[tuple[str, Path, tuple[str, ...]]]:
    """(name, route directory, Gram parts) for each block this job stacks."""
    import cha
    import response_route as rr

    blocks = [("cha", cha.out_dir(derivatives, target), ("profile",)),
              ("stimulus", sr.out_dir(derivatives, space, scenario, pct, draw, target), ("film", "probe"))]
    films, _ = rr.shared_films(parts, windows, scenario, pct, draw, target)
    if films:
        blocks.append(("response", rr.out_dir(derivatives, scenario, pct, draw, target), ("film",)))
    return blocks


def run_job(args: argparse.Namespace, log=print) -> Path:
    import cha
    import encoding as enc
    import films as fm
    import partitions as pt
    from joblib import Parallel, delayed

    t0 = time.time()
    paths = enc.Paths()
    cleaned = cha.Paths().cleaned
    parts = pt.load_partitions(paths.derivatives)
    windows = fm.load_windows(paths.derivatives)
    subs = sorted(parts.loc[(parts["scenario"] == args.scenario) & (parts["pct"] == args.pct)
                            & (parts["draw"] == args.draw) & (parts["target"] == args.target), "subject"].unique())
    if args.target not in subs or len(subs) != 3:
        raise ValueError(f"job resolves to subjects {subs}; expected the target and two template subjects")
    template_subs = [s for s in subs if s != args.target]
    seconds = {}

    specs = job_blocks(paths.derivatives, parts, windows, args.space, args.scenario, args.pct, args.draw,
                       args.target)
    names = [n for n, _, _ in specs]
    t1 = time.time()
    raw = [load_block(n, d, prts, template_subs, args.target) for n, d, prts in specs]
    common = common_columns(raw)
    blocks = [restrict(b, common) for b in raw]
    del raw
    coefs = [energy_scale(b, template_subs) for b in blocks]
    seconds["load"] = round(time.time() - t1, 1)
    log(f"blocks {names}; {len(common.template)} pieces, "
        f"{sum(c.size for c in common.template.values())} template columns; c_b {[f'{c:.3g}' for c in coefs]}")

    t1 = time.time()
    films, y = tuning_rows(parts, windows, args.scenario, args.pct, args.draw, args.target, template_subs, cleaned)
    seconds["tuning_rows"] = round(time.time() - t1, 1)
    log(f"tuning: {len(films)} films, {y[template_subs[0]].shape[0]} paired volumes")

    weights = simplex(len(blocks))
    labs = sorted(common.template, key=lambda lab: -common.template[lab].size)  # the largest pieces set the tail
    t1 = time.time()
    res = Parallel(n_jobs=args.n_jobs)(
        delayed(tune_piece)([{pair: b.tpl[pair][lab] for pair in b.tpl} for b in blocks], coefs, weights,
                            template_subs, {s: y[s][:, common.template[lab]] for s in template_subs})
        for lab in labs)
    results = {lab: r for lab, r in zip(labs, res) if r is not None}
    dropped = [lab for lab, r in zip(labs, res) if r is None]
    seconds["tuning"] = round(time.time() - t1, 1)
    table = tuning_table(results, weights, names)
    chosen = select(table, names)
    full = chosen["+".join(names)]
    log(f"tuning grid {len(weights)} w x {len(pr.LAMBDA_GRID)} lam in {seconds['tuning']:.0f} s; "
        f"{len(dropped)} pieces left out; chosen {full}")

    t1 = time.time()
    w = [full["weights"][n] * c for n, c in zip(names, coefs)]
    fitted = Parallel(n_jobs=args.n_jobs)(
        delayed(fit_piece)([{pair: b.tpl[pair][lab] for pair in b.tpl} for b in blocks],
                           [{pair: b.tgt[pair][lab] for pair in b.tgt} for b in blocks] if lab in common.target
                           else [], w, template_subs, args.target, common.target.get(lab))
        for lab in labs)
    seconds["fit"] = round(time.time() - t1, 1)

    dest = out_dir(paths.derivatives, args.space, args.scenario, args.pct, args.draw, args.target)
    dest.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest / "tuning.tsv", sep="\t", index=False, float_format="%.6g")
    tcols = {lab: common.template[lab][pos] for lab, pos in common.target.items()}
    for s in subs:
        cross = {lab: f[s] for lab, f in zip(labs, fitted) if s in f}
        cols = tcols if s == args.target else common.template
        cha.save_cross(sr.as_cross(cross, cols), dest / f"cross_sub-{s}.npz")
    seconds["total"] = round(time.time() - t0, 1)
    side = {
        "description": "Combined model, one partition job: stacked route blocks (energy-scaled, weighted), one "
                       "jointly built template, (w, lam) tuned on the template subjects' tuning films. "
                       "cross_sub-*.npz are the crosses at the chosen weights; "
                       "procrustes.transform_from_cross(cha.load_cross(path), n, selection[full].lam) gives the "
                       "chosen transform. tuning.tsv is the full grid; 'selection' gives the tuned faces "
                       "(leave-one-route-out, single routes), recomputable from the routes' Grams.",
        "space": args.space, "job": {"scenario": args.scenario, "pct": args.pct, "draw": args.draw,
                                      "target": args.target},
        "template_subjects": template_subs,
        "blocks": {n: {"route_dir": str(d), "gram_parts": list(prts), "scale_c": c}
                   for (n, d, prts), c in zip(specs, coefs)},
        "full_model": "+".join(names),
        "selection": chosen,
        "weight_step": WEIGHT_STEP, "lam_grid": [float(v) for v in pr.LAMBDA_GRID],
        "tuning": {"films": films, "paired_volumes": int(y[template_subs[0]].shape[0]),
                   "reference_grid": f"sub-{template_subs[0]}",
                   "objective": "mean over columns of per-column r between a template subject's tuning series and "
                                "the other's mapped through R_b(lam) R_a(lam)', averaged over both directions; "
                                "fit diagnostic, not a score",
                   "pieces_left_out": dropped},
        "n_pieces": {"template": len(common.template), "target": len(common.target)},
        "n_columns": {"template": int(sum(c.size for c in common.template.values())),
                      "target": int(sum(p.size for p in common.target.values()))},
        "template_iterations": sr.TEMPLATE_ITERATIONS,
        "code_version": sr._code_version(),
        "seconds": seconds,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (dest / "combined.json").write_text(json.dumps(side, indent=2) + "\n")
    log(f"wrote {dest} in {seconds['total']:.0f} s")
    return dest


def cmd_plan(args: argparse.Namespace) -> None:
    import encoding as enc
    import films as fm
    import partitions as pt

    paths = enc.Paths()
    parts = pt.load_partitions(paths.derivatives)
    windows = fm.load_windows(paths.derivatives)
    for name, d, prts in job_blocks(paths.derivatives, parts, windows, args.space, args.scenario, args.pct,
                                    args.draw, args.target):
        files = [d / f"grams_{k}_{p}.npz" for k in ("template", "target") for p in prts]
        missing = [f.name for f in files if not f.exists()]
        print(f"{name}: {d} " + (f"MISSING {missing}" if missing else "grams present"))
    job = parts[(parts["scenario"] == args.scenario) & (parts["pct"] == args.pct) & (parts["draw"] == args.draw)
                & (parts["target"] == args.target) & (parts["use"] == "tuning")]
    print(f"tuning films: {job['stimulus_id'].nunique()}")
    print(f"out: {out_dir(paths.derivatives, args.space, args.scenario, args.pct, args.draw, args.target)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    for verb in ("plan", "fit"):
        p = sub.add_parser(verb)
        p.add_argument("--space", default="ebind", choices=("ebind", "vgg19", "psytwill"))
        p.add_argument("--scenario", default="primary", choices=("primary", "secondary"))
        p.add_argument("--pct", type=int, required=True)
        p.add_argument("--draw", type=int, required=True)
        p.add_argument("--target", required=True)
        if verb == "fit":
            p.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args()
    {"plan": cmd_plan, "fit": run_job}[args.verb](args)


if __name__ == "__main__":
    main()
