#!/usr/bin/env python3
"""Combined model tuned by segment identification, with per-network (w, λ) shrunk toward a global pair.

A follow-up to ``combined.py``, with the same blocks, Grams, tuning films and
(w, λ) grid. Two things change:

Objective. ``combined.py`` tunes on per-column temporal r. Here the
objective is M2b (``scoring.m2b``), computed on the tuning films between the
two template subjects. b is carried into a's space by ``R_b R_a'`` and
compared with a's own series, and a into b's the same way. Each film is
z-scored per column on both sides, and the series are cut into 10-TR
segments. Each segment's pattern is ranked against every tuning segment of
the other subject, and the score is rank accuracy, averaged over both
directions. It is computed per family-A network, ``O_n(θ)``. The global
objective ``Ō(θ)`` is the mean of ``O_n`` over the networks.

Per-network weights. Each cortical piece takes the family-A network that
holds most of its columns. Pieces with no family-A majority (subcortex,
hippocampus) take the global pair. For a shrinkage κ:

    θ_n(κ) = argmax_θ [(1 - κ) O_n(θ) + κ Ō(θ)],   κ ∈ KAPPAS

κ = 1 is the global pair for every network. κ is chosen per job by 2-fold
cross-validation over the tuning films: pick θ_n on one half, score
``mean_n O_n(θ_n)`` on the other, and average the two folds. Ties go to the
larger κ. ``θ_n(κ*)`` is then taken on all tuning films. Ties within an
argmax go to the first cell in grid order (weights outer, λ inner, as
``combined.tuning_table``).

Decomposition. The objective is computed piece by piece. Within a network,
segment pattern correlations need only sums over the network's pieces: the
cross-products of the segment patterns, and each pattern's sum and sum of
squares. A worker returns those per (cell, direction). The segment-by-segment
correlation, and from it the rank accuracy on any subset of segments (a
cross-validation half), comes from the sums. A column is dropped from a
cell when it is non-finite on either side in any film, as ``scoring.m2b``
drops it. The ``check`` verb recomputes one network and cell directly through
``scoring.m2b`` on the carried series and requires agreement.

Models written (route trees beside ``combined-<space>``):

  combined-<space>-segid   κ = 1: one global (w, λ), chosen by the M2b objective
  combined-<space>-pernet  κ = κ*: per-network (w, λ), per-piece λ in the sidecar

Each tree holds ``cross_sub-*.npz`` (crosses at each piece's weights) and
``combined.json``. ``pieces`` there gives each piece's network, weights and λ,
so ``procrustes.transform_from_cross(cross, n, {piece: lam})`` is the model.
The pernet tree also holds ``tuning.parquet`` (O per network × cell × fold)
and ``kappa.tsv`` (the CV score per κ).

Verbs:

  plan   the job's blocks, Gram files, tuning films and piece → network assignment; fits nothing
  fit    one job: both trees
  check  one job: the decomposed objective against ``scoring.m2b`` on carried series (one network, a few cells)

Usage:
    python combined_pernet.py plan --space psytwill --pct 0 --draw 0 --target 03
    python combined_pernet.py fit --space psytwill --pct 0 --draw 0 --target 03 --n-jobs 16
    python combined_pernet.py check --space psytwill --pct 0 --draw 0 --target 03 --network SomMot
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "src" / "python", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import combined as cb  # noqa: E402
import procrustes as pr  # noqa: E402
import scoring as sc  # noqa: E402
import stimulus_route as sr  # noqa: E402

KAPPAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CV_SEED = 20261122
SPACES = ("ebind", "psytwill")
#: Family-A network names (score_route.network_labels gives one per grayordinate).
FAMILY_A_NETWORKS = ("Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis")


# ---------------------------------------------------------------------------
# segments
# ---------------------------------------------------------------------------

def segment_rows(film_lengths: list[int], length: int = sc.SEGMENT_TRS) -> tuple[np.ndarray, np.ndarray]:
    """``(rows, film)``: each segment's row indices into the concatenated block (S x length), and its film index."""
    rows, film, at = [], [], 0
    for f, n in enumerate(film_lengths):
        for sl in sc.segment_slices(n, length):
            rows.append(np.arange(at + sl.start, at + sl.stop))
            film.append(f)
        at += n
    return np.array(rows, dtype=np.int64).reshape(-1, length), np.array(film, dtype=np.int64)


def zscore_films(y: np.ndarray, bounds: list[tuple[int, int]]) -> np.ndarray:
    """Per-column z-score within each film (``scoring.zscore_columns`` per film)."""
    out = np.empty_like(y)
    for a, b in bounds:
        out[a:b] = sc.zscore_columns(y[a:b])
    return out


def _bounds(film_lengths: list[int]) -> list[tuple[int, int]]:
    ends = np.cumsum(film_lengths)
    return [(int(e - n), int(e)) for e, n in zip(ends, film_lengths)]


def pattern_sums(target: np.ndarray, template: np.ndarray, seg: np.ndarray) -> tuple:
    """One direction's sums over one piece's kept columns: (C, s_t, ss_t, s_p, ss_p, n_entries).

    ``target`` and ``template`` are z-scored per film already; a column non-finite
    in either is left out (``scoring.m2b``).
    """
    ok = np.isfinite(target).all(axis=0) & np.isfinite(template).all(axis=0)
    t = target[:, ok][seg].reshape(seg.shape[0], -1).astype(np.float64)
    p = template[:, ok][seg].reshape(seg.shape[0], -1).astype(np.float64)
    return t @ p.T, t.sum(1), (t * t).sum(1), p.sum(1), (p * p).sum(1), t.shape[1]


def piece_sums(tpl_grams: list[dict], coefs: list[float], weights: list[tuple], template_subs: list[str],
               y: dict[str, np.ndarray], bounds: list[tuple[int, int]], seg: np.ndarray,
               lam_grid=pr.LAMBDA_GRID) -> dict[str, np.ndarray] | None:
    """One piece's pattern sums over the (w, λ) grid, both directions.

    Shapes: ``C`` (n_w, n_lam, 2, S, S); ``st, sst, sp, ssp`` (n_w, n_lam, 2, S);
    ``n`` (n_w, n_lam, 2). Direction 0 is a's own series against b carried into a;
    direction 1 is b's own against a carried into b. ``None`` when a tuning
    column is non-finite (as ``combined.tune_piece``, the piece is left out).
    """
    from threadpoolctl import threadpool_limits

    a, b = template_subs
    if not all(np.isfinite(v).all() for v in y.values()):
        return None
    nw, nl, s = len(weights), len(lam_grid), seg.shape[0]
    out = {"C": np.zeros((nw, nl, 2, s, s)), **{k: np.zeros((nw, nl, 2, s)) for k in ("st", "sst", "sp", "ssp")},
           "n": np.zeros((nw, nl, 2))}
    keys = ("C", "st", "sst", "sp", "ssp", "n")
    anat = None
    with threadpool_limits(1):
        for i, w in enumerate(weights):
            g = cb.stack([{pair: {"p": m} for pair, m in blk.items()} for blk in tpl_grams],
                         [wb * cbb for wb, cbb in zip(w, coefs)])
            _, cross = sr.gram_template(g, template_subs, ["p"])
            msv = {u: cb._mean_sv(cross[u]["p"]) for u in template_subs}
            for j, lam in enumerate(lam_grid):
                if np.isinf(lam) and anat is not None:  # identity: the same for every weight vector
                    for k in keys:
                        out[k][i, j] = out[k][0, j]
                    continue
                r = {u: cb._rotation(cross[u]["p"], lam, msv[u]) for u in template_subs}
                carried = {a: y[b] @ (r[b] @ r[a].T), b: y[a] @ (r[a] @ r[b].T)}
                for d, (own, other) in enumerate(((a, b), (b, a))):
                    vals = pattern_sums(y[own], zscore_films(carried[own], bounds), seg)
                    for k, v in zip(keys, vals):
                        out[k][i, j, d] = v
                if np.isinf(lam):
                    anat = True
    return out


def chunk_sums(pieces: list, tpl_by_piece: dict, coefs: list[float], weights: list[tuple],
               template_subs: list[str], y_by_piece: dict, bounds, seg) -> tuple[dict | None, list]:
    """Sum ``piece_sums`` over a chunk of pieces (one network); returns (sums, pieces left out)."""
    tot, dropped = None, []
    for lab in pieces:
        r = piece_sums(tpl_by_piece[lab], coefs, weights, template_subs, y_by_piece[lab], bounds, seg)
        if r is None:
            dropped.append(lab)
            continue
        tot = r if tot is None else {k: tot[k] + r[k] for k in tot}
    return tot, dropped


def rank_accuracy(sums: dict, idx: np.ndarray | None = None) -> np.ndarray:
    """Mean rank accuracy per (w, λ) over both directions, segments restricted to ``idx`` (foils too).

    Correlation over pattern entries, as ``scoring.identification``; a tie counts half.
    """
    c, n = sums["C"], sums["n"][..., None, None]
    mt, mp = sums["st"] / sums["n"][..., None], sums["sp"] / sums["n"][..., None]
    vt = sums["sst"] / sums["n"][..., None] - mt ** 2
    vp = sums["ssp"] / sums["n"][..., None] - mp ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = (c / n - mt[..., :, None] * mp[..., None, :]) / np.sqrt(vt[..., :, None] * vp[..., None, :])
    if idx is not None:
        corr = corr[..., idx[:, None], idx[None, :]]
    m = corr.shape[-1]
    true = np.diagonal(corr, axis1=-2, axis2=-1)[..., None]
    below = (corr < true).sum(-1)
    ties = (corr == true).sum(-1) - 1  # the diagonal itself
    rank = (below + 0.5 * ties) / (m - 1)
    return rank.mean(axis=(-1, -2))  # over segments, then the two directions (equal counts)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def choose(obj: np.ndarray, kappa: float) -> np.ndarray:
    """Per network, the flat cell maximising ``(1 - κ) O_n + κ Ō``; ``obj`` is (n_net, n_cell). First max wins."""
    blend = (1 - kappa) * obj + kappa * obj.mean(axis=0, keepdims=True)
    return blend.argmax(axis=1)


def cv_kappa(halves: list[np.ndarray], kappas=KAPPAS) -> tuple[float, dict[float, float]]:
    """κ* by 2-fold CV: select on one half's objective, score ``mean_n O_n`` on the other. Ties -> larger κ."""
    nets = np.arange(halves[0].shape[0])
    score = {}
    for k in kappas:
        s = [halves[1 - h][nets, choose(halves[h], k)].mean() for h in (0, 1)]
        score[k] = float(np.mean(s))
    best = max(sorted(kappas, reverse=True), key=lambda k: score[k])  # max keeps the first (larger κ) on ties
    return best, score


def film_halves(n_films: int, seed: int) -> np.ndarray:
    """Fold (0/1) per film: a seeded permutation split in half (the first half takes the extra film)."""
    perm = np.random.default_rng(seed).permutation(n_films)
    fold = np.zeros(n_films, dtype=np.int64)
    fold[perm[(n_films + 1) // 2:]] = 1
    return fold


def job_seed(scenario: str, pct: int, draw: int, target: str) -> int:
    return CV_SEED + 100_000 * (scenario == "secondary") + 1000 * pct + 10 * draw + int(target)


# ---------------------------------------------------------------------------
# pieces -> networks
# ---------------------------------------------------------------------------

def piece_networks(common: sr.PieceColumns, networks: np.ndarray) -> tuple[dict, float]:
    """Each piece's family-A majority network ('' when it has none), and the share of family-A columns
    whose piece is assigned another network."""
    out, off, tot = {}, 0, 0
    for lab, cols in common.template.items():
        names, counts = np.unique(networks[cols], return_counts=True)
        best = names[np.argmax(counts)]
        out[lab] = best if best in FAMILY_A_NETWORKS else ""
        fa = np.isin(networks[cols], FAMILY_A_NETWORKS)
        tot += int(fa.sum())
        off += int((fa & (networks[cols] != out[lab])).sum())
    return out, off / max(tot, 1)


# ---------------------------------------------------------------------------
# job
# ---------------------------------------------------------------------------

def out_dir(derivatives: Path, space: str, model: str, scenario: str, pct: int, draw: int, target: str) -> Path:
    return (Path(derivatives) / "functional_space" / "routes" / f"combined-{space}-{model}" / scenario
            / f"pct-{pct:03d}" / f"draw-{draw:02d}" / f"target-{target}")


def tuning_films(parts, windows, scenario, pct, draw, target, template_subs, cleaned):
    """``combined.tuning_rows``, keeping each film's length: (films, {sub: block}, lengths)."""
    import films as fm
    import response_route as rr

    job = parts[(parts["scenario"] == scenario) & (parts["pct"] == pct) & (parts["draw"] == draw)
                & (parts["target"] == target) & (parts["use"] == "tuning")]
    per_sub = {s: sorted(job.loc[job["subject"] == s, "stimulus_id"]) for s in template_subs}
    films = per_sub[template_subs[0]]
    if not films or any(v != films for v in per_sub.values()):
        raise ValueError("tuning films missing or differ between the template subjects")
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
    per_film = [rr.paired_block(series, rows, template_subs, template_subs[0], [sid]) for sid in films]
    lengths = [int(b[template_subs[0]].shape[0]) for b in per_film]
    y = {s: np.concatenate([b[s] for b in per_film]).astype(np.float32) for s in template_subs}
    return films, y, lengths


def load_job(args, log=print) -> dict:
    """Everything a job needs: blocks on common columns, scales, tuning series, segments, piece networks."""
    import cha
    import encoding as enc
    import films as fm
    import grayordinates as go
    import partitions as pt
    import score_route as srt

    paths = enc.Paths()
    cleaned = cha.Paths().cleaned
    parts = pt.load_partitions(paths.derivatives)
    windows = fm.load_windows(paths.derivatives)
    subs = sorted(parts.loc[(parts["scenario"] == args.scenario) & (parts["pct"] == args.pct)
                            & (parts["draw"] == args.draw) & (parts["target"] == args.target), "subject"].unique())
    if args.target not in subs or len(subs) != 3:
        raise ValueError(f"job resolves to subjects {subs}; expected the target and two template subjects")
    template_subs = [s for s in subs if s != args.target]
    specs = cb.job_blocks(paths.derivatives, parts, windows, args.space, args.scenario, args.pct, args.draw,
                          args.target)
    if [n for n, _, _ in specs] != ["cha", "stimulus"]:
        raise ValueError(f"expected blocks cha + stimulus (a 0 % job); got {[n for n, _, _ in specs]}")
    raw = [cb.load_block(n, d, prts, template_subs, args.target) for n, d, prts in specs]
    common = cb.common_columns(raw)
    blocks = [cb.restrict(b, common) for b in raw]
    del raw
    coefs = [cb.energy_scale(b, template_subs) for b in blocks]
    frozen = json.loads((cb.out_dir(paths.derivatives, args.space, args.scenario, args.pct, args.draw,
                                    args.target) / "combined.json").read_text())
    for (n, _, _), c in zip(specs, coefs):
        if not np.isclose(c, frozen["blocks"][n]["scale_c"], rtol=1e-9):
            raise ValueError(f"{n}: energy scale {c} differs from the frozen combined job's")
    films, y, lengths = tuning_films(parts, windows, args.scenario, args.pct, args.draw, args.target,
                                     template_subs, cleaned)
    if films != frozen["tuning"]["films"] or y[template_subs[0]].shape[0] != frozen["tuning"]["paired_volumes"]:
        raise ValueError("tuning films or paired volumes differ from the frozen combined job's")
    table = pd.read_csv(go.grayordinates_path(cleaned), sep="\t")
    networks = srt.network_labels(table, cha.Paths().atlases)
    net_of, straddle = piece_networks(common, networks)
    seg, seg_film = segment_rows(lengths)
    log(f"blocks {[n for n, _, _ in specs]}; {len(common.template)} pieces; {len(films)} tuning films, "
        f"{len(seg)} segments; straddle share {straddle:.4f}")
    return {"paths": paths, "specs": specs, "blocks": blocks, "common": common, "coefs": coefs,
            "template_subs": template_subs, "subs": subs, "films": films, "y": y, "lengths": lengths,
            "seg": seg, "seg_film": seg_film, "net_of": net_of, "straddle": straddle, "networks": networks,
            "n": len(table), "frozen": frozen}


def network_sums(job: dict, n_jobs: int, weights: list[tuple], log=print) -> tuple[dict, list]:
    """Pattern sums per family-A network (pieces chunked within a network for the workers)."""
    from joblib import Parallel, delayed

    common, blocks, ts = job["common"], job["blocks"], job["template_subs"]
    bounds = _bounds(job["lengths"])
    by_net = {net: sorted((lab for lab, nn in job["net_of"].items() if nn == net),
                          key=lambda lab: -common.template[lab].size) for net in FAMILY_A_NETWORKS}
    n_chunks = max(1, 2 * n_jobs)
    total = sum(len(v) for v in by_net.values())
    tasks = []
    for net, labs in by_net.items():
        k = max(1, round(n_chunks * len(labs) / total))
        for c in range(k):
            if labs[c::k]:
                tasks.append((net, labs[c::k]))
    tpl = {lab: [{pair: b.tpl[pair][lab] for pair in b.tpl} for b in blocks] for _, labs in tasks for lab in labs}
    ys = {lab: {s: job["y"][s][:, common.template[lab]] for s in ts} for lab in tpl}
    res = Parallel(n_jobs=n_jobs)(
        delayed(chunk_sums)(labs, {lab: tpl[lab] for lab in labs}, job["coefs"], weights, ts,
                            {lab: ys[lab] for lab in labs}, bounds, job["seg"])
        for _, labs in tasks)
    sums, dropped = {}, []
    for (net, _), (s, d) in zip(tasks, res):
        dropped += d
        if s is not None:
            sums[net] = s if net not in sums else {k: sums[net][k] + s[k] for k in s}
    missing = [net for net in FAMILY_A_NETWORKS if net not in sums]
    if missing:
        raise ValueError(f"no tuning pieces for networks {missing}")
    return sums, dropped


def objective_tables(sums: dict, seg_film: np.ndarray, fold: np.ndarray) -> dict[str, np.ndarray]:
    """O (n_net, n_cell) on all tuning segments and on each CV half."""
    out = {}
    halves = {"all": None, "half0": np.flatnonzero(fold[seg_film] == 0), "half1": np.flatnonzero(fold[seg_film] == 1)}
    for key, idx in halves.items():
        out[key] = np.stack([rank_accuracy(sums[net], idx).reshape(-1) for net in FAMILY_A_NETWORKS])
    return out


def fit_model(job: dict, w_by_piece: dict, n_jobs: int) -> dict:
    """Crosses for every subject with each piece at its own weights (``combined.fit_piece``)."""
    from joblib import Parallel, delayed

    common, blocks, ts = job["common"], job["blocks"], job["template_subs"]
    target = [s for s in job["subs"] if s not in ts][0]
    labs = sorted(common.template, key=lambda lab: -common.template[lab].size)
    fitted = Parallel(n_jobs=n_jobs)(
        delayed(cb.fit_piece)([{pair: b.tpl[pair][lab] for pair in b.tpl} for b in blocks],
                              [{pair: b.tgt[pair][lab] for pair in b.tgt} for b in blocks]
                              if lab in common.target else [],
                              [wb * c for wb, c in zip(w_by_piece[lab], job["coefs"])], ts, target,
                              common.target.get(lab))
        for lab in labs)
    tcols = {lab: common.template[lab][pos] for lab, pos in common.target.items()}
    return {s: sr.as_cross({lab: f[s] for lab, f in zip(labs, fitted) if s in f},
                           tcols if s == target else common.template) for s in job["subs"]}


def run_fit(args, log=print) -> list[Path]:
    import cha

    t0 = time.time()
    seconds = {}
    job = load_job(args, log)
    seconds["load"] = round(time.time() - t0, 1)
    weights = cb.simplex(len(job["blocks"]))
    names = [n for n, _, _ in job["specs"]]
    cells = [(w, lam) for w in weights for lam in pr.LAMBDA_GRID]

    t1 = time.time()
    sums, dropped = network_sums(job, args.n_jobs, weights, log)
    seconds["tuning"] = round(time.time() - t1, 1)
    fold = film_halves(len(job["films"]), job_seed(args.scenario, args.pct, args.draw, args.target))
    obj = objective_tables(sums, job["seg_film"], fold)
    del sums
    kappa, cv = cv_kappa([obj["half0"], obj["half1"]])
    pick = {k: choose(obj["all"], k) for k in KAPPAS}
    glob = int(obj["all"].mean(axis=0).argmax())
    if not np.all(pick[1.0] == glob):
        raise AssertionError("κ = 1 does not reduce to the global pair")
    log(f"tuning {seconds['tuning']:.0f} s; {len(dropped)} pieces left out; κ* {kappa} (cv {cv}); "
        f"global {cells[glob]}; pernet {dict(zip(FAMILY_A_NETWORKS, [cells[c] for c in pick[kappa]]))}")

    rows = []
    for key, tab in obj.items():
        for ni, net in enumerate(FAMILY_A_NETWORKS):
            for ci, (w, lam) in enumerate(cells):
                rows.append({"fold": key, "network": net, **{f"w_{b}": wb for b, wb in zip(names, w)},
                             "lam": lam, "objective": float(tab[ni, ci])})
    tuning = pd.DataFrame(rows)

    net_cell = {"segid": {net: glob for net in FAMILY_A_NETWORKS},
                "pernet": dict(zip(FAMILY_A_NETWORKS, (int(c) for c in pick[kappa])))}
    dests = []
    for model, chosen in net_cell.items():
        t1 = time.time()
        piece_cell = {lab: chosen.get(net, glob) for lab, net in job["net_of"].items()}
        crosses = fit_model(job, {lab: cells[c][0] for lab, c in piece_cell.items()}, args.n_jobs)
        dest = out_dir(job["paths"].derivatives, args.space, model, args.scenario, args.pct, args.draw, args.target)
        dest.mkdir(parents=True, exist_ok=True)
        for s, cross in crosses.items():
            cha.save_cross(cross, dest / f"cross_sub-{s}.npz")
        if model == "pernet":
            tuning.to_parquet(dest / "tuning.parquet", index=False)
            pd.DataFrame({"kappa": list(cv), "cv_objective": list(cv.values())}).to_csv(
                dest / "kappa.tsv", sep="\t", index=False, float_format="%.6g")
        seconds[f"fit_{model}"] = round(time.time() - t1, 1)
        side = {
            "description": f"Combined model ({model}), one partition job: the frozen combined-{args.space} blocks "
                           "and Grams, (w, lam) chosen by M2b segment identification between the template "
                           "subjects on the tuning films"
                           + (", one global pair (kappa = 1)" if model == "segid" else
                              ", per family-A network, shrunk toward the global pair by kappa chosen in 2-fold "
                              "CV over tuning films")
                           + ". cross_sub-*.npz are crosses at each piece's weights; "
                             "procrustes.transform_from_cross(cha.load_cross(path), n, {piece: lam}) with "
                             "'pieces' gives the model.",
            "space": args.space, "model": model,
            "job": {"scenario": args.scenario, "pct": args.pct, "draw": args.draw, "target": args.target},
            "template_subjects": job["template_subs"],
            "blocks": {n: {"route_dir": str(d), "gram_parts": list(p), "scale_c": c}
                       for (n, d, p), c in zip(job["specs"], job["coefs"])},
            "networks": {net: {"weights": dict(zip(names, cells[c][0])), "lam": cells[c][1],
                               "objective": float(obj["all"][i, c])}
                         for i, (net, c) in enumerate(chosen.items())},
            "global": {"weights": dict(zip(names, cells[glob][0])), "lam": cells[glob][1],
                       "objective": float(obj["all"].mean(axis=0)[glob])},
            "kappa": kappa if model == "pernet" else 1.0,
            "cv": {str(k): v for k, v in cv.items()} if model == "pernet" else None,
            "pieces": {str(lab): {"network": job["net_of"][lab], "weights": dict(zip(names, cells[c][0])),
                                  "lam": cells[c][1]} for lab, c in piece_cell.items()},
            "straddle_share": job["straddle"],
            "tuning": {"films": job["films"], "film_lengths": job["lengths"], "segments": int(len(job["seg"])),
                       "fold": fold.tolist(), "seed": job_seed(args.scenario, args.pct, args.draw, args.target),
                       "pieces_left_out": [str(x) for x in dropped],
                       "objective": "M2b rank accuracy between the template subjects on the tuning films, per "
                                    "family-A network over its assigned pieces, both directions; fit diagnostic"},
            "weight_step": cb.WEIGHT_STEP, "lam_grid": [float(v) for v in pr.LAMBDA_GRID], "kappas": list(KAPPAS),
            "code_version": sr._code_version(), "seconds": seconds,
            "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        }
        (dest / "combined.json").write_text(json.dumps(side, indent=2) + "\n")
        dests.append(dest)
    log(f"wrote {[str(d) for d in dests]} in {time.time() - t0:.0f} s")
    return dests


def run_check(args, log=print) -> None:
    """The decomposed objective against ``scoring.m2b`` on the carried series, one network, a few cells."""
    job = load_job(args, log)
    weights = cb.simplex(len(job["blocks"]))
    ts, common = job["template_subs"], job["common"]
    a, b = ts
    labs = [lab for lab, net in job["net_of"].items() if net == args.network]
    bounds = _bounds(job["lengths"])
    tpl = {lab: [{pair: blk.tpl[pair][lab] for pair in blk.tpl} for blk in job["blocks"]] for lab in labs}
    ys = {lab: {s: job["y"][s][:, common.template[lab]] for s in ts} for lab in labs}
    tot, _ = chunk_sums(labs, tpl, job["coefs"], weights, ts, ys, bounds, job["seg"])
    mine = rank_accuracy(tot)
    worst = 0.0
    for i, j in [(0, 0), (2, 0), (len(weights) - 1, 3), (2, len(pr.LAMBDA_GRID) - 1)]:
        w, lam = weights[i], pr.LAMBDA_GRID[j]
        cols = np.concatenate([common.template[lab] for lab in labs])
        carried = {s: np.full((job["y"][s].shape[0], cols.size), np.nan, np.float32) for s in ts}
        at = 0
        for lab in labs:
            g = cb.stack([{pair: {"p": m} for pair, m in blk.items()} for blk in tpl[lab]],
                         [wb * c for wb, c in zip(w, job["coefs"])])
            _, cross = sr.gram_template(g, ts, ["p"])
            r = {u: pr.procrustes_from_cross(cross[u]["p"], lam) for u in ts}
            p = common.template[lab].size
            carried[a][:, at:at + p] = ys[lab][b] @ (r[b] @ r[a].T)
            carried[b][:, at:at + p] = ys[lab][a] @ (r[a] @ r[b].T)
            at += p
        accs = []
        for s in ts:
            ycat = np.concatenate([ys[lab][s] for lab in labs], axis=1)
            tf = {f"f{k}": ycat[x:y_] for k, (x, y_) in enumerate(bounds)}
            pf = {f"f{k}": carried[s][x:y_] for k, (x, y_) in enumerate(bounds)}
            res = sc.m2b(tf, pf, {args.network: np.arange(cols.size)})
            accs.append(res["rank_acc"].mean())
        direct = float(np.mean(accs))
        diff = abs(direct - float(mine[i, j]))
        worst = max(worst, diff)
        log(f"w {w} lam {lam}: decomposed {mine[i, j]:.6f} direct {direct:.6f} |diff| {diff:.2e}")
    if worst > 1e-6:
        raise AssertionError(f"decomposed objective differs from scoring.m2b by {worst:.2e}")
    log(f"objective check passed (max |diff| {worst:.2e})")

    # the per-piece fit path at the frozen job's weights must reproduce its saved crosses
    import encoding as enc
    import score_route as srt

    names = [n for n, _, _ in job["specs"]]
    full = job["frozen"]["selection"][job["frozen"]["full_model"]]
    w = tuple(full["weights"][n] for n in names)
    crosses = fit_model(job, {lab: w for lab in common.template}, args.n_jobs)
    cdir = cb.out_dir(enc.Paths().derivatives, args.space, args.scenario, args.pct, args.draw,
                      [s for s in job["subs"] if s not in ts][0])
    rel = srt._check_full(crosses, cdir, job["subs"])
    log(f"frozen-weights refit reproduces the saved crosses (max relative {rel:.2e})")


def cmd_plan(args) -> None:
    job = load_job(args)
    counts = pd.Series(job["net_of"]).replace("", "(global)").value_counts()
    print(counts.to_string())
    print(f"segments {len(job['seg'])}; films {job['lengths']}")
    for model in ("segid", "pernet"):
        print(out_dir(job["paths"].derivatives, args.space, model, args.scenario, args.pct, args.draw, args.target))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    for verb in ("plan", "fit", "check"):
        p = sub.add_parser(verb)
        p.add_argument("--space", default="psytwill", choices=SPACES)
        p.add_argument("--scenario", default="primary", choices=("primary", "secondary"))
        p.add_argument("--pct", type=int, default=0, choices=(0,))
        p.add_argument("--draw", type=int, required=True)
        p.add_argument("--target", required=True)
        p.add_argument("--n-jobs", type=int, default=1)
        if verb == "check":
            p.add_argument("--network", default="SomMot", choices=FAMILY_A_NETWORKS)
    args = ap.parse_args()
    {"plan": cmd_plan, "fit": run_fit, "check": run_check}[args.verb](args)


if __name__ == "__main__":
    main()
