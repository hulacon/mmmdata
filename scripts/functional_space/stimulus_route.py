#!/usr/bin/env python3
"""Stimulus route: measured responses paired with encoder predictions (Wasserman 2026).

Pre-registration §6 (stimulus rows), handoff 2026-09-29 in mmmdata-agents
``docs/workbench/functional-space/``. One job per (space, scenario, level,
draw, target). Every subject's encoder (``encoding.py``) is fit on that
subject's own alignment films; the template is built from the two template
subjects and frozen before the target enters; the target's transform comes
from its own films and its own encoder only (§2).

Rows. A subject's side of the pairing is its **measured** response on its own
films (z-scored per film window, as the encoder was trained) and its
**predicted** response everywhere else. The template side is always
predicted, even on a film a template subject also aligns on: measured-vs-
measured pairs on shared films are the response route's, not this route's.
So every row block lies on the grid of the one subject who measured it, and
no two subjects' volume grids are ever matched. Row blocks:

  template   each template subject's films (that subject measured, the other
             predicted on the measurer's grid) + the probe set (both predicted)
  target     the target's films (target measured, template subjects
             predicted on the target's grid) + the probe set (all predicted)

Predictions are centred per film window / probe clip and per column, like the
measured side, and not rescaled: a prediction is already on the scale of the
z-scored response it predicts, and rescaling would give columns the features
do not explain full weight.

Solver. Iterative Procrustes averaging only ever uses ``X_s' Z_s'`` per piece,
so the whole route runs on per-piece Gram matrices accumulated over row
chunks (the ~51k probe volumes never sit in memory at once):

  G[a, b][piece] = X_a[:, cols]' X_b[:, cols]

``gram_template`` reproduces ``procrustes.template_average`` exactly from
them, and the outputs are the same per-subject cross-products (``X_s'
template``) that the CHA route stores, so ``procrustes.transform_from_cross``
gives any λ, and the tuning step reads both routes alike. The Grams are also
what a jointly built combined template needs (handoff step 3); ``--save-grams``
keeps them.

Columns. A piece's template columns are those valid (finite in the encoder's
training data) in both template subjects; the target's are the template
columns also valid in the target. This is stricter than ``template_average``'s
``nanmean`` for a column valid in only one template subject.

Verbs:

  plan   report a job's inputs and sizes; fits nothing
  fit    one job: writes <derivatives>/functional_space/routes/stimulus-<space>/
         <scenario>/pct-<pct>/draw-<draw>/target-<sub>/ (crosses + sidecar)

Usage:
    python stimulus_route.py plan --space ebind --pct 0 --draw 0 --target 03
    python stimulus_route.py fit --space ebind --pct 0 --draw 0 --target 03 --backend torch_cuda
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

import procrustes as pr  # noqa: E402

TEMPLATE_ITERATIONS = 3  # as cha.py
PROBE_CHUNK = 4000  # probe volumes per chunk (whole clips; a chunk may run over by one clip)
BLOCKS = ("film", "probe")


# ---------------------------------------------------------------------------
# Gram accumulation
# ---------------------------------------------------------------------------

@dataclass
class PieceColumns:
    """Per piece, the template columns and the target's subset of them (as positions into the template's)."""

    template: dict[object, np.ndarray]  # label -> grayordinate columns
    target: dict[object, np.ndarray]  # label -> positions into template[label]


def piece_columns(labels: np.ndarray, valid: dict[str, np.ndarray], template_subs: list[str], target: str
                  ) -> PieceColumns:
    """Template columns valid in both template subjects; the target's = those also valid in the target."""
    both = np.logical_and.reduce([valid[s] for s in template_subs])
    tpl, tgt = {}, {}
    for lab in pr._piece_labels(labels):
        cols = np.flatnonzero((labels == lab) & both)
        if cols.size == 0:
            continue
        tpl[lab] = cols
        pos = np.flatnonzero(valid[target][cols])
        if pos.size:
            tgt[lab] = pos
    return PieceColumns(tpl, tgt)


class GramAccumulator:
    """Sums per-piece ``X_a' X_b`` over row chunks for a fixed set of subject pairs.

    ``pairs`` are ``(a, b)``; for a pair whose first subject is the target, the
    first factor takes the target's columns and the second the template's.
    """

    def __init__(self, pcols: PieceColumns, pairs: list[tuple[str, str]], target: str):
        self.pcols = pcols
        self.pairs = list(pairs)
        self.target = target
        self.g: dict[tuple[str, str], dict] = {pair: {} for pair in self.pairs}
        self.n_rows = 0

    def add(self, chunk: dict[str, np.ndarray]) -> None:
        """Add one row chunk: ``{subject: (rows, n_grayordinates)}``, the same rows for every subject."""
        n = {x.shape[0] for x in chunk.values()}
        if len(n) != 1:
            raise ValueError(f"subjects' chunks differ in rows: {n}")
        for a, b in self.pairs:
            xa, xb = chunk[a], chunk[b]
            out = self.g[(a, b)]
            for lab, cols in self.pcols.template.items():
                if a == self.target:
                    if lab not in self.pcols.target:
                        continue
                    ca = cols[self.pcols.target[lab]]
                else:
                    ca = cols
                m = (xa[:, ca].astype(np.float32, copy=False).T @ xb[:, cols].astype(np.float32, copy=False)
                     ).astype(np.float64)  # one chunk in float32; the sum over chunks in float64
                if not np.isfinite(m).all():
                    raise ValueError(f"non-finite Gram entries in piece {lab!r} for pair {a}-{b}")
                out[lab] = out[lab] + m if lab in out else m
        self.n_rows += n.pop()


def merge_grams(*grams: dict, weights=None) -> dict:
    """``sum_k w_k G_k`` over accumulators' ``g`` dicts (same pairs and pieces)."""
    weights = [1.0] * len(grams) if weights is None else weights
    out = {}
    for pair in grams[0]:
        out[pair] = {lab: sum(w * g[pair][lab] for w, g in zip(weights, grams)) for lab in grams[0][pair]}
    return out


# ---------------------------------------------------------------------------
# template and entry from Grams
# ---------------------------------------------------------------------------

def _g(grams: dict, a: str, b: str, lab) -> np.ndarray:
    return grams[(a, b)][lab] if (a, b) in grams else grams[(b, a)][lab].T


def gram_template(grams: dict, subs: list[str], labels: list, n_iter: int = TEMPLATE_ITERATIONS
                  ) -> tuple[dict[str, dict], dict[str, dict]]:
    """Iterative Procrustes averaging (as ``procrustes.template_average``) from per-piece Grams.

    Returns ``(rotations, cross)``: each subject's λ = 0 rotation per piece
    from the last iteration, and its cross-product against the final template
    (``X_s' template``), which ``procrustes.procrustes_from_cross`` turns into
    the transform for any λ.
    """
    rot = {s: {lab: np.eye(_g(grams, s, s, lab).shape[0]) for lab in labels} for s in subs}

    def cross_against_template(s, lab):
        return sum(_g(grams, s, t, lab) @ rot[t][lab] for t in subs) / len(subs)

    for _ in range(n_iter):
        rot = {s: {lab: pr.procrustes_from_cross(cross_against_template(s, lab), 0.0) for lab in labels}
               for s in subs}
    cross = {s: {lab: cross_against_template(s, lab) for lab in labels} for s in subs}
    return rot, cross


def target_cross(grams: dict, target: str, template_subs: list[str], rot: dict[str, dict], pcols: PieceColumns
                 ) -> dict:
    """The target's cross-product ``X_T' template`` per piece, over the target's columns."""
    out = {}
    for lab, pos in pcols.target.items():
        out[lab] = sum(grams[(target, s)][lab] @ rot[s][lab][:, pos] for s in template_subs) / len(template_subs)
    return out


def as_cross(cross: dict, cols: dict) -> dict[object, tuple[np.ndarray, np.ndarray]]:
    """In ``procrustes.transform_from_cross`` / ``cha.save_cross`` form: ``{label: (cols, M)}``."""
    return {lab: (cols[lab], m) for lab, m in cross.items()}


def alignment_diagnostics(cross: dict, lam_grid=pr.LAMBDA_GRID) -> dict:
    """Transform geometry and cross-covariance captured, on alignment data (not a score).

    ``captured``: sum_p tr(R_p' M_p) / sum_p tr(M_p), the aligned cross-covariance
    against the anatomical one at λ = 0. ``tr_over_p``: mean tr(R)/p over pieces
    at each λ (the transition the λ grid has to span).
    """
    num = den = 0.0
    trp = {str(lam): [] for lam in lam_grid}
    for m in cross.values():
        for lam in lam_grid:
            r = pr.procrustes_from_cross(m, lam)
            if lam == 0.0:
                num += float(np.trace(r.T @ m))
                den += float(np.trace(m))
            trp[str(lam)].append(np.trace(r) / r.shape[0])
    return {"captured": round(num / den, 4) if den else None,
            "tr_over_p": {k: round(float(np.mean(v)), 4) for k, v in trp.items()}}


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def center_blocks(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each group's (film window's / clip's) column mean."""
    out = np.empty_like(x)
    for g in dict.fromkeys(np.asarray(groups).tolist()):
        sel = groups == g
        out[sel] = x[sel] - x[sel].mean(axis=0, keepdims=True)
    return out


@dataclass
class SubjectFilms:
    """One subject's alignment films: design on its own grid, measured series, film groups."""

    x: np.ndarray
    y: np.ndarray  # z-scored per film window
    groups: np.ndarray


def film_rows(measurer: str, films: dict[str, SubjectFilms], encoders: dict, predictors: list[str]
              ) -> dict[str, np.ndarray]:
    """One film block on ``measurer``'s grid: its measured series, and every predictor's prediction."""
    f = films[measurer]
    rows = {measurer: f.y}
    for s in predictors:
        if s != measurer:
            rows[s] = center_blocks(encoders[s].predict(f.x), f.groups)
    return rows


def probe_batches(design, ids: list[str], chunk: int = PROBE_CHUNK):
    """Yield ``(x, groups)`` over consecutive whole clips of about ``chunk`` volumes.

    ``design(stimulus_id)`` returns one clip's FIR design; clips are never
    split, so per-clip centring sees the whole clip.
    """
    cur: list[tuple[np.ndarray, str]] = []
    n = 0
    for sid in ids:
        x = design(sid)
        cur.append((x, sid))
        n += x.shape[0]
        if n >= chunk:
            yield np.concatenate([c[0] for c in cur]), np.concatenate([[c[1]] * c[0].shape[0] for c in cur])
            cur, n = [], 0
    if cur:
        yield np.concatenate([c[0] for c in cur]), np.concatenate([[c[1]] * c[0].shape[0] for c in cur])


# ---------------------------------------------------------------------------
# job
# ---------------------------------------------------------------------------

def out_dir(derivatives: Path, space: str, scenario: str, pct: int, draw: int, target: str) -> Path:
    return (Path(derivatives) / "functional_space" / "routes" / f"stimulus-{space}" / scenario
            / f"pct-{pct:03d}" / f"draw-{draw:02d}" / f"target-{target}")


def job_films(parts: pd.DataFrame, windows: pd.DataFrame, scenario: str, pct: int, draw: int, target: str
              ) -> dict[str, pd.DataFrame]:
    """Each subject's alignment-film window rows for one partition job."""
    import partitions as pt

    job = parts[(parts["scenario"] == scenario) & (parts["pct"] == pct) & (parts["draw"] == draw)
                & (parts["target"] == target) & (parts["use"] != "tuning")]
    if job.empty:
        raise KeyError(f"no partition job {scenario} {pct}% draw {draw} target {target}")
    out = {}
    for sub, g in job.groupby("subject"):
        rows = windows[(windows["sub"] == sub) & (windows["role"] == "alignment")
                       & windows["stimulus_id"].isin(g["stimulus_id"])]
        if len(rows) != pt.N_FILMS:
            raise ValueError(f"sub-{sub}: {len(rows)} alignment windows for {pt.N_FILMS} partition films")
        out[str(sub)] = rows
    if target not in out or len(out) != 3:
        raise ValueError(f"job resolves to subjects {sorted(out)}; expected the target and two template subjects")
    return out


def run_job(args: argparse.Namespace, log=print) -> Path:
    import encoding as enc
    import films as fm
    import grayordinates as go
    import partitions as pt
    import cha
    import pieces as pc

    t0 = time.time()
    paths = enc.Paths()
    cha_paths = cha.Paths()
    windows = fm.load_windows(paths.derivatives)
    parts = pt.load_partitions(paths.derivatives)
    rows = job_films(parts, windows, args.scenario, args.pct, args.draw, args.target)
    subs = sorted(rows)
    template_subs = [s for s in subs if s != args.target]
    table = pd.read_csv(go.grayordinates_path(cha_paths.cleaned), sep="\t")
    parcels, names = pc.load_cortex_parcels(cha_paths.atlases)
    labels = pc.piece_labels(table, parcels, names, pc.load_hipp_unfold_x(go.hipp_template_path(cha_paths.cleaned)))
    seconds = {}

    # measured films and encoders
    films, encoders = {}, {}
    for s in subs:
        x, y, groups, bands = enc.film_design(rows[s], paths.cache, args.space, cha_paths.cleaned)
        films[s] = SubjectFilms(x, enc._zscore_film_windows(y, groups), groups)
        t1 = time.time()
        encoders[s] = enc.Encoder(bands, backend=args.backend).fit(x, films[s].y, groups)
        seconds[f"fit_sub-{s}"] = round(time.time() - t1, 1)
        log(f"sub-{s}: {len(rows[s])} films, {x.shape[0]} volumes, encoder {seconds[f'fit_sub-{s}']:.0f} s "
            f"{encoders[s].diagnostics_}")
    valid = {s: encoders[s].valid_ for s in subs}
    pcols = piece_columns(labels, valid, template_subs, args.target)
    tpl_pairs = [(a, b) for i, a in enumerate(template_subs) for b in template_subs[i:]]
    tgt_pairs = [(args.target, s) for s in template_subs]
    acc = {(kind, blk): GramAccumulator(pcols, tpl_pairs if kind == "template" else tgt_pairs, args.target)
           for kind in ("template", "target") for blk in BLOCKS}

    # film blocks
    t1 = time.time()
    for m in template_subs:
        acc[("template", "film")].add(film_rows(m, films, encoders, template_subs))
    acc[("target", "film")].add(film_rows(args.target, films, encoders, template_subs))
    seconds["film_blocks"] = round(time.time() - t1, 1)

    # probe blocks, chunked by whole clips
    t1 = time.time()
    tr = float(rows[args.target]["repetition_time"].iat[0])
    ids = enc.probe_ids(paths.cache, args.space)
    n_probe = n_chunks = 0
    for x, groups in probe_batches(lambda sid: enc.probe_design(paths.cache, args.space, tr, [sid])[0],
                                   ids, args.chunk):
        pred = {s: center_blocks(encoders[s].predict(x), groups) for s in subs}
        acc[("template", "probe")].add({s: pred[s] for s in template_subs})
        acc[("target", "probe")].add(pred)
        n_probe += x.shape[0]
        n_chunks += 1
    seconds["probe_blocks"] = round(time.time() - t1, 1)
    log(f"probe: {len(ids)} clips, {n_probe} volumes in {n_chunks} chunks ({seconds['probe_blocks']:.0f} s)")

    # template, then the target into it
    tpl_grams = merge_grams(acc[("template", "film")].g, acc[("template", "probe")].g)
    tgt_grams = merge_grams(acc[("target", "film")].g, acc[("target", "probe")].g)
    lab_list = list(pcols.template)
    rot, cross = gram_template(tpl_grams, template_subs, lab_list)
    cross[args.target] = target_cross(tgt_grams, args.target, template_subs, rot, pcols)
    tcols = {lab: pcols.template[lab][pos] for lab, pos in pcols.target.items()}

    dest = out_dir(paths.derivatives, args.space, args.scenario, args.pct, args.draw, args.target)
    dest.mkdir(parents=True, exist_ok=True)
    for s in subs:
        cols = tcols if s == args.target else pcols.template
        cha.save_cross(as_cross(cross[s], cols), dest / f"cross_sub-{s}.npz")
    if args.save_grams:
        for (kind, blk), a in acc.items():
            save_grams(a.g, dest / f"grams_{kind}_{blk}.npz", pcols)
    seconds["total"] = round(time.time() - t0, 1)
    side = {
        "description": "Stimulus route, one partition job: per-piece cross-products (X_s' template) for every "
                       "subject; procrustes.transform_from_cross(cha.load_cross(path), n, lam) gives the "
                       "transform into the template for any lam.",
        "space": args.space, "job": {"scenario": args.scenario, "pct": args.pct, "draw": args.draw,
                                      "target": args.target},
        "template_subjects": template_subs,
        "films": {f"sub-{s}": sorted(rows[s]["stimulus_id"]) for s in subs},
        "rows": {f"{k}_{b}": a.n_rows for (k, b), a in acc.items()},
        "pairing": "subject side measured on its own films (z-scored per window), predicted elsewhere; "
                   "template side always predicted; predictions centred per window/clip, not rescaled",
        "encoders": {f"sub-{s}": encoders[s].diagnostics_ for s in subs},
        "n_pieces": {"template": len(pcols.template), "target": len(pcols.target)},
        "n_columns": {"template": int(sum(c.size for c in pcols.template.values())),
                      "target": int(sum(p.size for p in pcols.target.values()))},
        "template_iterations": TEMPLATE_ITERATIONS,
        "alignment_diagnostics": {f"sub-{s}": alignment_diagnostics(cross[s]) for s in subs},
        "alignment_diagnostics_note": "alignment data only, not a score: 'captured' = aligned over anatomical "
                                      "cross-covariance with the template at lam 0; 'tr_over_p' = mean tr(R)/p "
                                      "per lam",
        "n_grayordinates": int(len(table)), "backend": args.backend, "probe_chunk": args.chunk,
        "grams_saved": bool(args.save_grams),
        "code_version": _code_version(),
        "seconds": seconds,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (dest / "stimulus.json").write_text(json.dumps(side, indent=2) + "\n")
    log(f"wrote {dest} in {seconds['total']:.0f} s")
    return dest


def save_grams(g: dict, path: Path, pcols: PieceColumns | None = None) -> None:
    """Per-pair, per-piece Grams (float32); ``pcols`` also stores the columns they index."""
    arrays, index = {}, []
    for i, ((a, b), per_piece) in enumerate(g.items()):
        for j, (lab, m) in enumerate(per_piece.items()):
            key = f"g{i}_{j}"
            arrays[key] = m.astype(np.float32)
            index.append((a, b, str(lab), key))
    arrays["index"] = np.array(index, dtype=object)
    if pcols is not None:
        labs = list(pcols.template)
        arrays["pcols_labels"] = np.array([str(lab) for lab in labs], dtype=object)
        for j, lab in enumerate(labs):
            arrays[f"pcols_t{j}"] = pcols.template[lab].astype(np.int32)
            arrays[f"pcols_p{j}"] = pcols.target.get(lab, np.array([], np.int64)).astype(np.int32)
    np.savez(path, **arrays)


def load_grams(path: Path) -> tuple[dict, PieceColumns | None]:
    """``save_grams``'s inverse: ``({(a, b): {label: G float64}}, columns or None)``; labels are ``str``."""
    z = np.load(path, allow_pickle=True)
    g: dict = {}
    for a, b, lab, key in z["index"]:
        g.setdefault((str(a), str(b)), {})[str(lab)] = z[key].astype(np.float64)
    pcols = None
    if "pcols_labels" in z.files:
        tpl, tgt = {}, {}
        for j, lab in enumerate(z["pcols_labels"]):
            tpl[str(lab)] = z[f"pcols_t{j}"].astype(np.int64)
            pos = z[f"pcols_p{j}"].astype(np.int64)
            if pos.size:
                tgt[str(lab)] = pos
        pcols = PieceColumns(tpl, tgt)
    return g, pcols


def _code_version() -> str:
    from neuroimaging import data_quality as dq

    return dq.code_version(REPO_ROOT)


def cmd_plan(args: argparse.Namespace) -> None:
    import encoding as enc
    import films as fm
    import partitions as pt

    paths = enc.Paths()
    rows = job_films(pt.load_partitions(paths.derivatives), fm.load_windows(paths.derivatives),
                     args.scenario, args.pct, args.draw, args.target)
    for s, r in sorted(rows.items()):
        role = "target" if s == args.target else "template"
        print(f"sub-{s} ({role}): {len(r)} films, {int(r['n'].sum())} volumes")
    ids = enc.probe_ids(paths.cache, args.space)
    print(f"probe: {len(ids)} cached clips ({args.space}); chunks of ~{args.chunk} volumes")
    print(f"out: {out_dir(paths.derivatives, args.space, args.scenario, args.pct, args.draw, args.target)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    for verb in ("plan", "fit"):
        p = sub.add_parser(verb)
        p.add_argument("--space", choices=("ebind", "vgg19"), required=True)
        p.add_argument("--scenario", default="primary", choices=("primary", "secondary"))
        p.add_argument("--pct", type=int, required=True)
        p.add_argument("--draw", type=int, required=True)
        p.add_argument("--target", required=True)
        p.add_argument("--chunk", type=int, default=PROBE_CHUNK)
        if verb == "fit":
            p.add_argument("--backend", default="torch_cuda")
            p.add_argument("--save-grams", action="store_true",
                           help="also keep the per-block Grams (for a jointly built combined template)")
    args = ap.parse_args()
    {"plan": cmd_plan, "fit": run_job}[args.verb](args)


if __name__ == "__main__":
    main()
