#!/usr/bin/env python3
"""Shared vs unique subspaces of a psytwill block, by held-out brain variance at matched dimension.

The read of model-readout settle 3 (shared vs unique subspaces). Every object is scored alone by the
harness (``harness.py subspaces``: one ``sub-<B>/<object>`` unit per object and subject, held-out
films R² per column, with per-group sums). This driver fits nothing. The objects come from
psytwill ``space subspaces``, which also wrote the draws:

  S, U, U:<m>      the shared subspace, the unique union, each member's unique subspace
  S@dNNN, U@dNNN   draw NNN of a d-dimensional span inside S and inside U (d = min of their
                   dims; paired by NNN)
  ref@dNNN         a random d-dimensional span of the block's coordinates (the reference)
  S@<m>.dNNN       a dim(U:<m>)-dimensional span inside S

Per subject, block and ROI (HCP-MMP1's 22 sections, ``families.familyD``, plus the whole mask,
``rescue.ALL_ROI``), with ROI R² the mean over the ROI's masked, valid columns, it reports:

  delta            mean over draws of R²(S@d) − R²(U@d): the block contrast
  S_draws, U_draws, ref
                   each side's mean over its draws, and the reference's
  S_vs_ref, U_vs_ref
                   each side's draw mean minus the reference's
  S, U             the whole subspaces (unequal dims; descriptive)
  U:<m>, U:<m>_vs_S
                   per member: its unique subspace, and minus the mean of its S draws

Each carries a paired film-bootstrap interval (``READ``: draws, seed; one draw of counts per
subject serves every object) and its bootstrap sd. Every draw's ΔR² is written out.

**Gate.** The block contrast is read only when S and U both passed ``space subspaces``'s split-half
gate; a member's rows only when its U:<m> did. A failed object is reported as such, not read.

**Pooling.** Per block, ``delta`` over the subjects' 22 ROI cells is partially pooled
(braintwill ``pooled_effects``: ROIs as units, fixed subject effects, the bootstrap sd as each
cell's se) at ``READ["pool_ci"]``. An ROI whose pooled interval lies above 0 is flagged
``above_zero``; below 0, ``below_zero`` (the reverse result). The whole-mask ROI stays out of
the pool and is reported per subject. Discovery read: no multiplicity correction (the pooling
is the shrinkage). No verdict is written.

Usage:
    python subspaces.py read --blocks L.<tag> --subs 03,04,05 --out <dir>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import harness as h  # noqa: E402  (puts functional_space and src/python on sys.path)
import rescue as rs  # noqa: E402

#: The read's fixed settings (a different value is a different read, recorded as such).
READ = {"n_boot": 2000, "seed": 0, "cell_ci": 0.95, "pool_ci": 0.90, "family": rs.RULE["family"],
        "min_cols": rs.RULE["min_cols"], "register": "films", "gate": "S and U pass for the block contrast; "
        "U:<m> passes for its member rows"}
DRAW = re.compile(r"(S|U|ref)@d(\d{3})")
MEMBER_DRAW = re.compile(r"S@(.+)\.d(\d{3})")


def plan(objects: dict[str, dict]) -> dict:
    """Which objects the read needs, from the projection meta's objects: draws by index, members."""
    draws: dict[str, dict[int, str]] = {"S": {}, "U": {}, "ref": {}}
    member_draws: dict[str, dict[int, str]] = {}
    for name in objects:
        m = DRAW.fullmatch(name)
        if m:
            draws[m.group(1)][int(m.group(2))] = name
            continue
        m = MEMBER_DRAW.fullmatch(name)
        if m:
            member_draws.setdefault(m.group(1), {})[int(m.group(2))] = name
    idx = sorted(draws["S"])
    if not idx or any(sorted(draws[k]) != idx for k in ("U", "ref")):
        raise SystemExit(f"the projection's S, U and ref draws do not share their indices "
                         f"({ {k: len(v) for k, v in draws.items()} }); re-run space subspaces with --draws")
    members = [n.split(":", 1)[1] for n, o in objects.items() if o["kind"] == "unique" and o["dim"]]
    for m in members:
        if sorted(member_draws.get(m, {})) != idx:
            raise SystemExit(f"member {m}: its S draws do not match the block's {len(idx)} draw indices")
    return {"draw_idx": idx, "draws": draws, "members": members, "member_draws": member_draws}


def combinations(p: dict, gate: dict[str, bool | None]) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """The read's linear combinations of object scores, and why any were left out."""
    n = len(p["draw_idx"])
    S = {p["draws"]["S"][r]: 1.0 / n for r in p["draw_idx"]}
    U = {p["draws"]["U"][r]: 1.0 / n for r in p["draw_idx"]}
    ref = {p["draws"]["ref"][r]: 1.0 / n for r in p["draw_idx"]}

    def minus(a, b):
        out = dict(a)
        for k, v in b.items():
            out[k] = out.get(k, 0.0) - v
        return out

    combos, skipped = {}, {}
    if gate.get("S") and gate.get("U"):
        combos.update({"delta": minus(S, U), "S_draws": S, "U_draws": U, "ref": ref,
                       "S_vs_ref": minus(S, ref), "U_vs_ref": minus(U, ref), "S": {"S": 1.0}, "U": {"U": 1.0}})
    else:
        skipped["block"] = f"gate: S {gate.get('S')}, U {gate.get('U')}"
    for m in p["members"]:
        name = f"U:{m}"
        if not gate.get(name):
            skipped[name] = f"gate: {gate.get(name)}"
            continue
        Sm = {p["member_draws"][m][r]: 1.0 / n for r in p["draw_idx"]}
        combos[name] = {name: 1.0}
        combos[f"{name}_vs_S"] = minus({name: 1.0}, Sm)
    return combos, skipped


def cell_stats(groups: dict, rois: dict[str, np.ndarray], counts: np.ndarray, combos: dict, ci: float) -> list[dict]:
    """Per combination and ROI: observed value, percentile interval, bootstrap sd, two-sided p."""
    from braintwill.encoding import combination_draws

    obs, draws = combination_draws(groups, rois, counts, combos)
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    rows = []
    for c in combos:
        for r in rois:
            bt = draws[c][r]
            tail = min(np.sum(bt <= 0), np.sum(bt >= 0))
            rows.append({"read": c, "roi": r, "value": obs[c][r], "lo": float(np.quantile(bt, lo)),
                         "hi": float(np.quantile(bt, hi)), "sd": float(np.std(bt, ddof=1)),
                         "p_two_sided": float(min(1.0, 2 * (1 + tail) / (bt.size + 1))),
                         "n_cols": int(np.asarray(rois[r]).size)})
    return rows


def draw_deltas(groups: dict, rois: dict[str, np.ndarray], p: dict) -> list[dict]:
    """Every draw's ΔR² (R²(S@dNNN) − R²(U@dNNN)) per ROI, on the observed films."""
    from braintwill.encoding import resampled_r2

    out = []
    for r_idx in p["draw_idx"]:
        a, b = (np.asarray(resampled_r2(groups[p["draws"][k][r_idx]], np.ones(len(groups[p["draws"][k][r_idx]]["labels"]))))
                for k in ("S", "U"))
        for roi, ix in rois.items():
            out.append({"draw": r_idx, "roi": roi, "delta": float(np.nanmean(a[ix]) - np.nanmean(b[ix]))})
    return out


def pool(cells: pd.DataFrame, ci: float) -> pd.DataFrame:
    """``delta`` partially pooled over ROIs with fixed subject effects (whole-mask ROI left out)."""
    from braintwill.encoding import pooled_effects

    rows = []
    for block, c in cells[(cells["read"] == "delta") & (cells["roi"] != rs.ALL_ROI)].groupby("block"):
        if c["roi"].nunique() < 2:
            continue
        pe = pooled_effects(c["value"], c["sd"], c["roi"], c["sub"], ci=ci)
        for roi, u in pe["units"].items():
            rows.append({"block": block, "roi": roi, "mean": u["mean"], "sd": u["sd"], "lo": u["lo"], "hi": u["hi"],
                         "p_positive": u["p_positive"], "raw": u["raw"], "n_cells": u["n_cells"],
                         "above_zero": bool(u["lo"] > 0), "below_zero": bool(u["hi"] < 0),
                         "tau_median": pe["tau"]["median"], "tau_upper_mass": pe["tau"]["upper_mass"]})
        m = pe["mean"]
        rows.append({"block": block, "roi": "(mean over ROIs)", "mean": m["mean"], "sd": m["sd"], "lo": m["lo"],
                     "hi": m["hi"], "p_positive": m["p_positive"], "raw": None, "n_cells": int(len(c)),
                     "above_zero": bool(m["lo"] > 0), "below_zero": bool(m["hi"] < 0),
                     "tau_median": pe["tau"]["median"], "tau_upper_mass": pe["tau"]["upper_mass"]})
    return pd.DataFrame(rows)


def subject_read(paths, cache, block: str, sub: str, labels: np.ndarray) -> dict:
    """One block and subject: every needed unit's group sums, the ROIs, the cells and the draws."""
    from braintwill.encoding import draw_counts

    st = h.subspace_setup(paths, block, sub)
    objs = st["meta"]["subspaces"]["objects"]
    p = plan(objs)
    gate = {n: o.get("gate_passed") for n, o in objs.items()}
    combos, skipped = combinations(p, gate)
    need = sorted({s for w in combos.values() for s in w})
    groups, valid, units = {}, None, {}
    for obj in need:
        _, key = h.subspace_key(st, block, obj, objs[obj], sub)
        got = cache.get(key)
        if got is None:
            raise SystemExit(f"{h.subspace_space(block, obj)} sub-{sub} is not scored: {cache.path(key)}; run "
                             f"harness.py subspaces --block {block} --sub {sub}")
        if got.groups is None:
            raise SystemExit(f"{cache.path(key)} has no per-group sums")
        groups[obj] = got.groups
        valid = got.valid if valid is None else valid & got.valid
        units[obj] = str(cache.path(key))
    mask = np.asarray(st["mask"], bool)
    rois = rs.roi_columns(labels, mask, valid, READ["min_cols"]) if need else {}
    if need:
        rois[rs.ALL_ROI] = np.flatnonzero(mask & valid)
    n_groups = len(next(iter(groups.values()))["labels"]) if groups else 0
    counts = draw_counts(n_groups, READ["n_boot"], READ["seed"]) if groups else None
    cells = cell_stats(groups, rois, counts, combos, READ["cell_ci"]) if combos else []
    draws = draw_deltas(groups, rois, p) if "delta" in combos else []
    return {"cells": cells, "draws": draws, "skipped": skipped, "units": units,
            "n_draws": len(p["draw_idx"]), "members": p["members"], "gate": gate,
            "projection": str(st["proj"]), "mask": str(st["mask_path"])}


def cmd_read(args) -> None:
    import families as fam
    import grayordinates as go
    from braintwill.encoding import ScoreCache

    import cha

    blocks = [b for b in args.blocks.split(",") if b]
    paths = h.Paths()
    cache = ScoreCache(paths.out)
    grid = pd.read_csv(go.grayordinates_path(paths.cleaned), sep="\t")
    labels = np.asarray(fam.family_labels(grid, cha.Paths().atlases)[READ["family"]])
    subs = args.subs.split(",") if args.subs else list(h.fm.SUBJECTS)
    cell_rows, draw_rows, entry = [], [], {}
    for b in blocks:
        for sub in subs:
            g = subject_read(paths, cache, b, sub, labels)
            cell_rows += [{"block": b, "sub": sub, **r} for r in g["cells"]]
            draw_rows += [{"block": b, "sub": sub, **r} for r in g["draws"]]
            entry.setdefault(b, {})[f"sub-{sub}"] = {k: g[k] for k in ("skipped", "units", "n_draws", "members",
                                                                       "gate", "projection", "mask")}
            print(f"sub-{sub} {b}: {len(g['units'])} units, {len(g['cells'])} cells, skipped {g['skipped']}",
                  flush=True)
    cells = pd.DataFrame(cell_rows)
    pooled = pool(cells, READ["pool_ci"]) if len(cells) else pd.DataFrame()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"subspaces_{'+'.join(blocks)}_{READ['register']}"
    cells.to_csv(f"{stem}_cells.csv", index=False)
    pd.DataFrame(draw_rows).to_csv(f"{stem}_draws.csv", index=False)
    pooled.to_csv(f"{stem}_pooled.csv", index=False)
    side = {"read": READ, "pooled_roi_excluded": rs.ALL_ROI, "blocks": entry, "subjects": subs,
            "provenance": h.provenance("numpy", {}),
            "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    Path(f"{stem}.json").write_text(json.dumps(side, indent=2, default=str) + "\n")
    print(f"wrote {stem}.json, {stem}_cells.csv, {stem}_draws.csv, {stem}_pooled.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("read")
    r.add_argument("--blocks", required=True, help="comma-separated projectable blocks (V, A, L or <letter>.<tag>)")
    r.add_argument("--subs", help="comma-separated subjects (default: the harness's)")
    r.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()
    {"read": cmd_read}[args.verb](args)


if __name__ == "__main__":
    main()
