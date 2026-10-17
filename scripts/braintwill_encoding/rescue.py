#!/usr/bin/env python3
"""Brain rescue test for psytwill-space members (psytwill-space-next settle 2).

A member the rest of its block covers is pruned unless it adds unique brain variance. The
test (mmmdata-agents psytwill-space-next, DECIDED 2026-10-16) is an add-one ΔR²: per subject
and register, the reduced block's scores alone (``rescue-<B>``) against the reduced block
plus the member (``rescue-<B>+<m>``), both scored by the harness (``harness.py score``) with
per-group sums kept. This driver writes the reduced block's projection the harness reads and
applies the rule to the cached scores. It fits nothing itself.

Verbs:

  project  the reduced block's fit (a ``psytwill space fit`` manifest) projected onto one
           corpus: ``movies`` (the films' frames, 0.5 s grid) or ``shared1000`` (images, V
           only) -> ``<derivatives>/encoding_harness/rescue/projection/<corpus>_<B>.parquet``
           plus a ``.json`` sidecar pinning the manifest and weights by sha256
  read     one register's rule over every subject for a block and its members -> JSON + CSV

The rule (``RULE``): per subject and ROI (HCP-MMP1's 22 sections, ``families.familyD``), the
mean ΔR² over columns in the ROI and in the register's reliability mask; a one-sided
bootstrap p over the register's CV groups (films; items or whole triplets on tb-enc),
paired (one draw of counts per subject, used for the base and every member), Holm-adjusted
across the ROIs scored for that member and subject. A member is rescued on the register if
one ROI has adjusted p < ``alpha`` in at least ``min_subjects`` subjects **and** the median
over subjects of its ROI-mean ΔR² is at least ``floor``. An ROI with fewer than ``min_cols``
masked columns in a subject is not scored for that subject.

Usage:
    python rescue.py project --block V --corpus movies --manifest <fit>.json
    python rescue.py read --block V --members m1,m2 --register films --out <dir>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import harness as h  # noqa: E402  (puts functional_space and src/python on sys.path)

#: The decided rule (psytwill-space-next DECIDED 2026-10-16). A different cut is a different
#: read, recorded as such; these are not tuning knobs.
RULE = {"family": "familyD", "alpha": 0.025, "min_subjects": 2, "floor": 0.002,
        "min_cols": 20, "n_boot": 2000, "seed": 0, "holm": "across scored ROIs, per member x subject"}
#: Draws resampled at once: bounds memory at ~draws x masked columns x 8 bytes per sum.
DRAW_CHUNK = 200
#: The film window grid ``harness`` builds bands on.
WINDOW_S = h.fs_enc.FRAME_S


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------

def projection_inputs(stimfeat: Path, block: str, corpus: str) -> tuple[Path, list[str]]:
    """The store table a block's members are read from on ``corpus``, and the key flags."""
    if corpus == "movies":
        table = h.RESCUE_FILM_TABLE[block]
        return (Path(stimfeat) / "psytwill" / f"{table}_features.parquet",
                ["--key", "stimulus_id,time", "--window", str(WINDOW_S)])
    if corpus == "shared1000":
        if block not in h.RESCUE_TB_BLOCKS:
            raise SystemExit(f"block {block} is not scored on tb-enc (image-keyed only)")
        return Path(stimfeat) / "psytwill" / f"{h.RESCUE_TB_TABLE}_features.parquet", ["--key", "stimulus_id"]
    raise SystemExit(f"corpus {corpus!r}: one of {sorted(h.RESCUE_CORPUS.values())}")


def cmd_project(args) -> None:
    from core.config import load_config

    paths = h.Paths()
    psytwill = Path(load_config()["paths"]["stimfeat_env"]) / "bin" / "psytwill"
    manifest = Path(args.manifest).resolve()
    m = json.loads(manifest.read_text())
    if m.get("block") != args.block:
        raise SystemExit(f"{manifest} is block {m.get('block')!r}, not {args.block}")
    weights = manifest.with_name(m["weights"])
    if not weights.exists():
        raise SystemExit(f"{weights} (the manifest's weights) is missing")
    features, key = projection_inputs(paths.stimfeat, args.block, args.corpus)
    if not features.exists():
        raise SystemExit(f"{features} is missing")
    out = h.rescue_projection(paths.out, args.block, args.corpus)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; a projection is replaced only with --force")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(psytwill), "space", "project", "--features", str(features), *key, "--space", str(manifest),
           "--chunk-size", str(args.chunk_size), "-o", str(out)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    version = subprocess.run([str(psytwill), "--version"], check=True, capture_output=True, text=True).stdout.strip()
    side = {
        "block": args.block, "corpus": args.corpus, "k": m["k"], "members": m["members"],
        "outside_targets": m.get("outside_targets"),
        "manifest": {"path": str(manifest), "sha256": sha256(manifest)},
        "weights": {"path": str(weights), "sha256": sha256(weights)},
        "features": {"path": str(features), "bytes": features.stat().st_size,
                     "mtime": _dt.datetime.fromtimestamp(features.stat().st_mtime, _dt.timezone.utc).isoformat()},
        "view": "covariance (psytwill space project default)",
        "psytwill_version": version, "command": cmd,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    out.with_suffix(".json").write_text(json.dumps(side, indent=2) + "\n")
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# the rule (pure: synthetic group sums test it)
# ---------------------------------------------------------------------------

def draw_counts(n_groups: int, n_boot: int, seed: int) -> np.ndarray:
    """``(n_boot, n_groups)`` bootstrap counts over groups; one draw serves every fit."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=n_boot)


def _restrict(groups: dict, cols: np.ndarray) -> dict:
    return {"labels": groups["labels"], "n": groups["n"],
            **{k: np.asarray(groups[k])[:, cols] for k in ("sy", "syy", "res")}}


def roi_deltas(base: dict, member: dict, rois: dict[str, np.ndarray], counts: np.ndarray
               ) -> dict[str, dict]:
    """Per ROI (column indices): observed mean ΔR² and its one-sided bootstrap p.

    ``base`` and ``member`` are ``EncodingScore.groups`` of the same responses; their group
    labels must agree, in order, so one row of ``counts`` resamples both alike.
    """
    from braintwill.encoding import resampled_r2

    if list(base["labels"]) != list(member["labels"]):
        raise RuntimeError("base and member scores hold different CV groups")
    if not rois:
        return {}
    cols = np.unique(np.concatenate(list(rois.values())))
    at = {c: i for i, c in enumerate(cols)}
    idx = {r: np.array([at[c] for c in v]) for r, v in rois.items()}
    b, m = _restrict(base, cols), _restrict(member, cols)
    ones = np.ones(len(base["labels"]))
    obs = resampled_r2(m, ones) - resampled_r2(b, ones)
    boot = {r: [] for r in rois}
    for s in range(0, counts.shape[0], DRAW_CHUNK):
        c = counts[s:s + DRAW_CHUNK]
        d = resampled_r2(m, c) - resampled_r2(b, c)
        for r, ix in idx.items():
            boot[r].append(d[:, ix].mean(axis=1))
    out = {}
    for r, ix in idx.items():
        bt = np.concatenate(boot[r])
        out[r] = {"delta": float(obs[ix].mean()), "p": float((1 + np.sum(bt <= 0)) / (bt.size + 1)),
                  "n_cols": int(ix.size)}
    return out


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p values (monotone, capped at 1)."""
    keys = sorted(p, key=lambda k: p[k])
    n, run, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (n - i) * p[k]))
        out[k] = run
    return out


def verdict(per_subject: dict[str, dict[str, dict]], rule: dict = RULE) -> dict:
    """Combine one member's per-subject ROI results (each already Holm-adjusted, key
    ``p_holm``) into the rule's verdict: per ROI the passing subjects and the median ΔR²."""
    rois = sorted({r for d in per_subject.values() for r in d})
    table, rescued_by = {}, []
    for r in rois:
        got = {s: d[r] for s, d in per_subject.items() if r in d}
        passing = sorted(s for s, v in got.items() if v["p_holm"] < rule["alpha"])
        med = float(np.median([v["delta"] for v in got.values()]))
        ok = len(passing) >= rule["min_subjects"] and med >= rule["floor"]
        table[r] = {"subjects_passing": passing, "n_subjects_scored": len(got), "median_delta": med,
                    "pass": ok}
        if ok:
            rescued_by.append(r)
    return {"rescued": bool(rescued_by), "rescued_by": rescued_by, "rois": table}


def roi_columns(labels: np.ndarray, mask: np.ndarray, valid: np.ndarray, min_cols: int) -> dict[str, np.ndarray]:
    """Column indices per ROI label inside ``mask`` and ``valid``; ROIs below ``min_cols`` dropped."""
    keep = np.asarray(mask, bool) & np.asarray(valid, bool)
    out = {}
    for r in sorted(set(labels.tolist()) - {""}):
        ix = np.flatnonzero((labels == r) & keep)
        if ix.size >= min_cols:
            out[r] = ix
    return out


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------

def _cached(paths, cache, register: str, space: str, sub: str):
    req = h.Request(argparse.Namespace(register=register, space=space, sub=sub, films=None,
                                       solver="random_search", n_iter=None), paths)
    got = cache.get(req.key)
    if got is None:
        raise SystemExit(f"{space} sub-{sub} on {register} is not scored: {cache.path(req.key)}; "
                         f"run harness.py score --register {register} --space {space} --sub {sub}")
    if got.groups is None:
        raise SystemExit(f"{cache.path(req.key)} has no per-group sums")
    return req.key, got


def cmd_read(args) -> None:
    import families as fam
    import grayordinates as go
    from braintwill.encoding import ScoreCache, resampled_r2

    import cha

    paths = h.Paths()
    cache = ScoreCache(paths.out)
    members = [m for m in args.members.split(",") if m]
    for m in members:
        h.parse_rescue(f"{h.RESCUE_PREFIX}{args.block}+{m}")
    table = pd.read_csv(go.grayordinates_path(paths.cleaned), sep="\t")
    labels = np.asarray(fam.family_labels(table, cha.Paths().atlases)[RULE["family"]])
    subs = args.subs.split(",") if args.subs else list(h.fm.SUBJECTS)
    rel = paths.out / "reliability" / f"register-{args.register}"
    per = {m: {} for m in members}
    keys, checks = {}, {}
    for sub in subs:
        mask_path = rel / f"sub-{sub}.npz"
        if not mask_path.exists():
            raise SystemExit(f"{mask_path} is missing; run reliability.py for this register")
        with np.load(mask_path) as z:
            mask = z["mask"]
        bkey, base = _cached(paths, cache, args.register, f"{h.RESCUE_PREFIX}{args.block}", sub)
        keys[f"sub-{sub}"] = {"base": bkey["feature_hash"], "spec": bkey["spec_hash"]}
        # the stored per-group sums must give the stored score back before any resample is read
        v = base.valid
        back = resampled_r2(_restrict(base.groups, np.flatnonzero(v)), np.ones(len(base.groups["labels"])))
        checks[f"sub-{sub}"] = float(np.nanmax(np.abs(back - base.r2[v])))
        if checks[f"sub-{sub}"] > 1e-4:
            raise RuntimeError(f"sub-{sub}: per-group sums miss the stored R² by {checks[f'sub-{sub}']:.2g}")
        rois = roi_columns(labels, mask, v, RULE["min_cols"])
        counts = draw_counts(len(base.groups["labels"]), RULE["n_boot"], RULE["seed"])
        for m in members:
            mkey, got = _cached(paths, cache, args.register, f"{h.RESCUE_PREFIX}{args.block}+{m}", sub)
            keys[f"sub-{sub}"][m] = mkey["feature_hash"]
            if not np.array_equal(got.valid, v):
                raise RuntimeError(f"sub-{sub} {m}: valid columns differ from the base's")
            d = roi_deltas(base.groups, got.groups, rois, counts)
            adj = holm({r: x["p"] for r, x in d.items()})
            per[m][sub] = {r: {**x, "p_holm": adj[r]} for r, x in d.items()}
            print(f"sub-{sub} {m}: {len(d)} ROIs; best ΔR² "
                  f"{max((x['delta'] for x in d.values()), default=float('nan')):.4f}", flush=True)
    out = {"block": args.block, "register": args.register, "subjects": subs, "rule": RULE,
           "members": {m: {**verdict(per[m]), "per_subject": per[m]} for m in members},
           "keys": keys, "group_sums_check_max_abs": checks,
           "provenance": h.provenance("read", {}),
           "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    stem = dest / f"rescue_{args.block}_{args.register}"
    stem.with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n")
    rows = [{"member": m, "sub": s, "roi": r, **x} for m in members for s, d in per[m].items() for r, x in d.items()]
    pd.DataFrame(rows).to_csv(stem.with_suffix(".csv"), index=False)
    for m in members:
        v = out["members"][m]
        print(f"{m}: {'RESCUED by ' + ', '.join(v['rescued_by']) if v['rescued'] else 'not rescued'}")
    print(f"wrote {stem}.json, {stem}.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    p = sub.add_parser("project")
    p.add_argument("--block", required=True, choices=sorted(h.RESCUE_FILM_TABLE))
    p.add_argument("--corpus", required=True, choices=sorted(h.RESCUE_CORPUS.values()))
    p.add_argument("--manifest", required=True, help="the reduced block's `psytwill space fit` manifest")
    p.add_argument("--chunk-size", type=int, default=10, help="stimulus_ids per projection chunk")
    p.add_argument("--force", action="store_true")
    r = sub.add_parser("read")
    r.add_argument("--block", required=True, choices=sorted(h.RESCUE_FILM_TABLE))
    r.add_argument("--members", required=True, help="comma-separated members scored as rescue-<B>+<m>")
    r.add_argument("--register", required=True, choices=sorted(h.RESCUE_CORPUS))
    r.add_argument("--subs", help="comma-separated subjects (default: the harness's)")
    r.add_argument("--out", required=True, help="directory for rescue_<B>_<register>.{json,csv}")
    args = ap.parse_args()
    {"project": cmd_project, "read": cmd_read}[args.verb](args)


if __name__ == "__main__":
    main()
