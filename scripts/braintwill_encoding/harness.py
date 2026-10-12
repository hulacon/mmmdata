#!/usr/bin/env python3
"""MMMData registers for braintwill's encoding harness (``braintwill.encoding``).

braintwill must not know this dataset exists, so everything that resolves a path, a film
window or a session lives here, and braintwill is imported as a library. The harness scores
any feature set against a register's fixed responses by nested grouped CV and caches the
result by (feature-set content, register spec, subject); see braintwill's
``braintwill/encoding/`` for the method.

Registers:

  films   each subject's 46-film alignment pool from ``functional_space/films.py``: the
          cleaned grayordinate series (89,820 columns) over each film's window, z-scored per
          window; FIR delays, α grid, fold rule and band-weight search are functional-space's
          (``functional_space/encoding.py``), so the film path is a refactor of those
          encoders, not a redesign. Groups are films. **ses-23 and ses-26 never enter**: only
          ``role == "alignment"`` rows are read, and a row from a held-out session is an
          error even if a later table change gave it that role.

Feature sets are the per-stimulus caches ``functional_space/encoding.py cache`` writes
(``<derivatives>/functional_space/features/<space>/``); only the register's films are read,
and their content hash is the cache key's feature half.

Scores go to ``<derivatives>/encoding_harness/register-<name>/spec-<hash>/features-<hash>/
sub-##/`` (``scores.npz`` + ``encoding.json``). A cached score is read back, never refit.

Verbs:

  plan         build one subject's register and report sizes, key and cache state
  score        nested-CV scores for one subject (cached)
  equivalence  refit one functional-space stimulus-route job's encoders through braintwill on
               that job's films and compare the fit diagnostics with the job's sidecar; also
               checks the design and responses equal ``encoding.film_design``'s exactly

Usage:
    python harness.py plan --space ebind --sub ##
    python harness.py score --space psytwill --sub ## --backend torch_cuda
    python harness.py equivalence --space ebind --scenario primary --pct 0 --draw 0 --target ##
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_FS = _REPO / "scripts" / "functional_space"
for _p in (_REPO / "src" / "python", _FS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _import_braintwill():
    """braintwill from the env, the sibling checkout, or ``$BRAINTWILL_SRC``; loud otherwise."""
    try:
        import braintwill.encoding  # noqa: F401
        return
    except ImportError:
        pass
    src = Path(os.environ.get("BRAINTWILL_SRC", _REPO.parent / "braintwill"))
    if not (src / "braintwill" / "encoding" / "__init__.py").exists():
        raise ImportError(f"braintwill.encoding not importable and no checkout at {src}; "
                          "set BRAINTWILL_SRC to the braintwill repository root")
    sys.path.insert(0, str(src))


_import_braintwill()

import encoding as fs_enc  # noqa: E402  functional_space/encoding.py: feature cache, settings
import films as fm  # noqa: E402
import grayordinates as go  # noqa: E402
from braintwill.encoding import (BandedRidge, EncodingSpec, FrameFeatures, RidgeSpec,  # noqa: E402
                                 ScoreCache, cache_key, feature_set_hash, fir_design, score)

TREE = "encoding_harness"
#: Outer folds of the nested CV (DECIDED 2026-10-12, encoding-harness settle 1).
OUTER_FOLDS = 5


def film_spec(solver: str = "random_search", n_iter: int | None = None) -> EncodingSpec:
    """functional-space's encoder settings, wrapped in the outer CV.

    ``random_search`` with functional-space's ``N_ITER_BANDS`` is the default and the only
    setting the equivalence covers; ``hyper_gradient`` is opt-in for fits with many bands.
    """
    ridge = RidgeSpec(alphas=tuple(fs_enc.ALPHAS), inner_folds=fs_enc.N_FOLDS, solver=solver,
                      n_iter=fs_enc.N_ITER_BANDS if n_iter is None else n_iter, random_state=0)
    return EncodingSpec(ridge=ridge, outer_folds=OUTER_FOLDS)


class Paths:
    def __init__(self) -> None:
        p = fs_enc.Paths()
        self.derivatives = p.derivatives
        self.features = p.cache
        self.cleaned = go.tree_root(p.derivatives)
        self.out = p.derivatives / TREE


# ---------------------------------------------------------------------------
# the films register
# ---------------------------------------------------------------------------

def alignment_rows(windows: pd.DataFrame, sub: str, films: list[str] | None = None) -> pd.DataFrame:
    """One subject's alignment-pool window rows (all of them, or the named films).

    The held-out sessions are refused here, not just filtered: a held-out row that reached
    this point with an alignment role would mean the role table changed under the harness.
    """
    rows = windows[(windows["sub"] == sub) & (windows["role"] == "alignment")]
    leaked = rows[rows["ses"].isin(fm.HELDOUT_SESSIONS)]
    if len(leaked):
        raise RuntimeError(f"sub-{sub}: {len(leaked)} alignment rows come from held-out sessions "
                           f"{sorted(set(leaked['ses']))}; the films register must never read them")
    if rows["stimulus_id"].duplicated().any():
        raise RuntimeError(f"sub-{sub}: an alignment film appears twice")
    if films is None:
        if rows["stimulus_id"].nunique() != fm.N_POOL:
            raise RuntimeError(f"sub-{sub}: {rows['stimulus_id'].nunique()} alignment films, "
                               f"expected the pool of {fm.N_POOL}")
        return rows
    missing = sorted(set(films) - set(rows["stimulus_id"]))
    if missing:
        raise RuntimeError(f"sub-{sub}: films not in the alignment pool: {missing}")
    return rows[rows["stimulus_id"].isin(films)]


def frame_features(features_root: Path, space: str, ids) -> list[FrameFeatures]:
    out = []
    for sid in ids:
        f = fs_enc.load_features(fs_enc.cache_path(features_root, space, sid))
        out.append(FrameFeatures(f.stimulus_id, f.times, f.x, fs_enc.FRAME_S, f.bands))
    return out


def film_design(rows: pd.DataFrame, feats: dict[str, FrameFeatures], cleaned: Path):
    """FIR design, per-window z-scored responses and film groups, in row order."""
    xs, ys, groups, bands, runs = [], [], [], None, {}
    for r in rows.itertuples(index=False):
        tr = float(r.repetition_time)
        starts = fs_enc.film_volume_starts(r.onset, r.start, r.n, tr)
        x, bands = fir_design(feats[r.stimulus_id], starts, tr, fs_enc.DELAYS_TR, end=r.play_s)
        xs.append(x)
        ys.append(fm.film_series(r, cleaned, runs))
        groups += [r.stimulus_id] * int(r.n)
    groups = np.array(groups)
    y = fs_enc._zscore_film_windows(np.concatenate(ys), groups)
    return np.concatenate(xs), y, groups, bands


def response_provenance(windows: pd.DataFrame, cleaned: Path) -> dict:
    """What fixes the films register's responses, hashed.

    Every subject's alignment window rows (so one spec serves all subjects) and the cleaned
    series manifest rows they read (regime version, code version, creation time).
    """
    rows = windows[windows["role"] == "alignment"].sort_values(["sub", "stimulus_id"])
    cols = ["sub", "ses", "run", "stimulus_id", "onset", "play_s", "repetition_time", "start", "n",
            "series"]
    manifest = pd.read_csv(Path(cleaned) / "manifest.tsv", sep="\t", dtype=str)
    used = manifest[manifest["path"].isin(set(rows["series"]))]
    if len(used) != rows["series"].nunique():
        raise RuntimeError(f"{len(used)} manifest rows for {rows['series'].nunique()} series")
    used = used.sort_values("path")[["path", "regime", "regime_version", "code_version", "created"]]
    h = hashlib.sha256(rows[cols].to_csv(index=False).encode() + used.to_csv(index=False).encode())
    return {"windows_sha256": h.hexdigest(), "n_series": int(len(used)),
            "regime_versions": sorted(set(used["regime_version"])),
            "code_versions": sorted(set(used["code_version"]))}


def film_register(windows: pd.DataFrame, cleaned: Path, films: list[str]) -> dict:
    trs = sorted(set(windows.loc[windows["role"] == "alignment", "repetition_time"].astype(float)))
    if len(trs) != 1:
        raise RuntimeError(f"alignment windows have several TRs {trs}")
    n_columns = len(pd.read_csv(go.grayordinates_path(cleaned), sep="\t"))
    return {
        "name": "films", "design": "fir", "tr": trs[0], "frame_s": fs_enc.FRAME_S,
        "delays_tr": list(fs_enc.DELAYS_TR), "groups": "film",
        "responses": {"tree": str(Path(cleaned).name), "columns": "grayordinates",
                      "n_columns": n_columns, "standardize": "z-score per film window",
                      **response_provenance(windows, cleaned)},
        "heldout_sessions_excluded": list(fm.HELDOUT_SESSIONS),
        "films": sorted(films),
    }


def _git(repo: Path, paths: tuple[str, ...]) -> dict:
    """HEAD, and whether the paths this code runs from have uncommitted changes.

    Scoped to those paths: the checkouts are shared with other sessions, whose unrelated
    uncommitted files would otherwise mark every run dirty and make the flag meaningless.
    """
    def run(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()
    return {"head": run("rev-parse", "--short", "HEAD"), "dirty_paths": list(paths),
            "dirty": bool(run("status", "--porcelain", "--", *paths))}


def provenance(backend: str, seconds: dict) -> dict:
    import braintwill

    bt = Path(braintwill.__file__).resolve().parents[1]
    mmm = ("scripts/braintwill_encoding", "scripts/functional_space", "src/python/core")
    return {"backend": backend, "seconds": seconds, "mmmdata": _git(_REPO, mmm),
            "braintwill": _git(bt, ("braintwill/encoding", "braintwill/ledger.py", "pyproject.toml")),
            "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}


def _ensure_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    dd = root / "dataset_description.json"
    if not dd.exists():
        dd.write_text(json.dumps({
            "Name": "encoding harness scores", "BIDSVersion": "1.10.0", "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "braintwill.encoding",
                             "Description": "nested grouped-CV banded ridge, per-column held-out R²; "
                                            "driver mmmdata scripts/braintwill_encoding/harness.py"}],
        }, indent=2) + "\n")


def build(args, paths: Paths):
    windows = fm.load_windows(paths.derivatives)
    films = args.films.split(",") if getattr(args, "films", None) else None
    rows = alignment_rows(windows, args.sub, films)
    feats = {f.stimulus_id: f for f in frame_features(paths.features, args.space, rows["stimulus_id"])}
    register = film_register(windows, paths.cleaned, list(rows["stimulus_id"]))
    spec = film_spec(args.solver, args.n_iter)
    key = cache_key(feature_set_hash(feats.values()), register, spec, f"sub-{args.sub}")
    return rows, feats, register, spec, key


def _label(paths: Paths, space: str) -> dict:
    side = paths.features / space / "cache.json"
    return {"space": space, "cache": str(paths.features / space),
            "cache_json": json.loads(side.read_text()) if side.exists() else None}


def cmd_plan(args) -> None:
    paths = Paths()
    rows, feats, register, spec, key = build(args, paths)
    cache = ScoreCache(paths.out)
    first = next(iter(feats.values()))
    print(json.dumps({"sub": args.sub, "space": args.space, "films": len(rows),
                      "volumes": int(rows["n"].sum()),
                      "bands": {b: (sl.stop - sl.start) * len(fs_enc.DELAYS_TR) for b, sl in first.bands.items()},
                      "spec_hash": key["spec_hash"], "feature_hash": key["feature_hash"],
                      "path": str(cache.path(key)), "cached": cache.get(key) is not None}, indent=2))


def cmd_score(args) -> None:
    paths = Paths()
    rows, feats, register, spec, key = build(args, paths)
    cache = ScoreCache(paths.out)
    hit = cache.get(key)
    if hit is not None:
        print(f"cached: {cache.path(key)}\n{json.dumps(hit.summary)}")
        return
    t0 = time.time()
    x, y, groups, bands = film_design(rows, feats, paths.cleaned)
    t_load = time.time() - t0
    result = score(x, y, groups, bands, spec, backend=args.backend)
    seconds = {"load": round(t_load, 1), "score": round(time.time() - t0 - t_load, 1)}
    _ensure_root(paths.out)
    dest = cache.put(key, result, provenance(args.backend, seconds), _label(paths, args.space))
    print(f"wrote {dest}\n{json.dumps(result.summary)}\nfull fit {json.dumps(result.full_fit)}")


# ---------------------------------------------------------------------------
# equivalence with functional-space's encoders (settle 2)
# ---------------------------------------------------------------------------

#: How close a braintwill full-data fit must come to a functional-space job's recorded
#: encoder diagnostics. Counts are exact. The sidecar rounds CV quantiles to 4 dp, so they
#: may differ by one unit there. Fractions of columns at a grid edge may differ by GPU
#: float noise flipping a column's chosen α between neighbours: ≤ 0.001 (~80 columns).
TOLERANCE = {"exact": ("n_samples", "n_features", "n_targets", "alpha_median"),
             "cv_quantile_abs": 1e-4, "edge_fraction_abs": 1e-3, "cv_neg_mse_median_abs": 1e-5}


def compare(ours: dict, ref: dict) -> dict:
    out, ok = {}, True
    for k in TOLERANCE["exact"]:
        out[k] = {"ours": ours[k], "ref": ref[k], "pass": ours[k] == ref[k]}
    for k in ("alpha_at_grid_edge", "alpha_at_low_edge", "alpha_at_high_edge"):
        d = abs(ours[k] - ref[k])
        out[k] = {"ours": ours[k], "ref": ref[k], "abs_diff": d, "pass": d <= TOLERANCE["edge_fraction_abs"]}
    d = abs(ours["cv_neg_mse_median"] - ref["cv_neg_mse_median"])
    out["cv_neg_mse_median"] = {"ours": ours["cv_neg_mse_median"], "ref": ref["cv_neg_mse_median"],
                                "abs_diff": d, "pass": d <= TOLERANCE["cv_neg_mse_median_abs"]}
    q = np.abs(np.subtract(ours["cv_1_plus_score_q50_q95_q99"], ref["cv_1_plus_score_q50_q95_q99"]))
    out["cv_1_plus_score_q50_q95_q99"] = {"ours": ours["cv_1_plus_score_q50_q95_q99"],
                                          "ref": ref["cv_1_plus_score_q50_q95_q99"],
                                          "abs_diff": q.tolist(),
                                          "pass": bool(np.all(q <= TOLERANCE["cv_quantile_abs"] + 1e-9))}
    ok = all(v["pass"] for v in out.values())
    return {"pass": bool(ok), "fields": out}


def cmd_equivalence(args) -> None:
    import partitions as pt
    import stimulus_route as sr

    paths = Paths()
    job_dir = sr.out_dir(paths.derivatives, args.space, args.scenario, args.pct, args.draw, args.target)
    side = json.loads((job_dir / "stimulus.json").read_text())
    windows = fm.load_windows(paths.derivatives)
    rows = sr.job_films(pt.load_partitions(paths.derivatives), windows, args.scenario, args.pct, args.draw,
                        args.target)
    spec = film_spec()
    record = {"space": args.space, "job": side["job"], "job_code_version": side["code_version"],
              "job_backend": side["backend"], "tolerance": TOLERANCE, "subjects": {}}
    for s in sorted(rows):
        r = alignment_rows(windows, s, list(rows[s]["stimulus_id"]))  # the held-out guard, same row order
        feats = {f.stimulus_id: f for f in frame_features(paths.features, args.space, r["stimulus_id"])}
        x, y, groups, bands = film_design(r, feats, paths.cleaned)
        # the design and responses are functional-space's, bit for bit
        x0, y0, g0, b0 = fs_enc.film_design(r, paths.features, args.space, paths.cleaned)
        y0 = fs_enc._zscore_film_windows(y0, g0)
        same = {"x": bool(np.array_equal(x, x0)), "y": bool(np.array_equal(y, y0, equal_nan=True)),
                "groups": bool(np.array_equal(groups, g0)), "bands": bands == b0}
        t0 = time.time()
        fit = BandedRidge(bands, spec.ridge, args.backend).fit(x, y, groups)
        ours = fit.diagnostics(x.shape[0], x.shape[1])
        cmp = compare(ours, side["encoders"][f"sub-{s}"])
        cmp["design_identical"] = same
        cmp["pass"] = cmp["pass"] and all(same.values())
        cmp["seconds"] = round(time.time() - t0, 1)
        record["subjects"][f"sub-{s}"] = cmp
        print(f"sub-{s}: pass={cmp['pass']} design={same} "
              f"cv q={ours['cv_1_plus_score_q50_q95_q99']} vs {side['encoders'][f'sub-{s}']['cv_1_plus_score_q50_q95_q99']}",
              flush=True)
    record["pass"] = all(v["pass"] for v in record["subjects"].values())
    record["provenance"] = provenance(args.backend, {})
    _ensure_root(paths.out)
    dest = paths.out / "equivalence" / (f"{args.space}_{args.scenario}_pct-{args.pct:03d}_draw-{args.draw:02d}"
                                        f"_target-{args.target}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{'PASS' if record['pass'] else 'FAIL'} -> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    for name in ("plan", "score"):
        p = sub.add_parser(name)
        p.add_argument("--space", choices=list(fs_enc.FEATURE_SPACES), required=True)
        p.add_argument("--sub", required=True)
        p.add_argument("--films", help="comma-separated subset of the alignment pool (default: all of it)")
        p.add_argument("--solver", choices=("random_search", "hyper_gradient"), default="random_search")
        p.add_argument("--n-iter", type=int, default=None)
        p.add_argument("--backend", default="torch_cuda")
    e = sub.add_parser("equivalence")
    e.add_argument("--space", choices=list(fs_enc.FEATURE_SPACES), required=True)
    e.add_argument("--scenario", default="primary")
    e.add_argument("--pct", type=int, default=0)
    e.add_argument("--draw", type=int, default=0)
    e.add_argument("--target", required=True)
    e.add_argument("--backend", default="torch_cuda")
    args = ap.parse_args()
    {"plan": cmd_plan, "score": cmd_score, "equivalence": cmd_equivalence}[args.verb](args)


if __name__ == "__main__":
    main()
