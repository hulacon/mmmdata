#!/usr/bin/env python3
"""Within-subject arm: a subject's own stimulus encoder identifies its own held-out film segments.

Post hoc (opened 2026-10-14, not pre-registered); the design record is
mmmdata-agents ``docs/workbench/functional-space-followups/``. The scoring is
M2b's segment matrix (``scoring.m2b_segments``): the target side is the
subject's measured held-out window, the template side is its own encoder's
prediction of the same window, so the guess is the same Pearson argmax as the
between-subject routes.

Training sets (``--train``):

  partition  the subject's alignment films of one partition job (the encoder
             the stimulus route fitted; as M4 refits it), one job per
             (draw, target) at the given level
  pool       every alignment-pool film of the subject (one job per subject)

Test sets (never in any training set):

  heldout    the 12 unique films of the held-out sessions
  repeat     the repeated films' held-out showings (one pool per showing); the
             ``pool`` job also scores the within-subject data ceiling there:
             each showing against the subject's other held-out showing, and
             against the mean of its alignment-session showings (``dropped_repeat``),
             paired by nearest film time

Encoders: ``encoding.Encoder`` on ``encoding.film_design`` exactly as the
stimulus route and M4 fit them (film windows z-scored per window); predictions
are centred per window, as the route centred them. Regions: family A networks,
family B ROIs and family D sections (``families.family_labels``), named
``<family>:<roi>``.

Verbs:

  score    one job -> <derivatives>/functional_space/scores/within/<train>/
           [<scenario>/pct-<pct>/draw-<draw>/]target-<sub>/ (within_segments.parquet, within.json)

Usage:
    python within_subject.py score --train partition --pct 0 --draw 0 --target 03
    python within_subject.py score --train pool --target 03
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

import scoring as sc  # noqa: E402

SPACES = ("ebind", "psytwill")
REGION_FAMILIES = ("schaefer7n", "familyB", "familyD")


def out_dir(derivatives: Path, train: str, target: str, scenario: str = "primary", pct: int = 0,
            draw: int | None = None) -> Path:
    root = Path(derivatives) / "functional_space" / "scores" / "within" / train
    if train == "pool":
        return root / f"target-{target}"
    return root / scenario / f"pct-{pct:03d}" / f"draw-{draw:02d}" / f"target-{target}"


def region_columns(labels: dict[str, np.ndarray], families=REGION_FAMILIES) -> dict[str, np.ndarray]:
    """``{"<family>:<roi>": columns}`` over the scored families."""
    return {f"{fam}:{roi}": cols for fam in families for roi, cols in sc.network_columns(labels[fam]).items()}


def measured(rows: pd.DataFrame, cleaned: Path) -> dict[str, np.ndarray]:
    """Each showing's full window (own grid), z-scored per column, keyed by the row index."""
    import films as fm

    cache: dict = {}
    return {k: sc.zscore_columns(fm.film_series(r, cleaned, cache)).astype(np.float32) for k, r in rows.iterrows()}


def predicted(model, rows: pd.DataFrame, cache_root: Path, space: str, cleaned: Path) -> dict[str, np.ndarray]:
    """The encoder's prediction of each showing's window, centred per window, keyed by the row index.

    One showing per call, so two showings of one film never share a window group.
    """
    import encoding as enc
    import stimulus_route as sr

    out = {}
    for k in rows.index:
        x, _, g, _ = enc.film_design(rows.loc[[k]], cache_root, space, cleaned)
        out[k] = sr.center_blocks(model.predict(x), g).astype(np.float32)
    return out


def keyed(windows: pd.DataFrame, sub: str, role: str) -> pd.DataFrame:
    r = windows[(windows["sub"] == sub) & (windows["role"] == role)].copy()
    r.index = r["stimulus_id"] + "@" + r["showing"].astype(str)
    if r.index.duplicated().any():
        raise ValueError(f"sub-{sub}: duplicate {role} showing keys")
    return r


def by_film(d: dict[str, np.ndarray], keys: list[str]) -> dict[str, np.ndarray]:
    """Re-key ``stimulus_id@showing`` entries by stimulus id (one showing per film)."""
    out = {k.split("@")[0]: d[k] for k in keys}
    if len(out) != len(keys):
        raise ValueError(f"two showings of one film in one pool: {keys}")
    return out


def ceiling_pairs(rep: pd.DataFrame, other: pd.DataFrame, data: dict[str, np.ndarray],
                  dropped: dict[str, np.ndarray]) -> list[tuple[str, int, dict, dict]]:
    """Within-subject data ceiling on the repeat pool: (reference name, showing, target films, template films).

    For each held-out showing: against the other held-out showing of the same
    film, and against the mean of the film's alignment-session showings, each
    paired to the target showing by nearest film time.
    """
    import films as fm

    out = []
    showings = sorted(rep["showing"].unique())
    for a in showings:
        tgt, other_ho, other_mean = {}, {}, {}
        for sid, g in rep.groupby("stimulus_id"):
            ra = g[g["showing"] == a]
            rb = g[g["showing"] != a]
            if len(ra) != 1 or len(rb) != 1:
                raise ValueError(f"{sid}: expected two held-out showings, got {sorted(g['showing'])}")
            ka, kb = ra.index[0], rb.index[0]
            sl = fm.paired_slices({"a": ra.iloc[0], "b": rb.iloc[0]}, "a")
            od = other[other["stimulus_id"] == sid]
            rows = {"a": ra.iloc[0], **{k: r for k, r in od.iterrows()}}
            sm = fm.paired_slices(rows, "a")
            lo, hi = max(sl["a"].start, sm["a"].start), min(sl["a"].stop, sm["a"].stop)  # one target slice for both
            shift = lambda s, s_a: slice(s.start + lo - s_a.start, s.start + hi - s_a.start)  # noqa: E731
            tgt[sid] = data[ka][lo:hi]
            other_ho[sid] = data[kb][shift(sl["b"], sl["a"])]
            other_mean[sid] = np.mean([dropped[k][shift(sm[k], sm["a"])] for k in od.index], axis=0)
        out += [("other-heldout-showing", int(a), tgt, other_ho), ("mean-alignment-showings", int(a), tgt, other_mean)]
    return out


def run_score(args: argparse.Namespace, log=print) -> Path:
    import cha
    import encoding as enc
    import families as fam
    import films as fm
    import grayordinates as go
    import partitions as pt
    import stimulus_route as sr

    t0 = time.time()
    paths = enc.Paths()
    cpaths = cha.Paths()
    cleaned = cpaths.cleaned
    windows = fm.load_windows(paths.derivatives)
    s = args.target
    if args.train == "partition":
        if args.draw is None:
            raise ValueError("--train partition needs --draw")
        parts = pt.load_partitions(paths.derivatives)
        train = sr.job_films(parts, windows, args.scenario, args.pct, args.draw, s)[s]
    else:
        train = windows[(windows["sub"] == s) & (windows["role"] == "alignment")]
    if train.empty:
        raise ValueError(f"sub-{s}: no training films")
    held = keyed(windows, s, "heldout")
    rep = keyed(windows, s, "heldout_repeat")
    if set(train["stimulus_id"]) & (set(held["stimulus_id"]) | set(rep["stimulus_id"])):
        raise ValueError("a test film is in the training set")

    table = pd.read_csv(go.grayordinates_path(cleaned), sep="\t")
    regions = region_columns(fam.family_labels(table, cpaths.atlases))
    data = measured(pd.concat([held, rep]), cleaned)
    held_keys = sorted(held.index)
    showings = sorted(rep["showing"].unique())
    rep_keys = {a: sorted(rep.index[rep["showing"] == a]) for a in showings}

    frames, info, seconds = [], {}, {}
    for space in args.spaces:
        t1 = time.time()
        x, y, groups, bands = enc.film_design(train, paths.cache, space, cleaned)
        model = enc.Encoder(bands, backend=args.backend).fit(x, enc._zscore_film_windows(y, groups), groups)
        del x, y
        pred = predicted(model, pd.concat([held, rep]), paths.cache, space, cleaned)
        name = f"within-{args.train}-{space}"
        frames.append(sc.m2b_segments(by_film(data, held_keys), by_film(pred, held_keys), regions)
                      .assign(model=name, set="heldout"))
        for a, ks in rep_keys.items():
            frames.append(sc.m2b_segments(by_film(data, ks), by_film(pred, ks), regions)
                          .assign(model=name, set=f"repeat@{a}"))
        info[space] = {"encoder": model.diagnostics_, "training_films": int(train["stimulus_id"].nunique()),
                       "training_volumes": int(groups.size)}
        seconds[space] = round(time.time() - t1, 1)
        log(f"{space}: {train['stimulus_id'].nunique()} training films, encoder "
            f"{model.diagnostics_['cv_1_plus_score_q50_q95_q99']} ({seconds[space]:.0f} s)")
        del model, pred

    if args.train == "pool":  # the data ceiling does not depend on the encoder: score it once per subject
        other = keyed(windows, s, "dropped_repeat")
        dropped = measured(other, cleaned)
        for ref, a, tgt, tpl in ceiling_pairs(rep, other, data, dropped):
            frames.append(sc.m2b_segments(tgt, tpl, regions).assign(model=f"ceiling-{ref}", set=f"repeat@{a}"))

    dest = out_dir(paths.derivatives, args.train, s, args.scenario, args.pct, args.draw)
    dest.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(dest / "within_segments.parquet", index=False)
    seconds["total"] = round(time.time() - t0, 1)
    side = {
        "description": "post hoc within-subject arm (opened 2026-10-14): the subject's own encoder predicts its own "
                       "held-out film windows; segment identification as M2b (scoring.m2b_segments); the pool job "
                       "adds the within-subject data ceiling on the repeat pool",
        "job": {"train": args.train, "target": s,
                **({"scenario": args.scenario, "pct": args.pct, "draw": args.draw} if args.train == "partition" else {})},
        "spaces": info, "training_films": sorted(train["stimulus_id"].unique()),
        "test": {"heldout": held_keys, "repeat": {str(a): ks for a, ks in rep_keys.items()}},
        "regions": list(REGION_FAMILIES), "backend": args.backend, "segment_trs": sc.SEGMENT_TRS,
        "code_version": sc._code_version(), "seconds": seconds,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (dest / "within.json").write_text(json.dumps(side, indent=2) + "\n")
    log(f"wrote {dest} in {seconds['total']:.0f} s")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("score")
    s.add_argument("--train", required=True, choices=("partition", "pool"))
    s.add_argument("--target", required=True)
    s.add_argument("--scenario", default="primary", choices=("primary", "secondary"))
    s.add_argument("--pct", type=int, default=0)
    s.add_argument("--draw", type=int)
    s.add_argument("--spaces", nargs="+", default=list(SPACES), choices=SPACES)
    s.add_argument("--backend", default="torch_cuda")
    args = ap.parse_args()
    {"score": run_score}[args.verb](args)


if __name__ == "__main__":
    main()
