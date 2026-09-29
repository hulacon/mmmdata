#!/usr/bin/env python3
"""Overlap partitions of the alignment-film pool for the functional-space routes.

Pre-registration §4 (overlap design) and §3.4 (tuning data). Every subject
aligns on ``N_FILMS`` films at every level; *s* of them are shared by all three
subjects and the rest are unique to each. The same draws serve every route, so
route comparisons are paired.

Scenarios:

  primary    one partition per draw serves all three targets: *s* shared films
             plus ``N_FILMS - s`` unique films per subject, all disjoint
  secondary  "template shared, target disjoint", 0% only: per target, the two
             template subjects share the same ``N_FILMS`` films and the target
             sees ``N_FILMS`` others

Tuning films for a (draw, target) job are the pool films that neither template
subject aligns on: the target's own unique films plus the undrawn films. The
template pair watched them all (every subject saw the whole pool), and no fit
in that job reads the template pair's responses to them.

Draw ``d`` at a level is seeded by ``(SEED, level, d)``, so raising the draw
count keeps every earlier draw unchanged. The count is fixed before the freeze
from the dry run (§4, §11.6).

Verbs:

  build   write <derivatives>/functional_space/partitions/{partitions,partition_summary}.tsv
  check   assert the invariants; exits non-zero on any failure

Usage:
    python partitions.py build [--draws-zero 50] [--draws-other 20]
    python partitions.py check
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "src" / "python", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import films  # noqa: E402
from core.config import load_config  # noqa: E402
from neuroimaging import data_quality as dq  # noqa: E402

SEED = 20260929
N_FILMS = 15
#: Overlap level (percent) -> shared films *s* (pre-registration §4).
LEVELS = {100: 15, 50: 8, 25: 4, 0: 0}
#: Working draw counts (§4); the final ones are fixed from the dry run before the freeze.
DEFAULT_DRAWS_ZERO = 50
DEFAULT_DRAWS_OTHER = 20
COLUMNS = ["scenario", "pct", "s", "draw", "target", "subject", "stimulus_id", "use"]
USES = ("align_shared", "align_unique", "tuning")


def out_dir(derivatives: Path) -> Path:
    return Path(derivatives) / "functional_space" / "partitions"


def load_partitions(derivatives: Path) -> pd.DataFrame:
    """The partition table. Missing is a loud error, never an empty table."""
    path = out_dir(derivatives) / "partitions.tsv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `partitions.py build`")
    return pd.read_csv(path, sep="\t", dtype={"target": str, "subject": str})


def expected_tuning(pct: int, pool: int, scenario: str = "primary") -> int:
    """Tuning-film count per job: the pool minus the template pair's alignment films."""
    s = LEVELS[pct]
    template_films = N_FILMS if scenario == "secondary" else s + 2 * (N_FILMS - s)
    return pool - template_films


def rng_for(pct: int, draw: int, scenario: str) -> np.random.Generator:
    return np.random.default_rng([SEED, pct, draw, 0 if scenario == "primary" else 1])


def primary_draw(pool: list[str], subjects: tuple[str, ...], pct: int, draw: int) -> dict[str, dict[str, list[str]]]:
    """``{subject: {"shared": [...], "unique": [...]}}`` for one primary draw."""
    s = LEVELS[pct]
    need = s + len(subjects) * (N_FILMS - s)
    if need > len(pool):
        raise ValueError(f"{pct}% needs {need} films; the pool has {len(pool)}")
    order = rng_for(pct, draw, "primary").permutation(len(pool))
    picked = [pool[i] for i in order[:need]]
    shared = sorted(picked[:s])
    out = {}
    for k, sub in enumerate(subjects):
        lo = s + k * (N_FILMS - s)
        out[sub] = {"shared": shared, "unique": sorted(picked[lo: lo + N_FILMS - s])}
    return out


def secondary_draw(pool: list[str], subjects: tuple[str, ...], target: str, draw: int
                   ) -> dict[str, dict[str, list[str]]]:
    """Template pair share ``N_FILMS`` films; the target gets ``N_FILMS`` disjoint ones."""
    if 2 * N_FILMS > len(pool):
        raise ValueError(f"the secondary scenario needs {2 * N_FILMS} films; the pool has {len(pool)}")
    order = np.random.default_rng([SEED, 0, draw, 1, subjects.index(target)]).permutation(len(pool))
    template = sorted(pool[i] for i in order[:N_FILMS])
    own = sorted(pool[i] for i in order[N_FILMS: 2 * N_FILMS])
    return {sub: ({"shared": [], "unique": own} if sub == target else {"shared": template, "unique": []})
            for sub in subjects}


def job_rows(scenario: str, pct: int, draw: int, target: str, assignment: dict, pool: list[str]) -> list[dict]:
    base = {"scenario": scenario, "pct": pct, "s": LEVELS[pct], "draw": draw, "target": target}
    rows = []
    for sub, films_ in assignment.items():
        for use, key in (("align_shared", "shared"), ("align_unique", "unique")):
            rows += [{**base, "subject": sub, "stimulus_id": f, "use": use} for f in films_[key]]
    template = [s for s in assignment if s != target]
    used = set().union(*(set(assignment[s]["shared"]) | set(assignment[s]["unique"]) for s in template))
    tuning = sorted(set(pool) - used)
    for sub in template:
        rows += [{**base, "subject": sub, "stimulus_id": f, "use": "tuning"} for f in tuning]
    return rows


def build_partitions(pool: list[str], subjects: tuple[str, ...], draws_zero: int, draws_other: int) -> pd.DataFrame:
    pool = sorted(pool)
    rows = []
    for pct in LEVELS:
        n = draws_zero if pct == 0 else draws_other
        for d in range(n):
            a = primary_draw(pool, subjects, pct, d)
            for t in subjects:
                rows += job_rows("primary", pct, d, t, a, pool)
    for d in range(draws_zero):
        for t in subjects:
            rows += job_rows("secondary", 0, d, t, secondary_draw(pool, subjects, t, d), pool)
    return pd.DataFrame(rows, columns=COLUMNS)


def check_partitions(df: pd.DataFrame, pool: list[str], subjects: tuple[str, ...]) -> list[str]:
    """Invariants of the partition table; returns failure messages (empty = pass)."""
    fails = []
    pool_set = set(pool)
    if not set(df["stimulus_id"]) <= pool_set:
        fails.append("a partition uses a film outside the alignment pool")
    if not set(df["use"]) <= set(USES):
        fails.append(f"unknown uses {sorted(set(df['use']) - set(USES))}")
    for (scen, pct, draw, target), g in df.groupby(["scenario", "pct", "draw", "target"]):
        tag = f"{scen} {pct}% draw {draw} target {target}"
        align = g[g["use"] != "tuning"]
        sets = {s: set(align.loc[align["subject"] == s, "stimulus_id"]) for s in subjects}
        if any(len(v) != N_FILMS for v in sets.values()):
            fails.append(f"{tag}: alignment counts {[len(v) for v in sets.values()]}, expected {N_FILMS} each")
        if align.duplicated(["subject", "stimulus_id"]).any():
            fails.append(f"{tag}: a subject has a film twice")
        template = [s for s in subjects if s != target]
        if scen == "primary":
            common = set.intersection(*sets.values())
            if len(common) != LEVELS[pct]:
                fails.append(f"{tag}: {len(common)} films shared by all, expected {LEVELS[pct]}")
            for a in subjects:
                for b in subjects:
                    if a < b and (sets[a] & sets[b]) != common:
                        fails.append(f"{tag}: sub-{a} and sub-{b} share films beyond the common set")
        else:
            if sets[template[0]] != sets[template[1]]:
                fails.append(f"{tag}: the template pair do not share all their films")
            if sets[target] & sets[template[0]]:
                fails.append(f"{tag}: the target shares films with the template")
        tune = g[g["use"] == "tuning"]
        if set(tune["subject"]) != set(template):
            fails.append(f"{tag}: tuning rows are not the template pair's")
        tset = set(tune["stimulus_id"])
        if tset & (sets[template[0]] | sets[template[1]]):
            fails.append(f"{tag}: a tuning film is one the template pair aligns on")
        want = expected_tuning(pct, len(pool), scen)
        for s in template:
            n = int((tune["subject"] == s).sum())
            if n != want:
                fails.append(f"{tag}: sub-{s} has {n} tuning films, expected {want}")
    # draws within a level are distinct partitions
    for (scen, pct, target), g in df[df["use"] != "tuning"].groupby(["scenario", "pct", "target"]):
        keys = g.groupby("draw").apply(lambda x: tuple(sorted(zip(x["subject"], x["stimulus_id"]))),
                                       include_groups=False)
        if keys.duplicated().any():
            fails.append(f"{scen} {pct}% target {target}: duplicate draws")
    return fails


def pool_and_subjects(derivatives: Path) -> tuple[list[str], tuple[str, ...], pd.DataFrame]:
    windows = films.load_windows(derivatives)
    align = windows[windows["role"] == "alignment"]
    per_sub = align.groupby("sub")["stimulus_id"].apply(frozenset)
    if per_sub.nunique() != 1:
        raise ValueError("the alignment pool differs between subjects; rerun `films.py check`")
    return sorted(per_sub.iloc[0]), tuple(sorted(per_sub.index)), windows


def summarise(df: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Per job and subject: films and minutes of alignment data (the data-efficiency axis)."""
    w = windows[windows["role"] == "alignment"].set_index(["sub", "stimulus_id"])
    minutes = (w["n"] * w["repetition_time"] / 60.0)
    align = df[df["use"] != "tuning"].copy()
    align["minutes"] = [minutes.loc[(s, f)] for s, f in zip(align["subject"], align["stimulus_id"])]
    return (align.groupby(["scenario", "pct", "s", "draw", "target", "subject"], as_index=False)
            .agg(n_films=("stimulus_id", "size"), minutes=("minutes", "sum")))


def cmd_build(args: argparse.Namespace) -> None:
    derivatives = Path(load_config()["paths"]["output_dir"])
    pool, subjects, windows = pool_and_subjects(derivatives)
    df = build_partitions(pool, subjects, args.draws_zero, args.draws_other)
    fails = check_partitions(df, pool, subjects)
    if fails:
        sys.exit("partitions failed their checks; nothing written:\n  " + "\n  ".join(fails[:20]))
    dest = out_dir(derivatives)
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / "partitions.tsv", sep="\t", index=False)
    summary = summarise(df, windows)
    summary.to_csv(dest / "partition_summary.tsv", sep="\t", index=False)
    side = {
        "description": "Overlap partitions of the functional-space alignment-film pool, one row per "
                       "(scenario, level, draw, target, subject, film, use).",
        "seed": SEED, "n_films": N_FILMS, "levels": LEVELS,
        "draws": {"zero": args.draws_zero, "other": args.draws_other},
        "draw_counts_final": False,
        "pool_size": len(pool), "subjects": list(subjects),
        "film_windows": str(films.windows_path(derivatives)),
        "code_version": dq.code_version(REPO_ROOT),
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (dest / "partitions.json").write_text(json.dumps(side, indent=2) + "\n")
    jobs = df.drop_duplicates(["scenario", "pct", "draw", "target"]).groupby(["scenario", "pct"]).size()
    print(f"{dest}/partitions.tsv: {len(df)} rows")
    print("jobs (draw x target):\n" + jobs.to_string())
    print("alignment minutes per subject:\n"
          + summary.groupby(["scenario", "pct"])["minutes"].describe()[["mean", "min", "max"]].round(1).to_string())


def cmd_check(args: argparse.Namespace) -> None:
    derivatives = Path(load_config()["paths"]["output_dir"])
    pool, subjects, _ = pool_and_subjects(derivatives)
    df = load_partitions(derivatives)
    fails = check_partitions(df, pool, subjects)
    if fails:
        sys.exit("FAIL\n  " + "\n  ".join(fails[:20]))
    print(f"ok: {df.drop_duplicates(['scenario', 'pct', 'draw', 'target']).shape[0]} jobs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    b = sub.add_parser("build")
    b.add_argument("--draws-zero", type=int, default=DEFAULT_DRAWS_ZERO)
    b.add_argument("--draws-other", type=int, default=DEFAULT_DRAWS_OTHER)
    sub.add_parser("check")
    args = ap.parse_args()
    {"build": cmd_build, "check": cmd_check}[args.verb](args)


if __name__ == "__main__":
    main()
