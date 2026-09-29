#!/usr/bin/env python3
"""Film catalogue and per-showing volume windows for the functional-space routes.

Every NATencoding film showing of the target subjects gets a role and a window
of cleaned-series volumes. The routes read films only through this table, so
the trimming rule lives in one place. The design record is mmmdata-agents
``docs/workbench/functional-space/`` (pre-registration §3.1, §6).

Roles:

  alignment       a unique film in a non-held-out session (the alignment pool)
  heldout         a unique film in a held-out session (test data, §3.2)
  heldout_repeat  a repeated film's showing in a held-out session (scored apart)
  dropped_repeat  a repeated film's showing in an alignment session; the
                  repeated films are shared by construction, so no route uses them

Window (DECIDED 2026-09-29): the volumes lying wholly inside
``[onset + SHIFT_S + BUFFER_S, onset + play + SHIFT_S)``, where ``play`` is
the shorter of the logged duration and the video file's length. Onset and
duration come from the events file; nothing is fitted to the BOLD. The logged
duration is the task's routine length, not playback, and one file is much
shorter than its showing, so the file length caps it. The volume convention
(volume ``i`` spans ``[i*TR, (i+1)*TR)``) is data-quality tier 2's
``window()``, reused here.

Verbs:

  build   write <derivatives>/functional_space/films/film_windows.tsv (+ .json)
  check   assert the table's invariants; exits non-zero on any failure

Usage:
    python films.py build
    python films.py check
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "src" / "python", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import grayordinates as go  # noqa: E402
from core.config import load_config  # noqa: E402
from neuroimaging import data_quality as dq  # noqa: E402
from neuroimaging.data_quality_tier2 import movie_name_index, window  # noqa: E402

TASK = "NATencoding"
SUBJECTS = ("03", "04", "05")
#: Held-out film sessions, chosen by the §3.3 rule before any score (DECIDED 2026-09-28).
HELDOUT_SESSIONS = ("23", "26")
#: Hemodynamic shift of the window, in seconds (3 TRs at 1.5 s; the pilot's shift).
SHIFT_S = 4.5
#: Dropped from the start of each shifted window: the previous film's response is still decaying.
BUFFER_S = 6.0
REGIME = "reference"
ROLES = ("alignment", "heldout", "heldout_repeat", "dropped_repeat")
#: Pre-registration §3.1-3.2: pool size and held-out unique films, per subject.
N_POOL = 46
N_HELDOUT = 12
WINDOW_COLUMNS = [
    "sub", "ses", "run", "stimulus_id", "movie_name", "role", "showing",
    "onset", "duration", "file_s", "play_s", "repetition_time", "n_vol", "n_nss",
    "start", "n", "series",
]


class Paths:
    def __init__(self) -> None:
        cfg = load_config()["paths"]
        self.bids_root = Path(cfg["bids_project_dir"])
        self.derivatives = Path(cfg["output_dir"])
        self.cleaned = go.tree_root(self.derivatives)
        self.registry = self.bids_root / "stimuli" / "stimulus_registry" / "movies.tsv"
        self.movies = self.bids_root / "stimuli" / "movies"
        self.ffprobe = Path(cfg["stimfeat_env"]) / "bin" / "ffprobe"
        self.out = self.derivatives / "functional_space" / "films"


def windows_path(derivatives: Path) -> Path:
    return Path(derivatives) / "functional_space" / "films" / "film_windows.tsv"


def load_windows(derivatives: Path) -> pd.DataFrame:
    """The film-window table. Missing is a loud error, never an empty table."""
    path = windows_path(derivatives)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `films.py build`")
    return pd.read_csv(path, sep="\t", dtype={"sub": str, "ses": str, "run": str})


def file_seconds(ffprobe: Path, path: Path) -> float:
    """Container duration of a video file, from ffprobe."""
    if not ffprobe.exists():
        raise FileNotFoundError(f"ffprobe not found at {ffprobe}; check `stimfeat_env` in the mmmdata config")
    out = subprocess.run(
        [str(ffprobe), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return float(out)


def assign_roles(df: pd.DataFrame, heldout: tuple[str, ...] = HELDOUT_SESSIONS) -> pd.Series:
    """Role of each showing; a film shown in more than one session is a repeat."""
    n_sessions = df.groupby(["sub", "stimulus_id"])["ses"].transform("nunique")
    repeated = n_sessions > 1
    in_heldout = df["ses"].isin(heldout)
    role = np.select(
        [~repeated & ~in_heldout, ~repeated & in_heldout, repeated & in_heldout],
        ["alignment", "heldout", "heldout_repeat"],
        default="dropped_repeat",
    )
    return pd.Series(role, index=df.index)


def film_window(onset: float, play: float, tr: float, n_vol: int,
                shift: float = SHIFT_S, buffer: float = BUFFER_S) -> tuple[int, int]:
    """``(start, n)`` of the shifted, buffered window; ``n`` is 0 if the buffer eats the film."""
    return window(onset + shift + buffer, max(play - buffer, 0.0), tr, n_vol)


def build_table(paths: Paths) -> pd.DataFrame:
    manifest = go.load_manifest(paths.cleaned)
    runs = manifest[(manifest["task"] == TASK) & (manifest["regime"] == REGIME)
                    & manifest["sub"].isin(SUBJECTS)]
    if runs.empty:
        raise ValueError(f"{paths.cleaned}/manifest.tsv has no {TASK} runs under regime {REGIME!r}")
    names = movie_name_index(paths.registry)
    registry = pd.read_csv(paths.registry, sep="\t", dtype=str, keep_default_na=False).set_index("stimulus_id")
    lengths: dict[str, float] = {}
    rows = []
    for r in runs.sort_values(["sub", "ses", "run"]).itertuples(index=False):
        events = (paths.bids_root / f"sub-{r.sub}" / f"ses-{r.ses}" / "func"
                  / f"sub-{r.sub}_ses-{r.ses}_task-{TASK}_run-{r.run}_events.tsv")
        if not events.exists():
            raise FileNotFoundError(f"No events file for a cleaned run: {events}")
        ev = pd.read_csv(events, sep="\t")
        tr, n_vol = float(r.RepetitionTime), int(r.n_vol)
        for m in ev[ev["trial_type"] == "movie"].itertuples(index=False):
            key = str(m.movie_name).strip().casefold()
            if key not in names:
                raise KeyError(f"{events}: film {m.movie_name!r} is not in the stimulus registry")
            sid = names[key]
            if sid not in lengths:
                lengths[sid] = file_seconds(paths.ffprobe, paths.movies / registry.loc[sid, "video_file"])
            play = min(float(m.duration), lengths[sid])
            start, n = film_window(float(m.onset), play, tr, n_vol)
            rows.append({
                "sub": r.sub, "ses": r.ses, "run": r.run, "stimulus_id": sid,
                "movie_name": str(m.movie_name).strip(), "onset": float(m.onset),
                "duration": float(m.duration), "file_s": lengths[sid], "play_s": play,
                "repetition_time": tr, "n_vol": n_vol, "n_nss": int(r.n_nss),
                "start": start, "n": n, "series": r.path,
            })
    df = pd.DataFrame(rows).sort_values(["sub", "ses", "run", "onset"], ignore_index=True)
    df["showing"] = df.groupby(["sub", "stimulus_id"]).cumcount() + 1
    df["role"] = assign_roles(df)
    return df[WINDOW_COLUMNS]


def check_table(df: pd.DataFrame) -> list[str]:
    """Invariants of the window table; returns failure messages (empty = pass)."""
    fails = []
    bad_roles = set(df["role"]) - set(ROLES)
    if bad_roles:
        fails.append(f"unknown roles {sorted(bad_roles)}")
    for sub, g in df.groupby("sub"):
        n_pool = g.loc[g["role"] == "alignment", "stimulus_id"].nunique()
        n_held = g.loc[g["role"] == "heldout", "stimulus_id"].nunique()
        if n_pool != N_POOL:
            fails.append(f"sub-{sub}: {n_pool} alignment films, expected {N_POOL}")
        if n_held != N_HELDOUT:
            fails.append(f"sub-{sub}: {n_held} held-out films, expected {N_HELDOUT}")
        uniq = g[g["role"].isin(["alignment", "heldout"])]
        if uniq.duplicated("stimulus_id").any():
            fails.append(f"sub-{sub}: a unique film has more than one showing")
        if set(g.loc[g["role"] == "alignment", "stimulus_id"]) & set(g.loc[g["role"] == "heldout", "stimulus_id"]):
            fails.append(f"sub-{sub}: a film is both alignment and held-out")
    # the same film -> role assignment in every subject
    roles = df[df["role"].isin(["alignment", "heldout"])].pivot_table(
        index="stimulus_id", columns="sub", values="role", aggfunc="first")
    if roles.isna().any().any() or (roles.nunique(axis=1) > 1).any():
        fails.append("film roles differ between subjects")
    # windows lie inside the run, after the non-steady-state volumes, and are non-empty
    if (df["n"] <= 0).any():
        fails.append(f"{int((df['n'] <= 0).sum())} empty windows")
    if (df["start"] < df["n_nss"]).any():
        fails.append("a window starts inside the non-steady-state volumes")
    end = df["start"] + df["n"]
    if (end > df["n_vol"]).any():
        fails.append("a window runs past the end of its run")
    # clipped by the run end: the shifted window should fit wholly
    want_end = np.floor((df["onset"] + df["play_s"] + SHIFT_S) / df["repetition_time"] + 1e-6)
    clipped = want_end > df["n_vol"]
    if clipped.any():
        fails.append(f"{int(clipped.sum())} windows clipped by the run end")
    # windows of one run do not overlap
    for _, g in df.sort_values("onset").groupby(["sub", "ses", "run"]):
        s, e = g["start"].to_numpy(), (g["start"] + g["n"]).to_numpy()
        if (s[1:] < e[:-1]).any():
            fails.append(f"overlapping windows in sub-{g['sub'].iat[0]} ses-{g['ses'].iat[0]} run-{g['run'].iat[0]}")
    # volume count agrees with the played length: whole volumes only, so up to
    # one partial volume is lost at each end, never gained
    diff = df["n"] - (df["play_s"] - BUFFER_S) / df["repetition_time"]
    if ((diff > 1e-6) | (diff <= -2.0)).any():
        fails.append("a window's volume count disagrees with its played length")
    return fails


def cmd_build(args: argparse.Namespace) -> None:
    paths = Paths()
    df = build_table(paths)
    fails = check_table(df)
    if fails:
        sys.exit("film_windows failed its checks; nothing written:\n  " + "\n  ".join(fails))
    paths.out.mkdir(parents=True, exist_ok=True)
    tsv = paths.out / "film_windows.tsv"
    df.to_csv(tsv, sep="\t", index=False)
    capped = df.loc[df["play_s"] < df["duration"] - 1.0, ["stimulus_id", "duration", "file_s"]].drop_duplicates("stimulus_id")
    side = {
        "description": "Per-showing volume windows of NATencoding films for the functional-space routes.",
        "window": "volumes wholly inside [onset + shift_s + buffer_s, onset + play_s + shift_s); "
                  "volume i spans [i*TR, (i+1)*TR); play_s = min(duration, file_s)",
        "shift_s": SHIFT_S, "buffer_s": BUFFER_S, "heldout_sessions": list(HELDOUT_SESSIONS),
        "regime": REGIME, "roles": list(ROLES),
        "series_root": str(paths.cleaned),
        "capped_by_file_length": capped.to_dict(orient="records"),
        "code_version": dq.code_version(REPO_ROOT),
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    tsv.with_suffix(".json").write_text(json.dumps(side, indent=2) + "\n")
    print(f"{tsv}: {len(df)} showings")
    print(df.groupby(["sub", "role"]).agg(showings=("stimulus_id", "size"), films=("stimulus_id", "nunique"),
                                          volumes=("n", "sum")).to_string())
    if len(capped):
        print("capped by file length:\n" + capped.to_string(index=False))


def cmd_check(args: argparse.Namespace) -> None:
    df = load_windows(Paths().derivatives)
    fails = check_table(df)
    if fails:
        sys.exit("FAIL\n  " + "\n  ".join(fails))
    print(f"ok: {len(df)} showings, {df['sub'].nunique()} subjects")


def film_series(row, cleaned_root: Path, cache: dict | None = None) -> np.ndarray:
    """One showing's window of the cleaned series, ``(n, n_grayordinates)``.

    ``cache`` maps a series path to its loaded run, so several films of one
    run read the file once.
    """
    path = Path(cleaned_root) / row.series
    if cache is not None and path in cache:
        run = cache[path]
    else:
        run = go.load_run(path, cleaned_root)
        if cache is not None:
            cache[path] = run
    x = run.data[int(row.start): int(row.start) + int(row.n)]
    if np.isnan(x).all(axis=1).any():
        raise ValueError(f"{path.name}: window {row.start}+{row.n} contains a non-steady-state row")
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    sub.add_parser("build")
    sub.add_parser("check")
    args = ap.parse_args()
    {"build": cmd_build, "check": cmd_check}[args.verb](args)


if __name__ == "__main__":
    main()
