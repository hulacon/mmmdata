#!/usr/bin/env python3
"""Place the stimulus route's films and probe clips in a psytwill-space release.

The psytwill arm of the stimulus route (mmmdata-agents
``docs/workbench/functional-space/``, DECIDED 2026-10-07) encodes with the
release's V and A blocks and the V side of the VL relation, on the 0.5 s
frame grid. The films already have release families (``space release
project``) under ``<derivatives>/stimuli_features/psytwill/``. The probe clips
cannot go that way: the release's leak guard refuses a corpus that was a fit
input, and the A block was fitted on the probe corpora. So every corpus is
placed here with ``psytwill space project`` / ``space relate project`` on the
manifests the release pins, after ``space release verify``. The films are run
through the same commands so ``check`` can show the result equals their
release families.

Verbs:

  project  one corpus x one block -> <derivatives>/functional_space/
           psytwill_projection/<corpus>_<B>.parquet (+ .json sidecar); a V
           run also writes every relation side on V (<corpus>_VL.parquet)
  check    compare the films' tables here with their release families

Usage:
    python psytwill_projection.py project --corpus movie10 --block V
    python psytwill_projection.py check
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

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "src" / "python", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.config import load_config  # noqa: E402

RELEASE = "0.1.0"
WINDOW_S = 0.5
CORPORA = ("movies", "movie10", "friends")
BLOCK_TABLE = {"V": "frames", "A": "audio_frames"}  # block -> group table stem
FAMILY_MODEL = {"V": "pspace_v", "A": "pspace_a", "VL": "pspace_vl"}
FAMILY_TABLE = {"V": "frames", "A": "audio_frames", "VL": "frames"}
KEY = ("stimulus_id", "time")


class Paths:
    def __init__(self) -> None:
        cfg = load_config()["paths"]
        self.derivatives = Path(cfg["output_dir"])
        self.fit_corpora = Path(cfg["fit_corpora_dir"])
        self.psytwill = Path(cfg["stimfeat_env"]) / "bin" / "psytwill"
        self.release = self.fit_corpora / "space" / "releases" / f"psytwill-space_{RELEASE}.json"
        self.groups_mmm = self.derivatives / "stimuli_features" / "psytwill"
        self.out = self.derivatives / "functional_space" / "psytwill_projection"

    def features(self, corpus: str, block: str) -> Path:
        stem = BLOCK_TABLE[block]
        if corpus == "movies":
            return self.groups_mmm / f"movies_{stem}_features.parquet"
        return self.fit_corpora / corpus / "groups" / f"{corpus}_{stem}_features.parquet"

    def family(self, model: str) -> Path:
        return self.groups_mmm / f"movies_psytwill_space_{FAMILY_TABLE[model]}_features.parquet"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str]) -> str:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def cmd_project(args: argparse.Namespace) -> None:
    paths = Paths()
    run([str(paths.psytwill), "space", "release", "verify", str(paths.release)])
    rel = json.loads(paths.release.read_text())
    entry = rel["blocks"][args.block]
    manifest = Path(entry["manifest"])
    for key, path in (("manifest_sha256", manifest), ("weights_sha256", Path(entry["weights"]))):
        if sha256(path) != entry[key]:
            sys.exit(f"{path}: sha256 differs from release {RELEASE}'s {key}")
    features = paths.features(args.corpus, args.block)
    if not features.exists():
        sys.exit(f"{features} is missing")
    paths.out.mkdir(parents=True, exist_ok=True)
    out = paths.out / f"{args.corpus}_{args.block}.parquet"
    commands = [[str(paths.psytwill), "space", "project", "--features", str(features), "--key", ",".join(KEY),
                 "--window", str(WINDOW_S), "--space", str(manifest), "-o", str(out)]]
    run(commands[0])
    written = {args.block: out}
    for rname, rentry in rel["relations"].items():
        for side in ("a", "b"):
            if rentry.get(side) != args.block:
                continue
            rout = paths.out / f"{args.corpus}_{rname}.parquet"
            commands.append([str(paths.psytwill), "space", "relate", "project", "--relation", rentry["manifest"],
                             "--side", side, "--scores", str(out), "-o", str(rout)])
            run(commands[-1])
            written[rname] = rout
    side = {
        "release": {"version": RELEASE, "path": str(paths.release), "sha256": sha256(paths.release)},
        "corpus": args.corpus, "block": args.block, "window_s": WINDOW_S, "key": list(KEY),
        "features": {"path": str(features), "bytes": features.stat().st_size,
                     "mtime": _dt.datetime.fromtimestamp(features.stat().st_mtime, _dt.timezone.utc).isoformat()},
        "psytwill_version": run([str(paths.psytwill), "--version"]).strip(),
        "commands": commands,
        "why_not_release_project": "the release leak guard refuses a corpus that was a fit input; "
                                   "A_v0.3 was fitted on movie10 + friends (DECIDED 2026-10-07)",
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    for name, path in written.items():
        path.with_suffix(".json").write_text(json.dumps({**side, "table": name}, indent=2) + "\n")
    print(f"{args.corpus} {args.block}: wrote {', '.join(p.name for p in written.values())}")


def wide_family(path: Path, model: str):
    """A release family's rows for one model, wide on (stimulus_id, time)."""
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=[*KEY, "model", "feature", "value"],
                      filters=[("model", "=", model)]).to_pandas()
    return t.set_index([*KEY, "feature"])["value"].unstack("feature")


def cmd_check(args: argparse.Namespace) -> None:
    import pandas as pd

    paths = Paths()
    report, ok = {}, True
    for name, model in FAMILY_MODEL.items():
        here = pd.read_parquet(paths.out / f"movies_{name}.parquet")
        here["time"] = here["time"].astype(np.float64)  # `space project` writes time as a string
        here = here.set_index(list(KEY)).sort_index()
        fam = wide_family(paths.family(name), model).sort_index()
        # `space project` omits a row the block cannot place; the family keeps it as all-NaN
        absent = fam.index.difference(here.index)
        extra = here.index.difference(fam.index)
        rec = {"rows": [int(len(here)), int(len(fam))], "cols": [int(here.shape[1]), int(fam.shape[1])],
               "rows_only_in_family": int(len(absent)), "rows_only_here": int(len(extra)),
               "family_rows_absent_here_all_nan": bool(fam.loc[absent].isna().all(axis=None))}
        same_rows = (not len(extra)) and rec["family_rows_absent_here_all_nan"] and here.shape[1] == fam.shape[1]
        rec["same_rows"] = bool(same_rows)
        if same_rows:
            # columns are <name>_NNN here and <model>_NNN in the family: align on NNN
            here.columns = [c.rsplit("_", 1)[1] for c in here.columns]
            fam.columns = [c.rsplit("_", 1)[1] for c in fam.columns]
            here = here.reindex(index=fam.index, columns=fam.columns)
            a, b = here.to_numpy(np.float64), fam.to_numpy(np.float64)
            nan_a, nan_b = np.isnan(a), np.isnan(b)
            rec["same_nan_pattern"] = bool((nan_a == nan_b).all())
            both = ~(nan_a | nan_b)
            rec["max_abs_diff"] = float(np.abs(a[both] - b[both]).max())
            rec["max_abs_value"] = float(np.abs(b[both]).max())
            rec["pass"] = rec["same_nan_pattern"] and rec["max_abs_diff"] <= args.atol
        else:
            rec["pass"] = False
        ok &= rec["pass"]
        report[name] = rec
        print(name, json.dumps(rec))
    (paths.out / "check.json").write_text(json.dumps(
        {"atol": args.atol, "pass": bool(ok), "tables": report,
         "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}, indent=2) + "\n")
    if not ok:
        sys.exit("films here differ from their release families; see check.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    p = sub.add_parser("project", help="place one corpus in one release block (+ its relation sides)")
    p.add_argument("--corpus", choices=CORPORA, required=True)
    p.add_argument("--block", choices=list(BLOCK_TABLE), required=True)
    p.set_defaults(func=cmd_project)
    c = sub.add_parser("check", help="films here vs their release families")
    c.add_argument("--atol", type=float, default=1e-4)
    c.set_defaults(func=cmd_check)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
