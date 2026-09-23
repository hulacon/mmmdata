#!/usr/bin/env python
"""Compose every trial-based (TB) run onto the movie feature grid, once,
resumably -- the campaign half of contracts §4.3 item 5.

One `psytwill compose` call per BOLD run turns the run's events.tsv plus the
item feature stores into movie-schema tables (frames / audio_frames /
transcript_words on the 0.5 s grid) and, with media, a browsable render
(frames/, audio.m4a, transcript CSVs). Output is a per-subject BIDS tree
whose run root is the BIDS stem, inner layout unchanged from the verb:

    <derivatives>/stimuli_features/<dataset>/
      dataset_description.json
      sub-##/ses-##/sub-##_ses-##_task-TB<x>_run-##/
        features/movies_*_features.parquet (+ .meta.json)
        movies/<stem>/{frames/,audio.m4a,transcribe_*.csv}

Grid length rule (DECIDED 2026-09-22): the composed grid spans the BOLD run,
`n_volumes x RepetitionTime`, read from the run's BOLD header. The verb takes
that as `--lead-out = scan_end - last_event_offset`, computed here per run.
The task program holds fixation after its last trial until the scanner stops,
which is what the surplus is.

Usage
-----
    tb_compose_campaign.py plan                    # every run, lead-out, state
    tb_compose_campaign.py plan --json
    tb_compose_campaign.py run --dry-run           # commands only
    tb_compose_campaign.py run --subject 03        # one subject (array cell)
    tb_compose_campaign.py run --no-media          # tables only
    tb_compose_campaign.py verify                  # tree vs. the run list

Comparability (tb-timelines Settles-when 3): every compose reads the
per-model labels from `<dataset>/comparability.tsv` (a generated table beside
the tree, not code; `--comparability` overrides) into each sidecar's
`models.<m>.comparable`. A missing table is an error naming the path.

Idempotent: every run is handed to the verb, which rewrites tables only when
its input signature changes (events, stores, registry, lead-out, bin-coverage
rule, comparability table by content); media is skipped here when the frames
directory is already populated. `--force` redoes both. Any run whose media
has unmapped items, or whose compose exits non-zero, fails the campaign
loudly rather than being skipped.

Env: /gpfs/projects/hulacon/shared/envs/stimfeat (psytwill >= 0.19.0,
Pillow, soundfile) with ffmpeg on PATH for media.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))
try:
    from core.config import load_config
    _config = load_config(config_dir=_REPO_ROOT / "config")
    BIDS_ROOT = Path(_config["paths"]["bids_project_dir"])
    DERIV_ROOT = Path(_config["paths"]["output_dir"])
    STIMFEAT_ENV = Path(_config["paths"].get(
        "stimfeat_env", "/gpfs/projects/hulacon/shared/envs/stimfeat"))
except Exception:
    BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")
    DERIV_ROOT = BIDS_ROOT / "derivatives"
    STIMFEAT_ENV = Path("/gpfs/projects/hulacon/shared/envs/stimfeat")

STIM_DIR = BIDS_ROOT / "stimuli"
REGISTRY_DIR = STIM_DIR / "stimulus_registry"
FEATURES_ROOT = DERIV_ROOT / "stimuli_features"
STORE_DIR = FEATURES_ROOT / "psytwill"
DATASET_NAME = "tb"
OUT_ROOT = FEATURES_ROOT / DATASET_NAME

TASKS = ("TBencoding", "TBretrieval")
# The item stores the compose route was validated on (tb-timelines log
# 2026-09-15/18): one per modality. Other shared1000_*/twp1000_* stores
# carry the same text models and would collide on model name in the verb.
DEFAULT_STORES = (
    "shared1000_image_features.parquet",
    "twp1000_word_audio_frames_features.parquet",
    "twp1000_word_words_features.parquet",
)
PSYTWILL = STIMFEAT_ENV / "bin" / "psytwill"

_EVENTS_RE = re.compile(
    r"^sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_task-(?P<task>[^_]+)_run-(?P<run>[^_]+)_events\.tsv$"
)


@dataclass
class Run:
    subject: str
    session: str
    task: str
    run: str
    events: Path
    bold: Path

    @property
    def stem(self) -> str:
        return self.events.name.removesuffix("_events.tsv")

    @property
    def out_dir(self) -> Path:
        return OUT_ROOT / f"sub-{self.subject}" / f"ses-{self.session}" / self.stem

    @property
    def frames_dir(self) -> Path:
        return self.out_dir / "movies" / self.stem / "frames"

    def scan_end(self) -> float:
        """n_volumes x TR from the BOLD header (seconds)."""
        import nibabel as nib
        img = nib.load(str(self.bold))
        n_vol = int(img.shape[-1])
        tr = float(img.header.get_zooms()[-1])
        return round(n_vol * tr, 3)

    def last_offset(self) -> float:
        import pandas as pd
        ev = pd.read_csv(self.events, sep="\t", usecols=["onset", "duration"])
        return round(float((ev["onset"] + ev["duration"]).max()), 3)

    def lead_out(self) -> float:
        lo = round(self.scan_end() - self.last_offset(), 3)
        if lo < 0:
            raise RuntimeError(
                f"{self.stem}: events end {self.last_offset()} s after the "
                f"scan end {self.scan_end()} s; refusing to truncate"
            )
        return lo

    def tables_done(self) -> bool:
        feat = self.out_dir / "features"
        return feat.is_dir() and any(feat.glob("movies_*_features.meta.json"))

    def media_done(self) -> bool:
        return self.frames_dir.is_dir() and any(self.frames_dir.iterdir())


def discover_runs(subject: str | None = None) -> list[Run]:
    """Every TB encoding/retrieval events.tsv under the BIDS root that has a
    BOLD file beside it. A run with events but no BOLD is reported, not
    composed: the grid has nothing to span."""
    runs: list[Run] = []
    orphans: list[Path] = []
    for ev in sorted(BIDS_ROOT.glob("sub-*/ses-*/func/*_task-TB*_events.tsv")):
        m = _EVENTS_RE.match(ev.name)
        if not m or m["task"] not in TASKS:
            continue
        if subject and m["sub"] != subject:
            continue
        bold = ev.with_name(ev.name.replace("_events.tsv", "_bold.nii.gz"))
        if not bold.exists():
            orphans.append(ev)
            continue
        runs.append(Run(m["sub"], m["ses"], m["task"], m["run"], ev, bold))
    for o in orphans:
        print(f"WARNING: no BOLD beside {o.relative_to(BIDS_ROOT)}; skipped",
              file=sys.stderr)
    if not runs:
        raise SystemExit(
            f"no TB runs found under {BIDS_ROOT} (subject={subject!r}); "
            "check config/local.toml [paths] bids_project_dir"
        )
    return runs


def resolve_stores(names: list[str]) -> list[Path]:
    paths = []
    for n in names:
        p = Path(n) if os.sep in n else STORE_DIR / n
        if not p.exists():
            raise SystemExit(f"item store not found: {p}")
        paths.append(p)
    return paths


def psytwill_version() -> str:
    out = subprocess.run([str(PSYTWILL), "--version"], capture_output=True,
                         text=True, check=True).stdout.strip()
    return out.split()[-1]


def write_dataset_description(stores: list[Path]) -> Path:
    """§6.4: the tree carries its own dataset_description.json. Written once,
    left alone afterwards (a version bump is a deliberate edit)."""
    p = OUT_ROOT / "dataset_description.json"
    if p.exists():
        return p
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    desc = {
        "Name": "MMMData trial-based runs composed onto the movie feature grid",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "psytwill",
                "Version": psytwill_version(),
                "CodeURL": "https://github.com/hulacon/psytwill",
                "Description": "`psytwill compose`: events + item feature stores -> "
                               "movie-schema tables on the 0.5 s grid, plus a "
                               "browsable media render of each run",
            },
            {
                "Name": "mmmdata/scripts/tb_compose_campaign.py",
                "Description": "Campaign driver: one compose per TB BOLD run; grid "
                               "spans n_volumes x RepetitionTime of the run's BOLD "
                               "(lead-out = scan end - last event offset)",
            },
        ],
        "SourceDatasets": [
            {"URL": "../../..", "Description": "events.tsv and BOLD headers of the raw BIDS dataset"},
            {"URL": "../psytwill", "Description": "item feature stores: " + ", ".join(s.name for s in stores)},
            {"URL": "../../../stimuli", "Description": "stimulus registry and media (rendered frames/audio embed these)"},
        ],
        "HowToAcknowledge": "Composed tables follow the movie-table schema of "
                            "derivatives/stimuli_features/movies; each run root is "
                            "the BIDS stem of the BOLD run it spans.",
    }
    p.write_text(json.dumps(desc, indent=2) + "\n")
    return p


def comparability_path(args: argparse.Namespace) -> Path:
    p = args.comparability or (OUT_ROOT / "comparability.tsv")
    if not p.exists():
        raise SystemExit(
            f"comparability table not found: {p}. It is written by the "
            "tb-timelines render falsifier (mmmdata-agents workbench); pass "
            "--comparability PATH to use another.")
    return p


def compose_cmd(run: Run, stores: list[Path], lead_out: float, *,
                media: bool, force: bool, sparse: bool,
                comparability: Path) -> list[str]:
    cmd = [str(PSYTWILL), "compose", str(run.events),
           "--stores", *map(str, stores),
           "--registry", str(REGISTRY_DIR),
           "--lead-out", f"{lead_out:g}",
           "--comparability", str(comparability),
           "-o", str(run.out_dir), "--json"]
    if media:
        cmd += ["--media", "--stimuli-root", str(STIM_DIR)]
    if force:
        cmd.append("--force")
    if sparse:
        cmd.append("--sparse")
    return cmd


# ---------------------------------------------------------------------------
# verbs
# ---------------------------------------------------------------------------

def cmd_plan(args: argparse.Namespace) -> int:
    runs = discover_runs(args.subject)
    rows = []
    for r in runs:
        rows.append({
            "stem": r.stem, "subject": r.subject, "session": r.session,
            "task": r.task, "run": r.run,
            "scan_end": r.scan_end(), "last_offset": r.last_offset(),
            "lead_out": r.lead_out(),
            "tables": r.tables_done(), "media": r.media_done(),
            "out": str(r.out_dir),
        })
    if args.json:
        print(json.dumps({"dataset": str(OUT_ROOT), "n_runs": len(rows),
                          "runs": rows}, indent=2))
        return 0
    by_task = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    print(f"dataset: {OUT_ROOT}")
    print(f"{len(rows)} runs; tables done {sum(r['tables'] for r in rows)}, "
          f"media done {sum(r['media'] for r in rows)}")
    for task, trs in sorted(by_task.items()):
        ends = sorted({t["scan_end"] for t in trs})
        los = sorted({t["lead_out"] for t in trs})
        print(f"  {task}: {len(trs)} runs; scan_end {ends}; lead_out {los}")
    if args.todo:
        for row in rows:
            if not (row["tables"] and row["media"]):
                print(f"  todo {row['stem']} lead_out={row['lead_out']}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    runs = discover_runs(args.subject)
    stores = resolve_stores(args.stores)
    comparability = comparability_path(args)
    media = not args.no_media
    if media and shutil.which("ffmpeg") is None:
        raise SystemExit("--media needs ffmpeg on PATH (module load ffmpeg); "
                         "or pass --no-media")
    if not args.dry_run:
        print(f"dataset_description: {write_dataset_description(stores)}")
    failures = []
    t0 = time.time()
    for i, r in enumerate(runs, 1):
        lead_out = r.lead_out()
        want_media = media and (args.force or not r.media_done())
        # tables: always ask the verb — its input signature knows whether
        # they are stale (a relabel or a rule change leaves them in place)
        cmd = compose_cmd(r, stores, lead_out, media=want_media,
                          force=args.force, sparse=args.sparse,
                          comparability=comparability)
        if args.dry_run:
            print(" ".join(cmd))
            continue
        t1 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            failures.append(r.stem)
            print(f"[{i}/{len(runs)}] {r.stem}: FAILED exit {proc.returncode}\n"
                  f"{proc.stderr.strip()[-2000:]}", flush=True)
            if args.fail_fast:
                break
            continue
        try:
            summary = json.loads(proc.stdout)
        except json.JSONDecodeError:
            failures.append(r.stem)
            print(f"[{i}/{len(runs)}] {r.stem}: FAILED (no JSON summary)\n"
                  f"{proc.stdout[-1000:]}", flush=True)
            continue
        run_end = summary["runs"][r.stem]["run_end"]
        if abs(run_end - r.scan_end()) > 1e-6:
            failures.append(r.stem)
            print(f"[{i}/{len(runs)}] {r.stem}: FAILED grid end {run_end} != "
                  f"scan end {r.scan_end()}", flush=True)
            continue
        unmapped = sum(m.get("unmapped", 0) for m in (summary.get("media") or {}).values())
        if unmapped:
            failures.append(r.stem)
            print(f"[{i}/{len(runs)}] {r.stem}: FAILED {unmapped} unmapped media "
                  "items", flush=True)
            continue
        state = "up to date" if summary.get("up_to_date") else "composed"
        streams = ",".join(s.split("_features")[0].removeprefix("movies_")
                           for s in summary.get("streams", {}))
        med = summary.get("media", {}).get(r.stem)
        med_s = (f"; media {med['frames']} frames/{med['audio_items']} audio"
                 if med else "")
        print(f"[{i}/{len(runs)}] {r.stem}: {state} [{streams}] grid {run_end} s"
              f"{med_s} ({time.time() - t1:.0f} s)", flush=True)
    print(f"done: {len(runs) - len(failures)}/{len(runs)} ok in "
          f"{(time.time() - t0) / 60:.1f} min")
    if failures:
        print("FAILED runs:\n  " + "\n  ".join(failures))
        return 1
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """The tree against the run list: every run has its tables, the grid end
    recorded in the sidecar equals the BOLD scan end, and media is present
    when expected. Exit 1 on any gap."""
    runs = discover_runs(args.subject)
    comparability = comparability_path(args)
    problems = []
    n_tables = n_media = 0
    for r in runs:
        metas = sorted((r.out_dir / "features").glob("movies_*_features.meta.json")) \
            if r.out_dir.is_dir() else []
        if not metas:
            problems.append(f"{r.stem}: no composed tables")
            continue
        n_tables += 1
        meta = json.loads(metas[0].read_text())
        lo = (meta.get("inputs_signature") or {}).get("params", {}).get("lead_out")
        if lo is None or abs(lo - r.lead_out()) > 1e-6:
            problems.append(f"{r.stem}: sidecar lead_out {lo} != {r.lead_out()}")
        for mp in metas:
            m = json.loads(mp.read_text())
            table = (m.get("comparability") or {}).get("table")
            if table != str(comparability.resolve()):
                problems.append(f"{mp.name} ({r.stem}): comparability table "
                                f"{table} != {comparability}")
            unlabelled = sorted(k for k, v in (m.get("models") or {}).items()
                                if v.get("comparable") is None)
            if unlabelled:
                problems.append(f"{mp.name} ({r.stem}): no comparable label "
                                f"for {unlabelled}")
        if r.media_done():
            n_media += 1
        elif not args.no_media:
            problems.append(f"{r.stem}: no media frames")
    desc = OUT_ROOT / "dataset_description.json"
    if not desc.exists():
        problems.append(f"missing {desc}")
    print(f"{len(runs)} runs: {n_tables} with tables, {n_media} with media")
    if problems:
        print(f"{len(problems)} problem(s):\n  " + "\n  ".join(problems))
        return 1
    print("OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)

    def common(p):
        p.add_argument("--subject", help="bare label, e.g. 03")
        p.add_argument("--out-root", type=Path, default=None,
                       help=f"dataset root (default {OUT_ROOT}); a scratch "
                            "path here is how the campaign is smoke-tested")
        p.add_argument("--comparability", type=Path, default=None,
                       help="per-model comparability TSV (default "
                            "<out-root>/comparability.tsv)")

    p = sub.add_parser("plan", help="list runs, lead-outs, and state")
    common(p)
    p.add_argument("--json", action="store_true")
    p.add_argument("--todo", action="store_true", help="list runs not yet done")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="compose (and render) every run not done")
    common(p)
    p.add_argument("--stores", nargs="+", default=list(DEFAULT_STORES),
                   help=f"item stores (names under {STORE_DIR} or paths)")
    p.add_argument("--no-media", action="store_true")
    p.add_argument("--sparse", action="store_true", help="verb's --sparse")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("verify", help="check the tree against the run list")
    common(p)
    p.add_argument("--no-media", action="store_true", help="do not require media")
    p.set_defaults(func=cmd_verify)

    args = ap.parse_args(argv)
    if args.out_root is not None:
        global OUT_ROOT
        OUT_ROOT = args.out_root.resolve()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
