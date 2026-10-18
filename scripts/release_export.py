#!/usr/bin/env python3
"""
release_export.py — de-identify a STAGED copy of the dataset for release.

Never run on the working tree: every writing subcommand refuses the
configured BIDS root (or anything inside it), the source-data directory, and
any directory containing either. Copy or hardlink-copy the tree to a staging
path first (a cluster job, not done here); transforms replace files by
temp-file + rename, so hardlinked working-tree files keep their bytes.

Private outputs (date offsets, pronoun review report, flag list) must be
outside the stage and the BIDS root and must not exist yet. Private inputs
(redaction spec, names list) must be outside the stage. See the `release`
package docstring for the redaction spec format and the pronoun rules.

Usage:
  release_export.py shift    STAGE --offsets-out PRIV.tsv [--epoch YYYY-MM-DD] [--dry-run]
  release_export.py pronouns STAGE --report PRIV.tsv [--file phenotype/final_debriefing.tsv] [--columns a,b] [--dry-run]
  release_export.py redact   STAGE --spec SPEC.tsv [--note TEXT] [--dry-run]
  release_export.py flag     ROOT [--glob 'sub-*/**/*_beh.tsv' ...] [--columns word] [--out PRIV.tsv]
  release_export.py sweep    ROOT [--names-file NAMES.txt] [--epoch YYYY-MM-DD] [--ext .tsv ...]
  release_export.py export   STAGE --offsets-out PRIV.tsv --pronoun-report PRIV.tsv
                             [--redactions SPEC.tsv] [--names-file NAMES.txt] [--epoch ...] [--dry-run]

Exit codes: 0 success / clean sweep, 1 sweep findings, 2 refused or failed.
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))

from release import dates, pronouns, redact, sweep  # noqa: E402
from release.export import ExportOptions, run_export  # noqa: E402
from release.stage import (  # noqa: E402
    Stage, StageError, check_private_input, check_private_output,
    load_release_config, parse_epoch, write_private,
)


def _columns(value):
    return [c.strip() for c in value.split(",") if c.strip()] if value else None


def cmd_shift(args, cfg):
    stage = Stage(args.stage, cfg)
    epoch = parse_epoch(args.epoch) if args.epoch else cfg.epoch
    if not args.dry_run:
        if not args.offsets_out:
            raise ValueError("--offsets-out is required unless --dry-run")
        check_private_output(args.offsets_out, stage.root, cfg)
    plan = dates.plan_date_shift(stage, epoch)
    tag = "[dry-run] " if args.dry_run else ""
    print(f"{tag}date shift (epoch {epoch}) on {stage.root}:")
    for line in plan.summary_lines() or ["  no sessions.tsv files found"]:
        print(line)
    if args.dry_run or not plan.shifted:
        return 0
    write_private(args.offsets_out, dates.offsets_tsv(plan), stage.root, cfg)
    print(f"wrote date offsets (private) -> {args.offsets_out}")
    print(f"replaced {len(stage.commit())} staged file(s)")
    return 0


def cmd_pronouns(args, cfg):
    stage = Stage(args.stage, cfg)
    if not args.dry_run:
        if not args.report:
            raise ValueError("--report is required unless --dry-run")
        check_private_output(args.report, stage.root, cfg)
    res = pronouns.plan_pronouns(stage, args.file, _columns(args.columns))
    tag = "[dry-run] " if args.dry_run else ""
    print(f"{tag}pronouns in {res.file}: {res.n_changed} cell(s) changed, "
          f"{res.n_review} flagged for review")
    for c in res.changes:
        print(f"  {c.row}\t{c.key}\t{c.column}\t{c.status}")
    if args.dry_run or not res.changes:
        return 0
    write_private(args.report, pronouns.report_tsv(res), stage.root, cfg)
    print(f"wrote review report (private) -> {args.report}")
    print(f"replaced {len(stage.commit())} staged file(s)")
    return 0


def cmd_redact(args, cfg):
    stage = Stage(args.stage, cfg)
    spec = redact.load_spec(check_private_input(args.spec, stage.root))
    res = redact.plan_redactions(stage, spec, args.note or redact.DEFAULT_NOTE)
    tag = "[dry-run] " if args.dry_run else ""
    print(f"{tag}redaction: {len(spec)} entries, {res.n_replacements} replacement(s)")
    for rel, n in sorted(res.files.items()):
        print(f"  {rel}: {n}")
    for side in res.sidecars:
        print(f"  {side}: Redaction note")
    if not args.dry_run:
        print(f"replaced {len(stage.commit())} staged file(s)")
    return 0


def cmd_flag(args, cfg):
    stage = Stage(args.root, cfg, writable=False)
    if args.out:
        check_private_output(args.out, stage.root, cfg)
    flags = redact.flag_candidates(stage, args.glob or redact.DEFAULT_FLAG_GLOBS,
                                   _columns(args.columns))
    text = redact.flags_tsv(flags)
    print(redact.ADVISORY, file=sys.stderr)
    if args.out:
        write_private(args.out, text, stage.root, cfg)
        print(f"{len(flags)} candidate(s) -> {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


def cmd_sweep(args, cfg):
    stage = Stage(args.root, cfg, writable=False)
    epoch = parse_epoch(args.epoch) if args.epoch else cfg.epoch
    names = None
    if args.names_file:
        names = sweep.load_names(check_private_input(args.names_file, stage.root))
    else:
        print("WARNING: no --names-file; person/user names not checked", file=sys.stderr)
    findings = sweep.sweep(stage, epoch, names, tuple(args.ext) if args.ext else
                           sweep.TEXT_EXTENSIONS)
    print("path:line\tkind\tmatch")
    for f in findings:
        print(f.format())
    status = "CLEAN" if not findings else f"{len(findings)} finding(s) {sweep.summary(findings)}"
    print(f"sweep of {stage.root}: {status}", file=sys.stderr)
    return 1 if findings else 0


def cmd_export(args, cfg):
    opts = ExportOptions(
        epoch=parse_epoch(args.epoch) if args.epoch else cfg.epoch,
        offsets_out=Path(args.offsets_out) if args.offsets_out else None,
        pronoun_report=Path(args.pronoun_report) if args.pronoun_report else None,
        pronoun_file=args.pronoun_file,
        pronoun_columns=_columns(args.pronoun_columns),
        skip_pronouns=args.skip_pronouns,
        redaction_spec=Path(args.redactions) if args.redactions else None,
        names_file=Path(args.names_file) if args.names_file else None,
        dry_run=args.dry_run,
    )
    code, _ = run_export(args.stage, cfg, opts)
    return code


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-dir", help="config directory (default: the repo's config/)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("shift", help="shift sessions.tsv acq_time to the epoch")
    s.add_argument("stage")
    s.add_argument("--offsets-out", help="private TSV of per-participant offsets")
    s.add_argument("--epoch", help="YYYY-MM-DD (default: [release] date_epoch)")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_shift)

    s = sub.add_parser("pronouns", help="neutralize gendered pronouns in free text")
    s.add_argument("stage")
    s.add_argument("--report", help="private review report TSV")
    s.add_argument("--file", default=pronouns.DEBRIEFING,
                   help=f"stage-relative TSV (default {pronouns.DEBRIEFING})")
    s.add_argument("--columns", help="comma-separated (default: all non-ID text columns)")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_pronouns)

    s = sub.add_parser("redact", help="apply a human-reviewed redaction spec")
    s.add_argument("stage")
    s.add_argument("--spec", required=True, help="redaction spec TSV (private)")
    s.add_argument("--note", help="override the sidecar Redaction note text")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_redact)

    s = sub.add_parser("flag", help="list candidate spans for the reader (advisory)")
    s.add_argument("root")
    s.add_argument("--glob", action="append",
                   help="root-relative glob, repeatable (default sub-*/**/*_beh.tsv)")
    s.add_argument("--columns", help="comma-separated text columns to scan")
    s.add_argument("--out", help="private TSV to write instead of stdout")
    s.set_defaults(func=cmd_flag)

    s = sub.add_parser("sweep", help="scan for leftover identifiers (exit 1 if any)")
    s.add_argument("root")
    s.add_argument("--names-file", help="private list of user/person names")
    s.add_argument("--epoch", help="shifted dates in [epoch, 1925-12-31] are allowed")
    s.add_argument("--ext", action="append", help="text extension to scan, repeatable")
    s.set_defaults(func=cmd_sweep)

    s = sub.add_parser("export", help="shift -> pronouns -> redact -> sweep")
    s.add_argument("stage")
    s.add_argument("--offsets-out")
    s.add_argument("--pronoun-report")
    s.add_argument("--pronoun-file", default=pronouns.DEBRIEFING)
    s.add_argument("--pronoun-columns")
    s.add_argument("--skip-pronouns", action="store_true")
    s.add_argument("--redactions", help="redaction spec TSV (private)")
    s.add_argument("--names-file")
    s.add_argument("--epoch")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_export)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        cfg = load_release_config(args.config_dir)
        return args.func(args, cfg)
    except (StageError, ValueError, KeyError, FileNotFoundError) as exc:
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"ERROR: {msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
