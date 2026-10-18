"""Orchestrate the export on a staged root: shift -> pronouns -> redact -> sweep.

Every step is planned in memory first. Nothing is written unless all steps
plan cleanly; then the private outputs (offsets, pronoun report) are written
outside the stage, then the staged files are replaced, then the sweep runs
on the result. With ``dry_run`` the sweep runs against the planned result
and nothing at all is written.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import dates, pronouns, redact, sweep
from .stage import (ReleaseConfig, Stage, check_private_input, check_private_output,
                    write_private)


@dataclass
class ExportOptions:
    epoch: _dt.date
    offsets_out: Path | None = None
    pronoun_report: Path | None = None
    pronoun_file: str = pronouns.DEBRIEFING
    pronoun_columns: list[str] | None = None
    skip_pronouns: bool = False
    redaction_spec: Path | None = None
    redaction_note: str = redact.DEFAULT_NOTE
    names_file: Path | None = None
    dry_run: bool = False


def run_export(stage_root: str | Path, cfg: ReleaseConfig, opts: ExportOptions,
               log: Callable[[str], None] = print) -> tuple[int, list[sweep.Finding]]:
    """Run the export. Returns (exit code, sweep findings): 0 clean, 1 findings."""
    stage = Stage(stage_root, cfg)
    tag = "[dry-run] " if opts.dry_run else ""

    # private inputs/outputs are checked before any work
    if not opts.dry_run:
        if opts.offsets_out is None:
            raise ValueError("--offsets-out is required (the secret date offsets)")
        check_private_output(opts.offsets_out, stage.root, cfg)
        if not opts.skip_pronouns:
            if opts.pronoun_report is None:
                raise ValueError("--pronoun-report is required unless pronouns are skipped")
            check_private_output(opts.pronoun_report, stage.root, cfg)
    spec = None
    if opts.redaction_spec is not None:
        spec = redact.load_spec(check_private_input(opts.redaction_spec, stage.root))
    names = None
    if opts.names_file is not None:
        names = sweep.load_names(check_private_input(opts.names_file, stage.root))

    log(f"{tag}stage: {stage.root}")
    # 1. dates
    shift = dates.plan_date_shift(stage, opts.epoch)
    log(f"{tag}1/4 date shift (epoch {opts.epoch}):")
    for line in shift.summary_lines() or ["  no sessions.tsv files found"]:
        log(line)
    # 2. pronouns
    pres = None
    if opts.skip_pronouns:
        log(f"{tag}2/4 pronouns: SKIPPED by request")
    else:
        pres = pronouns.plan_pronouns(stage, opts.pronoun_file, opts.pronoun_columns)
        log(f"{tag}2/4 pronouns in {pres.file}: {pres.n_changed} cell(s) changed, "
            f"{pres.n_review} flagged for review (columns: {', '.join(pres.columns)})")
    # 3. redaction
    if spec is None:
        log(f"{tag}3/4 redaction: NO SPEC GIVEN - transcripts are not redacted")
    else:
        rres = redact.plan_redactions(stage, spec, opts.redaction_note)
        log(f"{tag}3/4 redaction: {len(spec)} spec entries, {rres.n_replacements} "
            f"replacement(s) in {len(rres.files)} file(s); "
            f"{len(rres.sidecars)} sidecar note(s)")

    pending = sorted(stage.rel(p) for p in stage.pending)
    if opts.dry_run:
        log(f"[dry-run] would rewrite {len(pending)} staged file(s):")
        for rel in pending:
            log(f"  {rel}")
        if shift.shifted:
            log("[dry-run] would write the date offsets file (private)")
        if pres is not None and pres.changes:
            log("[dry-run] would write the pronoun review report (private)")
    else:
        if shift.shifted:
            write_private(opts.offsets_out, dates.offsets_tsv(shift), stage.root, cfg)
            log(f"wrote date offsets (private) -> {opts.offsets_out}")
        else:
            log("no participant needed shifting; offsets file not written")
        if pres is not None and pres.changes:
            write_private(opts.pronoun_report, pronouns.report_tsv(pres), stage.root, cfg)
            log(f"wrote pronoun review report (private) -> {opts.pronoun_report}")
        written = stage.commit()
        log(f"replaced {len(written)} staged file(s)")

    # 4. sweep
    if names is None:
        log(f"{tag}4/4 sweep: NO NAMES FILE GIVEN - person/user names not checked")
    findings = sweep.sweep(stage, opts.epoch, names)
    log(f"{tag}4/4 sweep: {len(findings)} finding(s) "
        f"{sweep.summary(findings) if findings else ''}".rstrip())
    for f in findings:
        log("  " + f.format())
    return (1 if findings else 0), findings
