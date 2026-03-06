"""CLI entry point for dcm2bids config generation.

Usage::

    python -m src.python.dcm2bids_config.cli --subject sub-03 --session ses-06
    python -m src.python.dcm2bids_config.cli --subject sub-03 --session all --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..core.config import load_config
from .config_builder import build_config
from .dicom_inspect import inspect_bold_series, inspect_fieldmaps
from .overrides import apply_overrides, load_overrides
from .session_defs import SESSION_SCHEDULE, get_session_def


def _resolve_dicom_dir(bids_root: Path, subject: str, session: str) -> Path:
    """Resolve the DICOM directory for a subject/session."""
    return bids_root / "sourcedata" / subject / session / "dicom"


def _resolve_overrides_path(config_dir: Path, subject: str) -> Path:
    """Resolve the overrides TOML path for a subject."""
    return config_dir / subject / "overrides.toml"


def _resolve_output_path(config_dir: Path, subject: str, session: str) -> Path:
    """Resolve the output config JSON path."""
    return config_dir / subject / f"{session}_conf.json"


def generate_one(
    subject: str,
    session: str,
    bids_root: Path,
    config_dir: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Generate a dcm2bids config for a single subject/session.

    Returns
    -------
    dict
        Result with keys: subject, session, status, config (if generated),
        output_path (if written), warnings.
    """
    result: dict = {
        "subject": subject,
        "session": session,
        "warnings": [],
    }

    # 1. Look up session definition
    try:
        session_def = get_session_def(session)
    except KeyError:
        result["status"] = "skipped"
        result["warnings"].append(f"{session} not in session schedule")
        return result

    # 2. Load overrides
    overrides_path = _resolve_overrides_path(config_dir, subject)
    overrides = load_overrides(overrides_path)
    ovr_result = apply_overrides(session, session_def, overrides)
    session_def = ovr_result.session_def
    override_fmap = ovr_result.fmap_info

    # 3. Check for empty task list (localizer/final without overrides)
    if not session_def.tasks and not session_def.anat:
        result["status"] = "skipped"
        result["warnings"].append(
            f"{session} ({session_def.session_type}) has no tasks defined. "
            f"Add task definitions to {overrides_path}"
        )
        return result

    # 4. Inspect DICOMs for fieldmap info (unless overridden)
    fmap_info = override_fmap
    if fmap_info is None and session_def.fmap_strategy != "none":
        dicom_dir = _resolve_dicom_dir(bids_root, subject, session)
        detection = inspect_fieldmaps(dicom_dir)
        result["warnings"].extend(detection.warnings)

        # Auto-select naturalistic vs naturalistic_fm based on actual DICOMs
        if (session_def.session_type == "naturalistic"
                and detection.strategy == "series_number"):
            from .session_defs import SESSION_TYPES
            session_def = SESSION_TYPES["naturalistic_fm"]
            result["warnings"].append(
                "Auto-selected naturalistic_fm (series_number strategy)"
            )
        elif (session_def.session_type == "naturalistic_fm"
                and detection.strategy == "series_description"):
            from .session_defs import SESSION_TYPES
            session_def = SESSION_TYPES["naturalistic"]
            result["warnings"].append(
                "Auto-selected naturalistic (series_description strategy)"
            )

        if detection.groups:
            fmap_info = detection.groups
        elif session_def.fmap_groups:
            result["status"] = "error"
            result["warnings"].append(
                f"No fieldmaps detected in {dicom_dir} but session "
                f"requires groups: {session_def.fmap_groups}"
            )
            return result

    # 5. Build config
    try:
        config = build_config(
            subject, session, session_def, fmap_info,
            run_protocols=ovr_result.run_protocols,
            run_series=ovr_result.run_series,
            fmap_desc_map=ovr_result.fmap_desc_map,
        )
    except ValueError as e:
        result["status"] = "error"
        result["warnings"].append(str(e))
        return result

    result["config"] = config

    # 5b. Check for truncated / duplicate BOLD series
    dicom_dir = _resolve_dicom_dir(bids_root, subject, session)
    if dicom_dir.is_dir():
        bold_check = inspect_bold_series(dicom_dir)
        if bold_check.truncated or bold_check.duplicates:
            result["warnings"].extend(bold_check.warnings)
            # Check if run_series overrides cover the duplicates
            if ovr_result.run_series and bold_check.duplicates:
                covered_tasks = set(ovr_result.run_series.keys())
                for proto, entries in bold_check.duplicates.items():
                    # Check if any task's protocol base matches
                    task_covered = any(
                        proto.rstrip("0123456789").rstrip("_run")
                        in t.protocol_base.replace("{n}", "")
                        for t in session_def.tasks
                        if t.task_label in covered_tasks
                    )
                    if task_covered:
                        result["warnings"].append(
                            f"  -> Duplicate '{proto}' is handled by "
                            f"run_series override"
                        )

    # 6. Write or preview
    output_path = _resolve_output_path(config_dir, subject, session)

    if dry_run:
        result["status"] = "dry_run"
        result["output_path"] = str(output_path)
    else:
        if output_path.exists() and not force:
            result["status"] = "skipped"
            result["warnings"].append(
                f"Output exists: {output_path} (use --force to overwrite)"
            )
            return result

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        result["status"] = "written"
        result["output_path"] = str(output_path)

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate dcm2bids configuration files for MMMData.",
    )
    parser.add_argument(
        "--subject", required=True,
        help="Subject ID (e.g. sub-03)",
    )
    parser.add_argument(
        "--session", required=True,
        help="Session ID (e.g. ses-06) or 'all' for all sessions",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview without writing files",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing config files",
    )
    parser.add_argument(
        "--config-dir",
        help="Override config output directory (default: code/dcm2bids_configfiles)",
    )
    parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Output full config JSON to stdout (for single session)",
    )
    args = parser.parse_args(argv)

    # Load project config
    cfg = load_config()
    bids_root = Path(cfg["paths"]["bids_project_dir"])

    if args.config_dir:
        config_dir = Path(args.config_dir)
    else:
        config_dir = Path(cfg["paths"]["code_root"]) / "config" / "dcm2bids_overrides"

    # Determine sessions
    if args.session == "all":
        sessions = sorted(SESSION_SCHEDULE.keys())
    else:
        sessions = [args.session]

    # Generate
    results = []
    for session in sessions:
        result = generate_one(
            args.subject, session, bids_root, config_dir,
            dry_run=args.dry_run, force=args.force,
        )
        results.append(result)

        # Print summary line
        status = result["status"]
        warnings = "; ".join(result["warnings"]) if result["warnings"] else ""
        icon = {"written": "+", "dry_run": "~", "skipped": "-", "error": "!"}[status]
        line = f"  [{icon}] {result['subject']}/{result['session']}: {status}"
        if warnings:
            line += f"  ({warnings})"
        print(line)

    # Optionally dump JSON for single session
    if args.output_json and len(results) == 1 and "config" in results[0]:
        print(json.dumps(results[0]["config"], indent=2))

    errors = [r for r in results if r["status"] == "error"]
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
