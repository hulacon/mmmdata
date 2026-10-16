"""CLI entry point for dcm2bids config generation.

Usage::

    python -m src.python.dcm2bids_config.cli --subject sub-03 --session ses-06
    python -m src.python.dcm2bids_config.cli --subject sub-03 --session all --dry-run
    python -m src.python.dcm2bids_config.cli --subject sub-06 --session ses-10 --layout duckbrain

``--layout duckbrain`` writes the same config to where duckbrain reads one,
``<out-dir>/sub-XX/ses-YY/dcm2bids_config.json`` (default out-dir
``config/dcm2bids_duckbrain``). It is the bridge that lets duckbrain convert
MMMData sessions from these session definitions until duckbrain has session
templates of its own (duckbrain TODO #10), at which point it retires.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..core.config import load_config
from .config_builder import build_config
from .dicom_inspect import (
    SeriesDir,
    ambiguous_func_matches,
    inspect_bold_series,
    inspect_fieldmaps,
    list_series,
)
from .overrides import apply_overrides, load_overrides
from .session_defs import SESSION_SCHEDULE, get_session_def


def _resolve_dicom_dir(source_root: Path, subject: str, session: str) -> Path:
    """Resolve the DICOM directory for a subject/session.

    DICOMs live in the sibling source tree (config ``paths.source_dir``), not
    under the BIDS root. The raw data moved out of ``<bids_root>/sourcedata``
    so the BIDS dataset can be shared without exposing PII.
    """
    return source_root / subject / session / "dicom"


def _resolve_overrides_path(config_dir: Path, subject: str) -> Path:
    """Resolve the overrides TOML path for a subject."""
    return config_dir / subject / "overrides.toml"


LAYOUTS = ("mmmdata", "duckbrain")


def _resolve_output_path(
    config_dir: Path, subject: str, session: str, layout: str = "mmmdata"
) -> Path:
    """Resolve the output config JSON path.

    ``mmmdata``: ``<dir>/sub-XX/ses-YY_conf.json``, beside ``overrides.toml``,
    read by ``scripts/run_dcm2bids.py``. ``duckbrain``:
    ``<dir>/sub-XX/ses-YY/dcm2bids_config.json``, the path duckbrain's
    ``resolve_dcm2bids_config_path`` reads under ``[paths] dcm2bids_config_dir``.
    """
    if layout == "duckbrain":
        return config_dir / subject / session / "dcm2bids_config.json"
    if layout == "mmmdata":
        return config_dir / subject / f"{session}_conf.json"
    raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")


@dataclass
class _DuplicateCheck:
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _series_list(hits: list[SeriesDir]) -> str:
    return ", ".join(
        f"Series {s.series_number} ({s.dicom_count} DICOMs)" for s in hits
    )


def _check_duplicate_series(
    config: dict, series: list[SeriesDir], overrides_path: Path, session: str,
) -> _DuplicateCheck:
    """Refuse a BOLD/SBRef description that matches more than one series.

    dcm2bids converts every series a description matches and appends its own
    run numbers to the duplicates, ordered by AcquisitionTime, so an aborted
    and repeated run would shift or collide with the explicit ``run-NN``. The
    fix is a ``run_series`` pin naming the series to keep, for the BOLD and
    its SBRef. A pinned description whose name alone matches several series
    is reported as resolved.
    """
    check = _DuplicateCheck()
    for desc, hits in ambiguous_func_matches(config["descriptions"], series):
        task = desc["sidecar_changes"]["TaskName"]
        check.errors.append(
            f"{desc['custom_entities']} {desc['suffix']}: "
            f"{desc['criteria'].get('SeriesDescription')!r} matches "
            f"{_series_list(hits)}. Pin the series to keep in "
            f"{overrides_path} as [{session}.run_series.{task}] "
            f"<run> = {{bold = N, sbref = M}}, or exclude the task."
        )
    pinned = [d for d in config["descriptions"]
              if d.get("datatype") == "func" and "SeriesNumber" in d["criteria"]]
    unpinned = [
        {**d, "criteria": {k: v for k, v in d["criteria"].items()
                           if k != "SeriesNumber"}}
        for d in pinned
    ]
    for desc, hits in ambiguous_func_matches(unpinned, series):
        check.notes.append(
            f"{desc['custom_entities']} {desc['suffix']}: "
            f"{_series_list(hits)} share a description; resolved by "
            f"SeriesNumber pin"
        )
    return check


def generate_one(
    subject: str,
    session: str,
    source_root: Path,
    config_dir: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
    layout: str = "mmmdata",
    out_dir: Path | None = None,
) -> dict:
    """Generate a dcm2bids config for a single subject/session.

    Overrides are always read from *config_dir*; the config is written under
    *out_dir* (default *config_dir*) in *layout* (see ``_resolve_output_path``).

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
        dicom_dir = _resolve_dicom_dir(source_root, subject, session)
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
            run_descriptions=ovr_result.run_descriptions,
            fmap_desc_map=ovr_result.fmap_desc_map,
        )
    except ValueError as e:
        result["status"] = "error"
        result["warnings"].append(str(e))
        return result

    result["config"] = config

    # 5b. Truncated series warn; a description matching two series refuses.
    dicom_dir = _resolve_dicom_dir(source_root, subject, session)
    if dicom_dir.is_dir():
        bold_check = inspect_bold_series(dicom_dir)
        result["warnings"].extend(
            w for w in bold_check.warnings if w.startswith("Truncated")
        )
        ambiguous = _check_duplicate_series(
            config, list_series(dicom_dir), overrides_path, session,
        )
        result["warnings"].extend(ambiguous.notes)
        if ambiguous.errors:
            result["status"] = "error"
            result["warnings"].extend(ambiguous.errors)
            return result
    else:
        result["warnings"].append(
            f"DICOM dir not found: {dicom_dir}; duplicate-series check skipped"
        )

    # 6. Write or preview
    output_path = _resolve_output_path(out_dir or config_dir, subject, session, layout)

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
        "--layout", choices=LAYOUTS, default="mmmdata",
        help="Output layout: mmmdata (sub-XX/ses-YY_conf.json) or duckbrain "
             "(sub-XX/ses-YY/dcm2bids_config.json)",
    )
    parser.add_argument(
        "--out-dir",
        help="Where to write configs (default: the overrides dir for "
             "--layout mmmdata, config/dcm2bids_duckbrain for --layout duckbrain). "
             "Overrides are still read from --config-dir.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Output full config JSON to stdout (for single session)",
    )
    args = parser.parse_args(argv)

    # Load project config
    cfg = load_config()
    source_root = Path(cfg["paths"]["source_dir"])

    if args.config_dir:
        config_dir = Path(args.config_dir)
    else:
        config_dir = Path(cfg["paths"]["code_root"]) / "config" / "dcm2bids_overrides"

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.layout == "duckbrain":
        out_dir = Path(cfg["paths"]["code_root"]) / "config" / "dcm2bids_duckbrain"
    else:
        out_dir = config_dir

    # Determine sessions
    if args.session == "all":
        sessions = sorted(SESSION_SCHEDULE.keys())
    else:
        sessions = [args.session]

    # Generate
    results = []
    for session in sessions:
        result = generate_one(
            args.subject, session, source_root, config_dir,
            dry_run=args.dry_run, force=args.force,
            layout=args.layout, out_dir=out_dir,
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
