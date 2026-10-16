"""Conversion tools for the MMMData MCP server.

Wraps existing raw2bids_converters scripts via subprocess.
All write operations default to dry_run=True for safety. This is a
maintainer toolset: it writes into the dataset, and ``submit_dcm2bids``
submits a SLURM job, so the server registers it only when asked
(``--toolsets conversion``).
"""

import json
import subprocess
from pathlib import Path
from typing import Annotated

from pydantic import Field

from ..paths import get_paths


def _find_python() -> str:
    """Find the best Python executable, preferring the mmmdata venv."""
    return get_paths().mmmdata_venv_python


def _get_converters_dir() -> Path:
    """Get the path to the raw2bids_converters directory."""
    return get_paths().converters_dir


def run_behavioral_conversion(
    subjects: Annotated[
        str | None,
        Field(
            description="Comma-separated subjects, e.g. 'sub-##,sub-##'. All if omitted."
        ),
    ] = None,
    tasks: Annotated[
        str | None,
        Field(
            description=(
                "Comma-separated conversion types, e.g. 'timed_events,psychopy_encoding'. "
                "All types if omitted."
            )
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        Field(description="Preview without writing files. Default: true."),
    ] = True,
    validate: Annotated[
        bool,
        Field(description="Run validation after conversion. Default: false."),
    ] = False,
) -> str:
    """Run behavioral data conversion from raw CSVs to BIDS events TSVs. Uses raw2bids_converters/run_all.py. Defaults to dry-run mode (preview only, no files written). Set dry_run=false only after confirming the preview looks correct."""
    python = _find_python()
    converters_dir = _get_converters_dir()

    script = converters_dir / "run_all.py"
    if not script.exists():
        return json.dumps({"error": f"Script not found: {script}"})

    cmd = [python, str(script)]
    if dry_run:
        cmd.append("--dry-run")
    if validate:
        cmd.append("--validate")
    if subjects:
        cmd.extend(["--subjects", subjects])
    if tasks:
        cmd.extend(["--tasks", tasks])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(converters_dir),
        )
        return json.dumps({
            "command": " ".join(cmd),
            "dry_run": dry_run,
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Conversion timed out after 600 seconds"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _get_mmmdata_root() -> Path:
    """Get the mmmdata repo root directory."""
    return get_paths().mmmdata_root


def validate_conversion(
    subjects: Annotated[
        str | None,
        Field(
            description="Comma-separated subjects, e.g. 'sub-##,sub-##'. All if omitted."
        ),
    ] = None,
    verbose: Annotated[
        bool,
        Field(description="Show detailed mismatch info. Default: false."),
    ] = False,
) -> str:
    """Validate converted BIDS events files against metainformation reference data. Reports matches, mismatches, and missing files."""
    python = _find_python()
    converters_dir = _get_converters_dir()

    script = converters_dir / "validate.py"
    if not script.exists():
        return json.dumps({"error": f"Script not found: {script}"})

    cmd = [python, str(script)]
    if subjects:
        cmd.extend(["--subjects", subjects])
    if verbose:
        cmd.append("--verbose")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(converters_dir),
        )
        return json.dumps({
            "command": " ".join(cmd),
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Validation timed out after 300 seconds"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def generate_dcm2bids_config(
    subject: Annotated[str, Field(description="Subject ID, e.g. 'sub-##'.")],
    session: Annotated[
        str,
        Field(
            description="Session ID (e.g. 'ses-##') or 'all' for all sessions. Default: 'all'."
        ),
    ] = "all",
    dry_run: Annotated[
        bool,
        Field(description="Preview without writing files. Default: true."),
    ] = True,
    force: Annotated[
        bool,
        Field(description="Overwrite existing config files. Default: false."),
    ] = False,
) -> str:
    """Generate dcm2bids configuration JSON for converting DICOM to BIDS. Creates config files from session definitions and per-subject overrides. Defaults to dry-run mode (preview only). Set dry_run=false to write config files to disk. Safe to run on login node."""
    python = _find_python()
    mmmdata_root = _get_mmmdata_root()

    script = mmmdata_root / "scripts" / "run_dcm2bids.py"
    if not script.exists():
        return json.dumps({"error": f"Script not found: {script}"})

    cmd = [
        python, str(script),
        "--subject", subject,
        "--session", session,
        "--generate-only",
    ]
    if dry_run:
        cmd.append("--dry-run")
    if force:
        cmd.append("--force")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(mmmdata_root),
        )
        return json.dumps({
            "command": " ".join(cmd),
            "dry_run": dry_run,
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Config generation timed out after 120 seconds"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def submit_dcm2bids(
    subject: Annotated[str, Field(description="Subject ID, e.g. 'sub-##'.")],
    session: Annotated[
        str,
        Field(
            description="Session ID (e.g. 'ses-##') or 'all' for all sessions. Default: 'all'."
        ),
    ] = "all",
) -> str:
    """Submit a dcm2bids DICOM-to-BIDS conversion job to SLURM. This generates the config and runs the full conversion. Returns the SLURM job ID for tracking. Requires explicit confirmation as this submits a real compute job."""
    python = _find_python()
    mmmdata_root = _get_mmmdata_root()

    script = mmmdata_root / "scripts" / "submit_dcm2bids.py"
    if not script.exists():
        return json.dumps({"error": f"Script not found: {script}"})

    cmd = [python, str(script), subject, session]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(mmmdata_root),
        )
        return json.dumps({
            "command": " ".join(cmd),
            "returncode": result.returncode,
            "stdout": result.stdout.strip() if result.stdout else "",
            "stderr": result.stderr.strip() if result.stderr else "",
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "SLURM submission timed out"})
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOL_FUNCTIONS = [
    run_behavioral_conversion,
    validate_conversion,
    generate_dcm2bids_config,
    submit_dcm2bids,
]
