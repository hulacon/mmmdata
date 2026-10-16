"""SLURM job tools for the MMMData MCP server.

Read-only: list a user's jobs and show one job's details. Opt-in
(``--toolsets slurm``), since they only mean something on the cluster.

Off-cluster there is no queue to read, and that is a normal state, not
a fault: the tools detect the absent binary up front and return an
error that says where the command does exist, rather than surfacing a
bare OSError from subprocess.
"""

import json
import shutil
import subprocess
from typing import Annotated

from pydantic import Field


def _slurm_missing(binary: str) -> str | None:
    """JSON error naming the fix if `binary` is not on PATH, else None."""
    if shutil.which(binary) is not None:
        return None
    return json.dumps({
        "error": (
            f"SLURM is not available on this host: `{binary}` is not on "
            f"PATH. Run this from a Talapas login or compute node -- "
            f"off-cluster (a laptop, a CI runner, the web workspace) "
            f"there is no job queue to read."
        ),
        "slurm_available": False,
    }, indent=2)


def list_jobs(
    user: Annotated[
        str | None,
        Field(description="Username to query (optional, defaults to current user)."),
    ] = None,
) -> str:
    """List current SLURM jobs. Defaults to the current user."""
    if (missing := _slurm_missing("squeue")) is not None:
        return missing

    cmd = ["squeue", "--format=%i|%j|%T|%M|%l|%P|%R", "--noheader"]
    if user:
        cmd.extend(["-u", user])
    else:
        cmd.append("--me")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "squeue timed out after 30 seconds"})
    except OSError as exc:
        return json.dumps({"error": f"could not run squeue: {exc}"})

    if result.returncode != 0:
        return json.dumps({"error": result.stderr.strip()})

    jobs = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) >= 7:
            jobs.append({
                "job_id": parts[0].strip(),
                "name": parts[1].strip(),
                "state": parts[2].strip(),
                "time": parts[3].strip(),
                "time_limit": parts[4].strip(),
                "partition": parts[5].strip(),
                "nodelist": parts[6].strip(),
            })
    return json.dumps(jobs, indent=2)


def job_info(
    job_id: Annotated[str, Field(description="The SLURM job ID.")],
) -> str:
    """Get detailed information about a specific SLURM job."""
    if (missing := _slurm_missing("scontrol")) is not None:
        return missing

    try:
        result = subprocess.run(
            ["scontrol", "show", "job", job_id],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "scontrol timed out after 30 seconds"})
    except OSError as exc:
        return json.dumps({"error": f"could not run scontrol: {exc}"})

    if result.returncode != 0:
        return json.dumps({"error": result.stderr.strip()})

    return json.dumps({"job_id": job_id, "details": result.stdout.strip()})


TOOL_FUNCTIONS = [list_jobs, job_info]
