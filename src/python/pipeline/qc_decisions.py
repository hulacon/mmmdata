"""QC decision access for Layer 2 streams.

Reads per-run JSON decisions written by ``neuroimaging.qc_dashboard``
(canonical location: ``derivatives/preprocessing_qc/sub-XX/{run_key}_decision.json``)
and exposes the set of runs that should be included in downstream streams.

One decision applies to both fMRIPrep variants — streams do not branch
on ``original`` vs ``nordic`` at this layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from neuroimaging.constants import DERIVATIVES_DIRS
from neuroimaging.io import _resolve_bids_root


EXCLUDE = "exclude"
KEEP = "keep"
INVESTIGATE = "investigate"
VALID_DECISIONS = {KEEP, EXCLUDE, INVESTIGATE}


def _decisions_dir(bids_root: Path, subject: str) -> Path:
    return bids_root / DERIVATIVES_DIRS["preprocessing_qc"] / f"sub-{subject}"


def _run_key_from_bold(bold_path: Path) -> str:
    """Strip .nii.gz to match the dashboard's run_key convention."""
    return bold_path.name.removesuffix(".nii.gz")


def load_decision(
    subject: str,
    run_key: str,
    bids_root: Optional[Path] = None,
) -> Optional[dict]:
    """Return the latest decision dict for one run, or None if not recorded."""
    bids_root = _resolve_bids_root(bids_root)
    json_path = _decisions_dir(bids_root, subject) / f"{run_key}_decision.json"
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text())
    history = data.get("decisions", [])
    if not history:
        return None
    return history[-1]


def get_included_runs(
    subject: str,
    session: str,
    bids_root: Optional[Path] = None,
    treat_investigate_as: str = "exclude",
) -> list[Path]:
    """Return sorted BOLD paths whose latest decision is not ``exclude``.

    Parameters
    ----------
    subject, session : str
        BIDS entities (without prefixes).
    bids_root : Path, optional
    treat_investigate_as : {'exclude', 'keep'}
        How to treat ``investigate`` decisions. Default ``'exclude'``
        (conservative — a run under review is held out of downstream
        streams until explicitly marked ``keep``).

    Returns
    -------
    list[Path]
        Raw BOLD paths (sorted) that should flow into Layer 2.

    Raises
    ------
    FileNotFoundError
        If any expected decision JSON is missing. Streams should not run
        on sessions where the QC gate hasn't been fully populated.
    """
    if treat_investigate_as not in {KEEP, EXCLUDE}:
        raise ValueError(
            f"treat_investigate_as must be 'keep' or 'exclude', got "
            f"{treat_investigate_as!r}"
        )

    bids_root = _resolve_bids_root(bids_root)
    func_dir = bids_root / f"sub-{subject}" / f"ses-{session}" / "func"
    if not func_dir.exists():
        return []
    bolds = sorted(func_dir.glob("*_bold.nii.gz"))

    included: list[Path] = []
    missing: list[str] = []
    for bold in bolds:
        run_key = _run_key_from_bold(bold)
        latest = load_decision(subject, run_key, bids_root=bids_root)
        if latest is None:
            missing.append(run_key)
            continue
        decision = latest.get("decision")
        if decision == EXCLUDE:
            continue
        if decision == INVESTIGATE and treat_investigate_as == EXCLUDE:
            continue
        if decision not in VALID_DECISIONS:
            raise ValueError(
                f"Invalid decision {decision!r} for sub-{subject} {run_key}"
            )
        included.append(bold)

    if missing:
        raise FileNotFoundError(
            f"Missing QC decisions for sub-{subject}/ses-{session}: "
            f"{missing}. Run scripts/generate_qc_stubs.py or record "
            f"decisions via the dashboard before running Layer 2 streams."
        )

    return included


def summarize(
    bids_root: Optional[Path] = None,
    subjects: Optional[list[str]] = None,
) -> dict[str, int]:
    """Count decisions by value across recorded JSONs. Useful for QA."""
    bids_root = _resolve_bids_root(bids_root)
    root = bids_root / DERIVATIVES_DIRS["preprocessing_qc"]
    if not root.exists():
        return {KEEP: 0, EXCLUDE: 0, INVESTIGATE: 0, "total": 0}

    counts = {KEEP: 0, EXCLUDE: 0, INVESTIGATE: 0, "total": 0}
    pattern = "sub-*" if subjects is None else None
    sub_dirs = (
        [root / f"sub-{s}" for s in subjects]
        if subjects is not None
        else sorted(root.glob(pattern))
    )
    for sub_dir in sub_dirs:
        if not sub_dir.exists():
            continue
        for json_path in sorted(sub_dir.glob("*_decision.json")):
            try:
                data = json.loads(json_path.read_text())
            except json.JSONDecodeError:
                continue
            history = data.get("decisions", [])
            if not history:
                continue
            latest = history[-1].get("decision")
            if latest in counts:
                counts[latest] += 1
            counts["total"] += 1
    return counts
