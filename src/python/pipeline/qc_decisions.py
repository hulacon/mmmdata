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
PENDING = "pending"
VALID_DECISIONS = {KEEP, EXCLUDE, INVESTIGATE, PENDING}

#: Decisions that represent a human call rather than a placeholder.
SIGNED_OFF_DECISIONS = {KEEP, EXCLUDE, INVESTIGATE}

#: Reviewer identifiers belonging to automation. Mirrors
#: ``neuroimaging.qc_dashboard.AUTOMATED_REVIEWERS`` so that records
#: written before the ``automated`` flag existed are still recognised.
AUTOMATED_REVIEWERS = {"auto-stub", "auto", "automated", ""}


def is_signed_off(record: Optional[dict]) -> bool:
    """Return True when *record* is a decision an identifiable human made.

    Mirrors ``neuroimaging.qc_dashboard.is_signed_off``; kept here so the
    pipeline layer does not import the dashboard.
    """
    if not record:
        return False
    if record.get("automated"):
        return False
    if record.get("decision") not in SIGNED_OFF_DECISIONS:
        return False
    reviewer = (record.get("reviewer") or "").strip().lower()
    return bool(reviewer) and reviewer not in AUTOMATED_REVIEWERS


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
    treat_pending_as: str = "exclude",
) -> list[Path]:
    """Return sorted BOLD paths cleared to flow into Layer 2.

    Parameters
    ----------
    subject, session : str
        BIDS entities (without prefixes).
    bids_root : Path, optional
    treat_investigate_as : {'exclude', 'keep'}
        How to treat ``investigate`` decisions. Default ``'exclude'``
        (conservative — a run under review is held out of downstream
        streams until explicitly marked ``keep``).
    treat_pending_as : {'exclude', 'keep'}
        How to treat runs no human has signed off on — those recorded as
        ``pending``, and any record attributable to automation rather than
        a person. Default ``'exclude'``: data nobody has reviewed does not
        enter an analysis. Pass ``'keep'`` to admit unreviewed runs, which
        restores the behaviour that applied when the auto-stub generator
        wrote ``keep`` directly.

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
    for name, value in (
        ("treat_investigate_as", treat_investigate_as),
        ("treat_pending_as", treat_pending_as),
    ):
        if value not in {KEEP, EXCLUDE}:
            raise ValueError(
                f"{name} must be 'keep' or 'exclude', got {value!r}"
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
        if decision not in VALID_DECISIONS:
            raise ValueError(
                f"Invalid decision {decision!r} for sub-{subject} {run_key}"
            )
        if not is_signed_off(latest):
            # No human has signed this off, whatever value it carries.
            if treat_pending_as == EXCLUDE:
                continue
            included.append(bold)
            continue
        if decision == EXCLUDE:
            continue
        if decision == INVESTIGATE and treat_investigate_as == EXCLUDE:
            continue
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
    """Count decisions by value across recorded JSONs. Useful for QA.

    ``signed_off`` counts records attributable to a named human;
    ``awaiting_signoff`` counts everything else, including automated
    stubs that carry a non-pending value.
    """
    bids_root = _resolve_bids_root(bids_root)
    empty = {
        KEEP: 0, EXCLUDE: 0, INVESTIGATE: 0, PENDING: 0,
        "signed_off": 0, "awaiting_signoff": 0, "total": 0,
    }
    root = bids_root / DERIVATIVES_DIRS["preprocessing_qc"]
    if not root.exists():
        return empty

    counts = dict(empty)
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
            latest = history[-1]
            value = latest.get("decision")
            if value in counts:
                counts[value] += 1
            if is_signed_off(latest):
                counts["signed_off"] += 1
            else:
                counts["awaiting_signoff"] += 1
            counts["total"] += 1
    return counts
