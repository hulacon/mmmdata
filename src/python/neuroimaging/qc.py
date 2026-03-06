"""Quality-control utilities for MRIQC and fMRIPrep derivatives.

Standalone library functions that return Python data structures.
No JSON serialization or Anthropic schemas — those live in mmmdata-agents.

Typical usage::

    from neuroimaging.qc import get_iqm_table, detect_outliers

    table = get_iqm_table(mriqc_dir, "bold", subject="03")
    outliers = detect_outliers(mriqc_dir, "bold", scope="global")
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Constants — key IQMs by modality
# ---------------------------------------------------------------------------

BOLD_KEY_IQMS: list[str] = [
    "fd_mean", "fd_num", "fd_perc", "tsnr", "dvars_std", "dvars_nstd",
    "efc", "fber", "snr", "gcor", "aqi", "aor", "fwhm_avg", "gsr_x", "gsr_y",
]

T1W_KEY_IQMS: list[str] = [
    "cnr", "cjv", "efc", "fber", "snr_total", "snr_gm", "snr_wm",
    "qi_1", "qi_2", "fwhm_avg", "inu_range", "inu_med",
    "tpm_overlap_gm", "tpm_overlap_wm", "tpm_overlap_csf", "wm2max",
]

# T2w uses the same metrics as T1w
T2W_KEY_IQMS: list[str] = T1W_KEY_IQMS

DWI_KEY_IQMS: list[str] = [
    "fd_mean", "fd_num", "fd_perc", "efc", "fber", "snr",
    "fwhm_avg", "gsr_x", "gsr_y",
]

_KEY_IQMS_BY_MODALITY: dict[str, list[str]] = {
    "bold": BOLD_KEY_IQMS,
    "T1w": T1W_KEY_IQMS,
    "T2w": T2W_KEY_IQMS,
    "dwi": DWI_KEY_IQMS,
}

# fMRIPrep confound columns we care about for motion summary
MOTION_COLS: list[str] = [
    "framewise_displacement", "dvars", "std_dvars", "rmsd",
    "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
]

# ---------------------------------------------------------------------------
# BIDS entity parsing
# ---------------------------------------------------------------------------

_ENTITY_RE = re.compile(
    r"sub-(?P<subject>[^_]+)"
    r"(?:_ses-(?P<session>[^_]+))?"
    r"(?:_task-(?P<task>[^_]+))?"
    r"(?:_acq-(?P<acq>[^_]+))?"
    r"(?:_dir-(?P<dir>[^_]+))?"
    r"(?:_run-(?P<run>[^_]+))?"
    r"(?:_(?P<suffix>bold|T1w|T2w|dwi))?"
)


def parse_bids_entities(filename: str) -> dict[str, str | None]:
    """Extract BIDS entities from a filename (stem or full name).

    Returns dict with keys: subject, session, task, acq, dir, run, suffix.
    Missing entities are ``None``.
    """
    stem = Path(filename).stem
    # Strip extra extensions (.nii, .json, etc.)
    while "." in stem:
        stem = Path(stem).stem
    m = _ENTITY_RE.search(stem)
    if not m:
        return {k: None for k in ("subject", "session", "task", "acq", "dir", "run", "suffix")}
    return {k: m.group(k) for k in ("subject", "session", "task", "acq", "dir", "run", "suffix")}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MODALITY_GLOB: dict[str, tuple[str, str]] = {
    "bold": ("func", "*_bold.json"),
    "T1w":  ("anat", "*_T1w.json"),
    "T2w":  ("anat", "*_T2w.json"),
    "dwi":  ("dwi",  "*_dwi.json"),
}


def collect_mriqc_jsons(
    mriqc_dir: str | Path,
    modality: str = "bold",
    subject: str | None = None,
    session: str | None = None,
) -> list[Path]:
    """Glob for MRIQC IQM JSON files matching filters.

    Parameters
    ----------
    mriqc_dir : path
        Path to the ``derivatives/mriqc/`` directory.
    modality : str
        One of ``'bold'``, ``'T1w'``, ``'T2w'``, ``'dwi'``.
    subject, session : str, optional
        Filter by subject/session ID (without ``sub-``/``ses-`` prefix).

    Returns
    -------
    list of Path
        Sorted list of matching JSON file paths.
    """
    mriqc_dir = Path(mriqc_dir)
    if modality not in _MODALITY_GLOB:
        raise ValueError(f"Unknown modality {modality!r}; expected one of {list(_MODALITY_GLOB)}")

    datatype, pattern = _MODALITY_GLOB[modality]
    sub_part = f"sub-{subject}" if subject else "sub-*"
    ses_part = f"ses-{session}" if session else "ses-*"

    results = sorted(mriqc_dir.glob(f"{sub_part}/{ses_part}/{datatype}/{pattern}"))
    # Exclude timeseries JSONs (MRIQC writes *_timeseries.json alongside IQM JSONs)
    return [p for p in results if "_timeseries" not in p.name]


def load_iqms(
    json_path: str | Path,
    key_metrics: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load a single MRIQC JSON and extract IQMs.

    Parameters
    ----------
    json_path : path
        Path to an MRIQC IQM JSON file.
    key_metrics : sequence of str, optional
        If provided, only include these metric keys (plus entities).
        Metrics not present in the file are returned as ``None``.

    Returns
    -------
    dict
        ``{"entities": {...}, "iqms": {...}}`` where *entities* are parsed
        BIDS entities and *iqms* are the metric values.
    """
    json_path = Path(json_path)
    with open(json_path) as f:
        data = json.load(f)

    entities = parse_bids_entities(json_path.name)
    # Remove non-IQM keys
    iqms = {k: v for k, v in data.items() if k not in ("bids_meta", "provenance")}

    if key_metrics is not None:
        iqms = {k: iqms.get(k) for k in key_metrics}

    return {"entities": entities, "iqms": iqms}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_reports(
    mriqc_dir: str | Path | None = None,
    fmriprep_dir: str | Path | None = None,
) -> dict[str, Any]:
    """List available QC HTML reports from MRIQC and/or fMRIPrep.

    Parameters
    ----------
    mriqc_dir, fmriprep_dir : path, optional
        Directories to scan.  Pass ``None`` to skip a pipeline.

    Returns
    -------
    dict
        ``{"mriqc": {...}, "fmriprep": {...}}`` with per-pipeline report
        lists and summary counts.
    """
    result: dict[str, Any] = {}

    for name, directory in [("mriqc", mriqc_dir), ("fmriprep", fmriprep_dir)]:
        if directory is None:
            continue
        d = Path(directory)
        if not d.exists():
            result[name] = {"error": f"Directory not found: {d}"}
            continue

        reports = sorted(d.glob("*.html"))
        parsed = []
        by_subject: dict[str, int] = {}
        by_modality: dict[str, int] = {}

        for r in reports:
            ents = parse_bids_entities(r.name)
            parsed.append({"filename": r.name, **ents})
            sub = ents.get("subject") or "unknown"
            by_subject[sub] = by_subject.get(sub, 0) + 1
            # For modality, use suffix or infer from filename
            mod = ents.get("suffix") or "other"
            # fMRIPrep uses _anat.html / _func.html naming
            if mod == "other":
                if "anat" in r.name:
                    mod = "anat"
                elif "func" in r.name:
                    mod = "func"
            by_modality[mod] = by_modality.get(mod, 0) + 1

        result[name] = {
            "directory": str(d),
            "total": len(reports),
            "by_subject": by_subject,
            "by_modality": by_modality,
            "reports": parsed,
        }

    return result


def get_iqm_table(
    mriqc_dir: str | Path,
    modality: str = "bold",
    subject: str | None = None,
    session: str | None = None,
    metrics: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Read per-run MRIQC IQMs into a list of flat dicts.

    Each dict contains BIDS entity fields (subject, session, task, run, etc.)
    alongside the requested IQM values.

    Parameters
    ----------
    mriqc_dir : path
        MRIQC derivatives directory.
    modality : str
        ``'bold'``, ``'T1w'``, ``'T2w'``, or ``'dwi'``.
    subject, session : str, optional
        Optional filters (without ``sub-``/``ses-`` prefix).
    metrics : sequence of str, optional
        Specific IQMs to extract.  Default: key metrics for the modality.

    Returns
    -------
    list of dict
        One dict per run with entity + metric fields.
    """
    paths = collect_mriqc_jsons(mriqc_dir, modality, subject, session)
    if metrics is None:
        metrics = _KEY_IQMS_BY_MODALITY.get(modality, BOLD_KEY_IQMS)

    rows = []
    for p in paths:
        loaded = load_iqms(p, key_metrics=metrics)
        row = {**loaded["entities"], **loaded["iqms"]}
        rows.append(row)
    return rows


def aggregate_iqms(
    mriqc_dir: str | Path,
    modality: str = "bold",
    group_by: str = "subject",
    metrics: Sequence[str] | None = None,
    subject: str | None = None,
) -> dict[str, Any]:
    """Compute summary statistics of IQMs grouped by a BIDS entity.

    Parameters
    ----------
    mriqc_dir : path
        MRIQC derivatives directory.
    modality : str
        ``'bold'``, ``'T1w'``, ``'T2w'``, or ``'dwi'``.
    group_by : str
        ``'subject'``, ``'session'``, ``'task'``, or ``'global'``.
    metrics : sequence of str, optional
        IQMs to summarize.  Default: key metrics for the modality.
    subject : str, optional
        Restrict to one subject (useful with ``group_by='session'``).

    Returns
    -------
    dict
        ``{"modality", "group_by", "metrics", "groups": {group_val: {metric: stats}}}``
    """
    import pandas as pd

    rows = get_iqm_table(mriqc_dir, modality, subject=subject, metrics=metrics)
    if not rows:
        return {"modality": modality, "group_by": group_by, "metrics": [], "groups": {}}

    if metrics is None:
        metrics = _KEY_IQMS_BY_MODALITY.get(modality, BOLD_KEY_IQMS)

    df = pd.DataFrame(rows)
    # Only keep metrics that actually appear as columns
    available_metrics = [m for m in metrics if m in df.columns]

    if group_by == "global":
        # Single group
        stats = {}
        for m in available_metrics:
            col = pd.to_numeric(df[m], errors="coerce")
            desc = col.describe()
            stats[m] = {
                "mean": _safe_float(desc.get("mean")),
                "std": _safe_float(desc.get("std")),
                "min": _safe_float(desc.get("min")),
                "25%": _safe_float(desc.get("25%")),
                "50%": _safe_float(desc.get("50%")),
                "75%": _safe_float(desc.get("75%")),
                "max": _safe_float(desc.get("max")),
            }
        return {
            "modality": modality,
            "group_by": "global",
            "metrics": available_metrics,
            "groups": {"all": {"n_runs": len(df), **stats}},
        }

    groups: dict[str, Any] = {}
    for name, grp in df.groupby(group_by):
        stats = {"n_runs": len(grp)}
        for m in available_metrics:
            col = pd.to_numeric(grp[m], errors="coerce")
            desc = col.describe()
            stats[m] = {
                "mean": _safe_float(desc.get("mean")),
                "std": _safe_float(desc.get("std")),
                "min": _safe_float(desc.get("min")),
                "25%": _safe_float(desc.get("25%")),
                "50%": _safe_float(desc.get("50%")),
                "75%": _safe_float(desc.get("75%")),
                "max": _safe_float(desc.get("max")),
            }
        groups[str(name)] = stats

    return {
        "modality": modality,
        "group_by": group_by,
        "metrics": available_metrics,
        "groups": groups,
    }


def _safe_float(val: Any) -> float | None:
    """Convert to float, returning None for NaN/None."""
    if val is None:
        return None
    try:
        import math
        f = float(val)
        return None if math.isnan(f) else round(f, 6)
    except (TypeError, ValueError):
        return None


def detect_outliers(
    mriqc_dir: str | Path,
    modality: str = "bold",
    scope: str = "global",
    subject: str | None = None,
    metrics: Sequence[str] | None = None,
    iqr_multiplier: float = 1.5,
) -> dict[str, Any]:
    """Flag runs whose IQMs fall outside the IQR-based fence.

    Parameters
    ----------
    mriqc_dir : path
        MRIQC derivatives directory.
    modality : str
        ``'bold'``, ``'T1w'``, ``'T2w'``, or ``'dwi'``.
    scope : str
        ``'global'``: outliers relative to all runs.
        ``'within_subject'``: outliers relative to each subject's own runs.
    subject : str, optional
        Restrict analysis to one subject.
    metrics : sequence of str, optional
        IQMs to check.  Default: key metrics for the modality.
    iqr_multiplier : float
        IQR multiplier for the fence.  1.5 = standard, 3.0 = extreme only.

    Returns
    -------
    dict
        Contains ``thresholds``, ``outliers`` (list of flagged runs),
        and ``summary_by_subject``.
    """
    import pandas as pd

    rows = get_iqm_table(mriqc_dir, modality, subject=subject, metrics=metrics)
    if not rows:
        return {
            "modality": modality, "scope": scope, "iqr_multiplier": iqr_multiplier,
            "n_runs_checked": 0, "n_outlier_runs": 0, "thresholds": {},
            "outliers": [], "summary_by_subject": {},
        }

    if metrics is None:
        metrics = _KEY_IQMS_BY_MODALITY.get(modality, BOLD_KEY_IQMS)

    df = pd.DataFrame(rows)
    available_metrics = [m for m in metrics if m in df.columns]

    outlier_records = []
    thresholds: dict[str, Any] = {}

    if scope == "global":
        # Compute thresholds across all runs
        for m in available_metrics:
            col = pd.to_numeric(df[m], errors="coerce")
            q1 = col.quantile(0.25)
            q3 = col.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            thresholds[m] = {
                "lower": _safe_float(lower),
                "upper": _safe_float(upper),
                "q1": _safe_float(q1),
                "q3": _safe_float(q3),
                "iqr": _safe_float(iqr),
            }

        # Flag outlier runs
        for idx, row in df.iterrows():
            flagged: dict[str, dict] = {}
            for m in available_metrics:
                val = pd.to_numeric(row.get(m), errors="coerce") if row.get(m) is not None else float("nan")
                if pd.isna(val):
                    continue
                t = thresholds[m]
                if t["lower"] is not None and val < t["lower"]:
                    flagged[m] = {"value": _safe_float(val), "direction": "low", "threshold": t["lower"]}
                elif t["upper"] is not None and val > t["upper"]:
                    flagged[m] = {"value": _safe_float(val), "direction": "high", "threshold": t["upper"]}

            if flagged:
                outlier_records.append({
                    "subject": row.get("subject"),
                    "session": row.get("session"),
                    "task": row.get("task"),
                    "run": row.get("run"),
                    "flagged_metrics": flagged,
                    "n_flags": len(flagged),
                })

    elif scope == "within_subject":
        for sub, grp in df.groupby("subject"):
            sub_thresholds: dict[str, dict] = {}
            for m in available_metrics:
                col = pd.to_numeric(grp[m], errors="coerce")
                q1 = col.quantile(0.25)
                q3 = col.quantile(0.75)
                iqr = q3 - q1
                sub_thresholds[m] = {
                    "lower": q1 - iqr_multiplier * iqr,
                    "upper": q3 + iqr_multiplier * iqr,
                }

            for idx, row in grp.iterrows():
                flagged = {}
                for m in available_metrics:
                    val = pd.to_numeric(row.get(m), errors="coerce") if row.get(m) is not None else float("nan")
                    if pd.isna(val):
                        continue
                    t = sub_thresholds[m]
                    if val < t["lower"]:
                        flagged[m] = {"value": _safe_float(val), "direction": "low",
                                      "threshold": _safe_float(t["lower"])}
                    elif val > t["upper"]:
                        flagged[m] = {"value": _safe_float(val), "direction": "high",
                                      "threshold": _safe_float(t["upper"])}
                if flagged:
                    outlier_records.append({
                        "subject": row.get("subject"),
                        "session": row.get("session"),
                        "task": row.get("task"),
                        "run": row.get("run"),
                        "flagged_metrics": flagged,
                        "n_flags": len(flagged),
                    })
            # Store per-subject thresholds
            thresholds[str(sub)] = {
                m: {"lower": _safe_float(v["lower"]), "upper": _safe_float(v["upper"])}
                for m, v in sub_thresholds.items()
            }

    # Summary by subject
    summary_by_subject: dict[str, dict] = {}
    subject_counts = df["subject"].value_counts().to_dict()
    outlier_subject_counts: dict[str, int] = {}
    for rec in outlier_records:
        s = rec["subject"]
        outlier_subject_counts[s] = outlier_subject_counts.get(s, 0) + 1
    for s, total in subject_counts.items():
        n_out = outlier_subject_counts.get(s, 0)
        summary_by_subject[str(s)] = {
            "n_runs": int(total),
            "n_outlier_runs": n_out,
            "pct_outlier": round(100 * n_out / total, 1) if total > 0 else 0,
        }

    return {
        "modality": modality,
        "scope": scope,
        "iqr_multiplier": iqr_multiplier,
        "n_runs_checked": len(df),
        "n_outlier_runs": len(outlier_records),
        "thresholds": thresholds,
        "outliers": outlier_records,
        "summary_by_subject": summary_by_subject,
    }


def summarize_motion(
    fmriprep_dir: str | Path,
    subject: str | None = None,
    session: str | None = None,
    task: str | None = None,
    fd_threshold: float = 0.5,
) -> dict[str, Any]:
    """Summarize fMRIPrep motion confounds across BOLD runs.

    Parameters
    ----------
    fmriprep_dir : path
        fMRIPrep derivatives directory.
    subject, session, task : str, optional
        Filters (without prefixes).
    fd_threshold : float
        FD threshold in mm for counting high-motion volumes.

    Returns
    -------
    dict
        ``{"n_runs", "fd_threshold_mm", "runs": [...], "summary_by_subject": {...}}``
    """
    import pandas as pd

    fmriprep_dir = Path(fmriprep_dir)
    sub_part = f"sub-{subject}" if subject else "sub-*"
    ses_part = f"ses-{session}" if session else "ses-*"
    task_part = f"task-{task}" if task else "task-*"

    # Match confounds files both with and without a run- entity
    pattern_with_run = f"{sub_part}/{ses_part}/func/*_{task_part}_*_desc-confounds_timeseries.tsv"
    pattern_no_run = f"{sub_part}/{ses_part}/func/*_{task_part}_desc-confounds_timeseries.tsv"
    tsv_files = sorted(
        set(fmriprep_dir.glob(pattern_with_run)) | set(fmriprep_dir.glob(pattern_no_run))
    )

    runs = []
    for tsv in tsv_files:
        entities = parse_bids_entities(tsv.name)

        # Read only the columns we need
        available_cols = pd.read_csv(tsv, sep="\t", nrows=0).columns.tolist()
        use_cols = [c for c in MOTION_COLS if c in available_cols]
        if not use_cols:
            continue

        df = pd.read_csv(tsv, sep="\t", usecols=use_cols)

        run_info: dict[str, Any] = {
            "subject": entities.get("subject"),
            "session": entities.get("session"),
            "task": entities.get("task"),
            "run": entities.get("run"),
            "n_volumes": len(df),
        }

        if "framewise_displacement" in df.columns:
            fd = pd.to_numeric(df["framewise_displacement"], errors="coerce")
            run_info["mean_fd"] = _safe_float(fd.mean())
            run_info["median_fd"] = _safe_float(fd.median())
            run_info["max_fd"] = _safe_float(fd.max())
            n_high = int((fd > fd_threshold).sum())
            run_info["n_high_motion"] = n_high
            # First volume has NaN FD, so denominator is n-1
            valid_count = int(fd.notna().sum())
            run_info["pct_high_motion"] = round(100 * n_high / valid_count, 1) if valid_count > 0 else 0

        if "dvars" in df.columns:
            dvars = pd.to_numeric(df["dvars"], errors="coerce")
            run_info["mean_dvars"] = _safe_float(dvars.mean())

        if "rmsd" in df.columns:
            rmsd = pd.to_numeric(df["rmsd"], errors="coerce")
            run_info["mean_rmsd"] = _safe_float(rmsd.mean())

        runs.append(run_info)

    # Summary by subject
    summary_by_subject: dict[str, dict] = {}
    for run_info in runs:
        s = run_info.get("subject", "unknown")
        if s not in summary_by_subject:
            summary_by_subject[s] = {"n_runs": 0, "fd_values": [], "high_motion_pcts": []}
        summary_by_subject[s]["n_runs"] += 1
        if "mean_fd" in run_info and run_info["mean_fd"] is not None:
            summary_by_subject[s]["fd_values"].append(run_info["mean_fd"])
        if "pct_high_motion" in run_info:
            summary_by_subject[s]["high_motion_pcts"].append(run_info["pct_high_motion"])

    # Compute per-subject averages
    for s, info in summary_by_subject.items():
        fd_vals = info.pop("fd_values")
        hm_vals = info.pop("high_motion_pcts")
        info["mean_fd_across_runs"] = _safe_float(sum(fd_vals) / len(fd_vals)) if fd_vals else None
        info["mean_high_motion_pct"] = round(sum(hm_vals) / len(hm_vals), 1) if hm_vals else None

    return {
        "n_runs": len(runs),
        "fd_threshold_mm": fd_threshold,
        "runs": runs,
        "summary_by_subject": summary_by_subject,
    }


def processing_status(
    bids_root: str | Path,
    mriqc_dir: str | Path,
    fmriprep_dir: str | Path,
    subject: str | None = None,
    pipeline: str = "both",
) -> dict[str, Any]:
    """Compare raw BIDS data against pipeline derivatives.

    Parameters
    ----------
    bids_root : path
        BIDS dataset root.
    mriqc_dir, fmriprep_dir : path
        Derivative directories.
    subject : str, optional
        Filter by subject ID (without ``sub-`` prefix).
    pipeline : str
        ``'mriqc'``, ``'fmriprep'``, or ``'both'``.

    Returns
    -------
    dict
        Per-subject processing status with counts and completion percentages.
    """
    bids_root = Path(bids_root)
    mriqc_dir = Path(mriqc_dir)
    fmriprep_dir = Path(fmriprep_dir)

    sub_pattern = f"sub-{subject}" if subject else "sub-*"

    # Collect raw BIDS BOLD files (the main thing we track for fMRIPrep/MRIQC)
    bids_bold_files = sorted(bids_root.glob(f"{sub_pattern}/ses-*/func/*_bold.nii.gz"))
    bids_anat_files = sorted(bids_root.glob(f"{sub_pattern}/ses-*/anat/*_T1w.nii.gz"))

    # Build sets of (subject, session, task, run) tuples for BOLD
    def _bold_key(path: Path) -> tuple:
        ents = parse_bids_entities(path.name)
        return (ents["subject"], ents["session"], ents["task"], ents.get("run"))

    bids_bold_keys = {_bold_key(f) for f in bids_bold_files}

    subjects_info: dict[str, Any] = {}

    # Group by subject
    all_subjects = sorted({k[0] for k in bids_bold_keys})

    for sub in all_subjects:
        sub_bids = {k for k in bids_bold_keys if k[0] == sub}
        sub_sessions = sorted({k[1] for k in sub_bids if k[1]})

        info: dict[str, Any] = {
            "bids_sessions": sub_sessions,
            "n_bids_bold": len(sub_bids),
        }

        if pipeline in ("mriqc", "both"):
            mriqc_jsons = collect_mriqc_jsons(mriqc_dir, "bold", subject=sub)
            mriqc_keys = set()
            for p in mriqc_jsons:
                ents = parse_bids_entities(p.name)
                mriqc_keys.add((ents["subject"], ents["session"], ents["task"], ents.get("run")))
            mriqc_sessions = sorted({k[1] for k in mriqc_keys if k[1]})
            missing_bold = sub_bids - mriqc_keys
            info["mriqc"] = {
                "processed_sessions": mriqc_sessions,
                "n_mriqc_bold": len(mriqc_keys),
                "n_missing_bold": len(missing_bold),
                "missing_runs": [
                    {"session": k[1], "task": k[2], "run": k[3]}
                    for k in sorted(missing_bold)
                ][:20],  # Cap at 20 to avoid huge output
            }

        if pipeline in ("fmriprep", "both"):
            confounds = sorted(fmriprep_dir.glob(
                f"sub-{sub}/ses-*/func/*_desc-confounds_timeseries.tsv"
            ))
            fmriprep_keys = set()
            for p in confounds:
                ents = parse_bids_entities(p.name)
                fmriprep_keys.add((ents["subject"], ents["session"], ents["task"], ents.get("run")))
            fmriprep_sessions = sorted({k[1] for k in fmriprep_keys if k[1]})
            missing_bold = sub_bids - fmriprep_keys
            info["fmriprep"] = {
                "processed_sessions": fmriprep_sessions,
                "n_fmriprep_bold": len(fmriprep_keys),
                "n_missing_bold": len(missing_bold),
                "missing_runs": [
                    {"session": k[1], "task": k[2], "run": k[3]}
                    for k in sorted(missing_bold)
                ][:20],
            }

        subjects_info[sub] = info

    # Totals
    total_bids = len(bids_bold_keys)
    totals: dict[str, Any] = {"n_bids_bold": total_bids}
    if pipeline in ("mriqc", "both"):
        total_mriqc = sum(s.get("mriqc", {}).get("n_mriqc_bold", 0) for s in subjects_info.values())
        totals["n_mriqc_bold"] = total_mriqc
        totals["pct_mriqc_complete"] = round(100 * total_mriqc / total_bids, 1) if total_bids > 0 else 0
    if pipeline in ("fmriprep", "both"):
        total_fmriprep = sum(s.get("fmriprep", {}).get("n_fmriprep_bold", 0) for s in subjects_info.values())
        totals["n_fmriprep_bold"] = total_fmriprep
        totals["pct_fmriprep_complete"] = round(100 * total_fmriprep / total_bids, 1) if total_bids > 0 else 0

    return {
        "pipeline": pipeline,
        "subjects": subjects_info,
        "totals": totals,
    }


def run_details(
    mriqc_dir: str | Path,
    fmriprep_dir: str | Path,
    subject: str,
    session: str,
    task: str,
    run: str | None = None,
    suffix: str = "bold",
) -> dict[str, Any]:
    """Get full IQM details and motion summary for a specific run.

    Parameters
    ----------
    mriqc_dir, fmriprep_dir : path
        Derivative directories.
    subject, session, task : str
        BIDS entities (without prefixes).
    run : str, optional
        Run number (e.g. ``'01'``).  Omit if task has no runs.
    suffix : str
        Modality suffix: ``'bold'``, ``'T1w'``, ``'T2w'``, ``'dwi'``.

    Returns
    -------
    dict
        Combined MRIQC IQMs and fMRIPrep motion summary.
    """
    import pandas as pd

    mriqc_dir = Path(mriqc_dir)
    fmriprep_dir = Path(fmriprep_dir)

    # Build expected filename pattern
    parts = [f"sub-{subject}", f"ses-{session}", f"task-{task}"]
    if run:
        parts.append(f"run-{run}")
    parts.append(suffix)
    stem = "_".join(parts)

    result: dict[str, Any] = {
        "subject": subject, "session": session, "task": task,
        "run": run, "suffix": suffix,
    }

    # MRIQC IQMs — find matching JSON
    datatype = {"bold": "func", "T1w": "anat", "T2w": "anat", "dwi": "dwi"}.get(suffix, "func")
    mriqc_json = mriqc_dir / f"sub-{subject}" / f"ses-{session}" / datatype / f"{stem}.json"

    if mriqc_json.exists():
        loaded = load_iqms(mriqc_json)  # All metrics, no filtering
        result["mriqc_iqms"] = loaded["iqms"]
    else:
        # Try glob in case of acq/dir entities we missed
        candidates = list((mriqc_dir / f"sub-{subject}" / f"ses-{session}" / datatype).glob(
            f"*_task-{task}_*_{suffix}.json"
        ))
        if run:
            candidates = [c for c in candidates if f"run-{run}" in c.name]
        candidates = [c for c in candidates if "_timeseries" not in c.name]
        if candidates:
            loaded = load_iqms(candidates[0])
            result["mriqc_iqms"] = loaded["iqms"]
            result["mriqc_file"] = candidates[0].name
        else:
            result["mriqc_iqms"] = None
            result["mriqc_note"] = f"No MRIQC JSON found for {stem}"

    # fMRIPrep motion — only for BOLD
    if suffix == "bold":
        confound_pattern = f"sub-{subject}/ses-{session}/func/*_task-{task}_*_desc-confounds_timeseries.tsv"
        confound_files = sorted(fmriprep_dir.glob(confound_pattern))
        if run:
            confound_files = [f for f in confound_files if f"run-{run}" in f.name]

        if confound_files:
            tsv = confound_files[0]
            available_cols = pd.read_csv(tsv, sep="\t", nrows=0).columns.tolist()
            use_cols = [c for c in MOTION_COLS if c in available_cols]

            if use_cols:
                df = pd.read_csv(tsv, sep="\t", usecols=use_cols)
                motion: dict[str, Any] = {"n_volumes": len(df)}

                if "framewise_displacement" in df.columns:
                    fd = pd.to_numeric(df["framewise_displacement"], errors="coerce")
                    motion["mean_fd"] = _safe_float(fd.mean())
                    motion["median_fd"] = _safe_float(fd.median())
                    motion["max_fd"] = _safe_float(fd.max())
                    motion["std_fd"] = _safe_float(fd.std())
                    for thresh in [0.2, 0.5, 0.9]:
                        n = int((fd > thresh).sum())
                        valid = int(fd.notna().sum())
                        motion[f"pct_fd_above_{thresh}mm"] = round(100 * n / valid, 1) if valid else 0

                if "dvars" in df.columns:
                    dvars = pd.to_numeric(df["dvars"], errors="coerce")
                    motion["mean_dvars"] = _safe_float(dvars.mean())
                    motion["max_dvars"] = _safe_float(dvars.max())

                if "rmsd" in df.columns:
                    rmsd = pd.to_numeric(df["rmsd"], errors="coerce")
                    motion["mean_rmsd"] = _safe_float(rmsd.mean())

                result["fmriprep_motion"] = motion
            else:
                result["fmriprep_motion"] = None
                result["fmriprep_note"] = "Confounds file found but no motion columns"
        else:
            result["fmriprep_motion"] = None
            result["fmriprep_note"] = "No fMRIPrep confounds file found"

    return result
