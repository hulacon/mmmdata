"""QC dashboard generator and decision tracking for MMMData.

Generates self-contained interactive HTML dashboards from MRIQC/fMRIPrep
derivatives. Tracks per-run QC decisions as JSON files.

Typical usage::

    from neuroimaging.qc_dashboard import generate_dashboard, save_decision, load_decisions

    save_decision(decisions_dir, "03", "04", "TBencoding", "01",
                  "exclude", "Excessive motion (fd_mean=0.8)", "bhutch")

    html_path = generate_dashboard(
        mriqc_dir, fmriprep_dir, decisions_dir,
        subject="03", save_path="/tmp/qc_dashboard_sub-03.html",
    )
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from . import qc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_DECISIONS = {"keep", "exclude", "investigate"}

# Subset of key IQMs to display in the dashboard table (keep it scannable)
_DASHBOARD_BOLD_COLS = ["fd_mean", "fd_perc", "tsnr", "dvars_std", "efc", "fber"]
_DASHBOARD_ANAT_COLS = ["cnr", "cjv", "efc", "fber", "snr_total", "qi_1", "wm2max"]
_DASHBOARD_DWI_COLS = ["fd_mean", "fd_perc", "efc", "fber", "snr"]

_DASHBOARD_COLS: dict[str, list[str]] = {
    "bold": _DASHBOARD_BOLD_COLS,
    "T1w": _DASHBOARD_ANAT_COLS,
    "T2w": _DASHBOARD_ANAT_COLS,
    "dwi": _DASHBOARD_DWI_COLS,
}


# ---------------------------------------------------------------------------
# Run key helpers
# ---------------------------------------------------------------------------

def _build_run_key(
    subject: str, session: str, task: str,
    run: str | None, suffix: str,
) -> str:
    """Construct a BIDS-style run key for lookups."""
    parts = [f"sub-{subject}", f"ses-{session}", f"task-{task}"]
    if run:
        parts.append(f"run-{run}")
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


# ---------------------------------------------------------------------------
# QC decision tracking
# ---------------------------------------------------------------------------

def save_decision(
    decisions_dir: str | Path,
    subject: str,
    session: str,
    task: str,
    run: str | None,
    decision: str,
    reason: str,
    reviewer: str,
    suffix: str = "bold",
) -> dict[str, Any]:
    """Save a QC decision for a specific run.

    Decisions are stored as individual JSON files under
    ``decisions_dir/sub-{subject}/``. Each file maintains a full
    decision history (list of entries) for audit trail.

    Parameters
    ----------
    decisions_dir : path
        Root directory for QC decisions (e.g. ``derivatives/qc_decisions/``).
    subject, session, task : str
        BIDS entities (without prefixes).
    run : str or None
        Run number or None.
    decision : str
        One of ``'keep'``, ``'exclude'``, ``'investigate'``.
    reason : str
        Free-text explanation.
    reviewer : str
        Reviewer identifier.
    suffix : str
        Modality suffix (default ``'bold'``).

    Returns
    -------
    dict
        The saved decision record.
    """
    if decision not in VALID_DECISIONS:
        raise ValueError(
            f"Invalid decision {decision!r}; expected one of {sorted(VALID_DECISIONS)}"
        )

    decisions_dir = Path(decisions_dir)
    run_key = _build_run_key(subject, session, task, run, suffix)
    sub_dir = decisions_dir / f"sub-{subject}"
    sub_dir.mkdir(parents=True, exist_ok=True)

    json_path = sub_dir / f"{run_key}_decision.json"

    # Load existing history or start fresh
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = {"run_key": run_key, "decisions": []}

    record = {
        "decision": decision,
        "reason": reason,
        "reviewer": reviewer,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    data["decisions"].append(record)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    return record


def load_decisions(
    decisions_dir: str | Path,
    subject: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load all QC decisions, optionally filtered by subject.

    Parameters
    ----------
    decisions_dir : path
        Root directory for QC decisions.
    subject : str, optional
        Filter to one subject.

    Returns
    -------
    dict
        Keyed by run_key. Each value has ``latest`` (most recent entry)
        and ``history`` (full list).
    """
    decisions_dir = Path(decisions_dir)
    if not decisions_dir.exists():
        return {}

    sub_pattern = f"sub-{subject}" if subject else "sub-*"
    result: dict[str, dict[str, Any]] = {}

    for json_path in sorted(decisions_dir.glob(f"{sub_pattern}/*_decision.json")):
        with open(json_path) as f:
            data = json.load(f)
        run_key = data.get("run_key", json_path.stem.replace("_decision", ""))
        history = data.get("decisions", [])
        if history:
            result[run_key] = {"latest": history[-1], "history": history}

    return result


# ---------------------------------------------------------------------------
# HTML dashboard generation
# ---------------------------------------------------------------------------

def generate_dashboard(
    mriqc_dir: str | Path,
    fmriprep_dir: str | Path | None = None,
    decisions_dir: str | Path | None = None,
    subject: str | None = None,
    modality: str = "bold",
    save_path: str | Path | None = None,
    iqr_multiplier: float = 1.5,
    fd_threshold: float = 0.5,
    bids_root: str | Path | None = None,
) -> str:
    """Generate a self-contained interactive QC dashboard as HTML.

    Parameters
    ----------
    mriqc_dir : path
        MRIQC derivatives directory.
    fmriprep_dir : path, optional
        fMRIPrep derivatives directory. If None, motion data is skipped.
    decisions_dir : path, optional
        QC decisions directory. If None, decision column shows 'pending'.
    subject : str, optional
        Generate for one subject. If None, all subjects.
    modality : str
        ``'bold'``, ``'T1w'``, ``'T2w'``, or ``'dwi'``.
    save_path : path, optional
        Where to write the HTML. Required (no sensible default).
    iqr_multiplier : float
        IQR multiplier for outlier detection.
    fd_threshold : float
        FD threshold for motion summary (bold only).
    bids_root : path, optional
        BIDS dataset root. If provided, adds a processing status section
        showing MRIQC/fMRIPrep completion per subject.

    Returns
    -------
    str
        Absolute path to the generated HTML file.
    """
    mriqc_dir = Path(mriqc_dir)

    # 1. Gather data
    iqm_rows = qc.get_iqm_table(mriqc_dir, modality, subject=subject)
    outlier_result = qc.detect_outliers(
        mriqc_dir, modality, scope="global", subject=subject,
        iqr_multiplier=iqr_multiplier,
    )

    motion_runs = None
    if modality == "bold" and fmriprep_dir is not None:
        motion_result = qc.summarize_motion(
            fmriprep_dir, subject=subject, fd_threshold=fd_threshold,
        )
        motion_runs = motion_result.get("runs", [])

    decisions = None
    if decisions_dir is not None:
        decisions_dir_path = Path(decisions_dir)
        if decisions_dir_path.exists():
            decisions = load_decisions(decisions_dir_path, subject=subject)

    report_result = qc.list_reports(mriqc_dir=mriqc_dir)
    report_map = {}
    if "mriqc" in report_result and "reports" in report_result["mriqc"]:
        for r in report_result["mriqc"]["reports"]:
            rk = _build_run_key(
                r.get("subject") or "", r.get("session") or "",
                r.get("task") or "", r.get("run"), r.get("suffix") or "",
            )
            report_map[rk] = r["filename"]

    # 1b. Processing status (cross-modality)
    processing = None
    if bids_root is not None:
        processing = qc.processing_status(
            bids_root, mriqc_dir,
            fmriprep_dir or mriqc_dir,
            subject=subject,
            pipeline="both" if fmriprep_dir else "mriqc",
        )

    # 2. Merge into unified run list
    key_metrics = _DASHBOARD_COLS.get(modality, _DASHBOARD_BOLD_COLS)
    runs = _merge_run_data(
        iqm_rows, outlier_result, motion_runs, decisions,
        report_map, modality, mriqc_dir,
    )

    # 2b. Subject-level summary (current modality)
    subject_summary = _build_subject_summary(
        outlier_result, motion_runs, decisions, modality,
    )

    # 3. Render HTML
    html = _render_html(
        runs, outlier_result, modality, subject,
        mriqc_dir, key_metrics, fd_threshold,
        processing=processing,
        subject_summary=subject_summary,
    )

    # 4. Save
    if save_path is None:
        raise ValueError("save_path is required")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html, encoding="utf-8")

    return str(save_path.resolve())


# ---------------------------------------------------------------------------
# Internal: Plotly JS embedding
# ---------------------------------------------------------------------------

def _get_plotly_js() -> str:
    """Return the Plotly JS source for inline embedding.

    Uses the installed plotly package so the dashboard works offline
    (e.g. on HPC nodes without internet access).
    """
    try:
        from plotly.offline import get_plotlyjs
        return get_plotlyjs()
    except ImportError:
        return "/* plotly not installed */"


# ---------------------------------------------------------------------------
# Internal: subject-level summary
# ---------------------------------------------------------------------------

def _build_subject_summary(
    outlier_result: dict,
    motion_runs: list[dict] | None,
    decisions: dict[str, dict] | None,
    modality: str,
) -> list[dict]:
    """Collapse per-run data into per-subject rows for the summary table.

    Returns a list of dicts with keys: subject, n_runs, n_outliers,
    pct_outlier, mean_fd, mean_high_motion_pct, n_reviewed, n_pending,
    n_exclude.
    """
    outlier_summary = outlier_result.get("summary_by_subject", {})

    # Aggregate motion per subject
    motion_by_sub: dict[str, dict[str, list[float]]] = {}
    if motion_runs:
        for r in motion_runs:
            s = r.get("subject", "")
            if s not in motion_by_sub:
                motion_by_sub[s] = {"fd_vals": [], "hm_pcts": []}
            if r.get("mean_fd") is not None:
                motion_by_sub[s]["fd_vals"].append(r["mean_fd"])
            if r.get("pct_high_motion") is not None:
                motion_by_sub[s]["hm_pcts"].append(r["pct_high_motion"])

    # Count decisions per subject (parse subject from run_key prefix)
    decision_counts: dict[str, dict[str, int]] = {}
    if decisions:
        for rk, dec_info in decisions.items():
            parts = rk.split("_")
            sub = parts[0].replace("sub-", "") if parts else ""
            if sub not in decision_counts:
                decision_counts[sub] = {"reviewed": 0, "keep": 0, "exclude": 0, "investigate": 0}
            decision_counts[sub]["reviewed"] += 1
            d = dec_info.get("latest", {}).get("decision", "")
            if d in decision_counts[sub]:
                decision_counts[sub][d] += 1

    all_subjects = sorted(set(
        list(outlier_summary.keys())
        + list(motion_by_sub.keys())
        + list(decision_counts.keys())
    ))

    rows = []
    for sub in all_subjects:
        os_data = outlier_summary.get(sub, {})
        n_runs = os_data.get("n_runs", 0)
        n_outliers = os_data.get("n_outlier_runs", 0)
        pct_outlier = os_data.get("pct_outlier", 0)

        ms = motion_by_sub.get(sub, {})
        fd_vals = ms.get("fd_vals", [])
        hm_pcts = ms.get("hm_pcts", [])

        dc = decision_counts.get(sub, {})
        n_reviewed = dc.get("reviewed", 0)

        rows.append({
            "subject": sub,
            "n_runs": n_runs,
            "n_outliers": n_outliers,
            "pct_outlier": pct_outlier,
            "mean_fd": round(sum(fd_vals) / len(fd_vals), 3) if fd_vals else None,
            "mean_high_motion_pct": round(sum(hm_pcts) / len(hm_pcts), 1) if hm_pcts else None,
            "n_reviewed": n_reviewed,
            "n_pending": n_runs - n_reviewed,
            "n_exclude": dc.get("exclude", 0),
        })

    return rows


# ---------------------------------------------------------------------------
# Internal: data merging
# ---------------------------------------------------------------------------

def _merge_run_data(
    iqm_rows: list[dict],
    outlier_result: dict,
    motion_runs: list[dict] | None,
    decisions: dict[str, dict] | None,
    report_map: dict[str, str],
    modality: str,
    mriqc_dir: Path,
) -> list[dict]:
    """Merge IQM, outlier, motion, and decision data per run."""
    # Build outlier lookup
    outlier_lookup: dict[str, dict] = {}
    for o in outlier_result.get("outliers", []):
        ok = _build_run_key(
            o.get("subject", ""), o.get("session", ""),
            o.get("task", ""), o.get("run"), modality,
        )
        outlier_lookup[ok] = o

    # Build motion lookup
    motion_lookup: dict[str, dict] = {}
    if motion_runs:
        for m in motion_runs:
            mk = _build_run_key(
                m.get("subject", ""), m.get("session", ""),
                m.get("task", ""), m.get("run"), modality,
            )
            motion_lookup[mk] = m

    merged = []
    for row in iqm_rows:
        rk = _build_run_key(
            row.get("subject", ""), row.get("session", ""),
            row.get("task", ""), row.get("run"), modality,
        )
        entry = {
            "run_key": rk,
            "subject": row.get("subject"),
            "session": row.get("session"),
            "task": row.get("task"),
            "run": row.get("run"),
            "iqms": {k: row.get(k) for k in _DASHBOARD_COLS.get(modality, [])},
            "is_outlier": rk in outlier_lookup,
            "flagged_metrics": outlier_lookup.get(rk, {}).get("flagged_metrics", {}),
            "motion": motion_lookup.get(rk),
            "decision": (decisions or {}).get(rk, {}).get("latest"),
            "report_filename": report_map.get(rk),
            "report_path": str(mriqc_dir / report_map[rk]) if rk in report_map else None,
        }
        merged.append(entry)

    return merged


# ---------------------------------------------------------------------------
# Internal: processing status rendering
# ---------------------------------------------------------------------------

def _render_processing_status(
    processing: dict,
    subject: str | None,
) -> str:
    """Render the processing status overview section."""
    totals = processing.get("totals", {})
    subjects = processing.get("subjects", {})
    pipeline = processing.get("pipeline", "both")

    # Summary cards
    n_bids = totals.get("n_bids_bold", 0)
    cards = [f'<div class="card">{n_bids}<span>BIDS BOLD Runs</span></div>']

    if "pct_mriqc_complete" in totals:
        pct = totals["pct_mriqc_complete"]
        cls = "card-ok" if pct >= 95 else "card-warn" if pct >= 80 else "card-error"
        cards.append(f'<div class="card {cls}">{pct}%<span>MRIQC Complete</span></div>')

    if "pct_fmriprep_complete" in totals:
        pct = totals["pct_fmriprep_complete"]
        cls = "card-ok" if pct >= 95 else "card-warn" if pct >= 80 else "card-error"
        cards.append(f'<div class="card {cls}">{pct}%<span>fMRIPrep Complete</span></div>')

    cards_html = '<div class="cards">' + "".join(cards) + "</div>"

    # Per-subject progress table
    headers = ["Subject", "Sessions", "BIDS BOLD"]
    if pipeline in ("mriqc", "both"):
        headers += ["MRIQC", "Missing"]
    if pipeline in ("fmriprep", "both"):
        headers += ["fMRIPrep", "Missing"]

    th = "".join(f"<th>{h}</th>" for h in headers)
    rows_html = []
    for sub, info in sorted(subjects.items()):
        cells = [
            f"<td>sub-{sub}</td>",
            f'<td>{len(info.get("bids_sessions", []))}</td>',
            f'<td>{info.get("n_bids_bold", 0)}</td>',
        ]
        if pipeline in ("mriqc", "both"):
            m = info.get("mriqc", {})
            n_missing = m.get("n_missing_bold", 0)
            missing_cls = "cell-flagged" if n_missing > 0 else ""
            cells.append(f'<td>{m.get("n_mriqc_bold", 0)}</td>')
            cells.append(f'<td class="{missing_cls}">{n_missing}</td>')
        if pipeline in ("fmriprep", "both"):
            fp = info.get("fmriprep", {})
            n_missing = fp.get("n_missing_bold", 0)
            missing_cls = "cell-flagged" if n_missing > 0 else ""
            cells.append(f'<td>{fp.get("n_fmriprep_bold", 0)}</td>')
            cells.append(f'<td class="{missing_cls}">{n_missing}</td>')

        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    table = (
        f'<table class="qc-table"><thead><tr>{th}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody></table>'
    )

    return f"{cards_html}\n{table}"


def _render_subject_summary(
    subject_summary: list[dict],
    modality: str,
) -> str:
    """Render the subject-level QC summary table."""
    if not subject_summary:
        return "<p>No subject data available.</p>"

    has_motion = modality == "bold"

    headers = ["Subject", "Runs", "Outliers", "% Outlier"]
    if has_motion:
        headers += ["Mean FD", "Mean % High Motion"]
    headers += ["Reviewed", "Pending", "Excluded"]

    th = "".join(f'<th data-col="{h}">{h}</th>' for h in headers)

    rows_html = []
    for row in subject_summary:
        pct_out = row["pct_outlier"]
        row_cls = "row-outlier" if pct_out > 20 else ""

        cells = [
            f'<td>sub-{row["subject"]}</td>',
            f'<td>{row["n_runs"]}</td>',
            f'<td>{row["n_outliers"]}</td>',
            f'<td>{pct_out}%</td>',
        ]
        if has_motion:
            mfd = row.get("mean_fd")
            mhm = row.get("mean_high_motion_pct")
            cells.append(f'<td>{f"{mfd:.3f}" if mfd is not None else ""}</td>')
            cells.append(f'<td>{f"{mhm:.1f}%" if mhm is not None else ""}</td>')

        cells += [
            f'<td>{row["n_reviewed"]}</td>',
            f'<td>{row["n_pending"]}</td>',
            f'<td class="{"cell-flagged" if row["n_exclude"] > 0 else ""}">{row["n_exclude"]}</td>',
        ]

        rows_html.append(f'<tr class="{row_cls}">{"".join(cells)}</tr>')

    return (
        f'<table class="qc-table"><thead><tr>{th}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody></table>'
    )


# ---------------------------------------------------------------------------
# Internal: HTML rendering
# ---------------------------------------------------------------------------

def _render_html(
    runs: list[dict],
    outlier_result: dict,
    modality: str,
    subject: str | None,
    mriqc_dir: Path,
    key_metrics: list[str],
    fd_threshold: float,
    processing: dict | None = None,
    subject_summary: list[dict] | None = None,
) -> str:
    """Render the full HTML dashboard string."""
    title_subject = f"sub-{subject}" if subject else "All Subjects"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    n_total = len(runs)
    n_outliers = sum(1 for r in runs if r["is_outlier"])
    n_reviewed = sum(1 for r in runs if r["decision"] is not None)
    n_pending = n_total - n_reviewed

    # New overview sections
    processing_html = ""
    if processing is not None:
        processing_html = (
            '<section id="processing-status">'
            '<h2>Processing Status</h2>'
            f'{_render_processing_status(processing, subject)}'
            '</section>'
        )

    subject_summary_html = ""
    if subject_summary:
        subject_summary_html = (
            '<section id="subject-summary">'
            f'<h2>Subject Summary ({modality})</h2>'
            f'{_render_subject_summary(subject_summary, modality)}'
            '</section>'
        )

    table_html = _render_table(runs, key_metrics, modality, mriqc_dir)
    chart_html = _render_iqm_charts(runs, outlier_result, key_metrics, modality, subject)
    motion_html = ""
    if modality == "bold" and any(r["motion"] for r in runs):
        motion_html = _render_motion_chart(runs, fd_threshold)
    outlier_detail_html = _render_outlier_detail(runs, mriqc_dir)

    plotly_js = _get_plotly_js()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QC Dashboard — {title_subject} ({modality})</title>
<script>{plotly_js}</script>
{_CSS}
</head>
<body>

<header>
  <h1>QC Dashboard</h1>
  <p class="meta">{title_subject} &middot; {modality} &middot; Generated {timestamp}</p>
  <div class="cards">
    <div class="card">{n_total}<span>Total Runs</span></div>
    <div class="card card-warn">{n_outliers}<span>Outliers</span></div>
    <div class="card card-ok">{n_reviewed}<span>Reviewed</span></div>
    <div class="card">{n_pending}<span>Pending</span></div>
  </div>
</header>

{processing_html}

{subject_summary_html}

<section id="run-table">
  <h2>Run Table</h2>
  {table_html}
</section>

<section id="iqm-distributions">
  <h2>IQM Distributions</h2>
  {chart_html}
</section>

{f'<section id="motion-overview"><h2>Motion Overview</h2>{motion_html}</section>' if motion_html else ''}

{f'<section id="outlier-detail"><h2>Outlier Detail</h2>{outlier_detail_html}</section>' if n_outliers > 0 else ''}

{_TABLE_SORT_JS}

</body>
</html>"""


def _render_table(
    runs: list[dict],
    key_metrics: list[str],
    modality: str,
    mriqc_dir: Path,
) -> str:
    """Render the sortable run table."""
    has_motion = modality == "bold" and any(r.get("motion") for r in runs)

    # Header
    headers = ["Subject", "Session", "Task", "Run", "Decision", "Outlier"]
    headers += key_metrics
    if has_motion:
        headers += ["mean_fd", "pct_high_motion"]
    headers.append("Report")

    th_cells = "".join(f'<th data-col="{h}">{h}</th>' for h in headers)
    thead = f"<thead><tr>{th_cells}</tr></thead>"

    # Rows
    tbody_rows = []
    for r in runs:
        decision = r.get("decision")
        dec_label = decision["decision"] if decision else "pending"
        dec_class = {
            "keep": "badge-keep", "exclude": "badge-exclude",
            "investigate": "badge-warn", "pending": "badge-pending",
        }.get(dec_label, "badge-pending")
        dec_title = ""
        if decision:
            dec_title = f'{decision.get("reviewer", "")} — {decision.get("reason", "")}'

        row_class = "row-outlier" if r["is_outlier"] else ""
        border_class = f"border-{dec_label}"

        cells = [
            f'<td>{r["subject"] or ""}</td>',
            f'<td>{r["session"] or ""}</td>',
            f'<td>{r["task"] or ""}</td>',
            f'<td>{r["run"] or ""}</td>',
            f'<td><span class="badge {dec_class}" title="{_escape(dec_title)}">{dec_label.upper()}</span></td>',
            f'<td>{"YES" if r["is_outlier"] else ""}</td>',
        ]

        flagged = r.get("flagged_metrics", {})
        for m in key_metrics:
            val = r["iqms"].get(m)
            val_str = f"{val:.4g}" if isinstance(val, (int, float)) and val is not None else ""
            cell_class = "cell-flagged" if m in flagged else ""
            cells.append(f'<td class="{cell_class}">{val_str}</td>')

        if has_motion:
            motion = r.get("motion") or {}
            mfd = motion.get("mean_fd")
            phm = motion.get("pct_high_motion")
            cells.append(f'<td>{f"{mfd:.3f}" if mfd is not None else ""}</td>')
            cells.append(f'<td>{f"{phm:.1f}%" if phm is not None else ""}</td>')

        if r.get("report_path"):
            link = f'file://{r["report_path"]}'
            cells.append(f'<td><a href="{link}" target="_blank">View</a></td>')
        else:
            cells.append("<td></td>")

        row_html = "".join(cells)
        tbody_rows.append(f'<tr class="{row_class} {border_class}">{row_html}</tr>')

    tbody = "<tbody>" + "\n".join(tbody_rows) + "</tbody>"
    return f'<table class="qc-table">{thead}{tbody}</table>'


def _render_iqm_charts(
    runs: list[dict],
    outlier_result: dict,
    key_metrics: list[str],
    modality: str,
    subject: str | None,
) -> str:
    """Render Plotly box plots for each key IQM."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return "<p>Plotly not available — charts skipped.</p>"

    n_metrics = len(key_metrics)
    if n_metrics == 0 or len(runs) == 0:
        return "<p>No data for charts.</p>"

    cols = min(n_metrics, 3)
    rows_count = (n_metrics + cols - 1) // cols
    fig = make_subplots(
        rows=rows_count, cols=cols,
        subplot_titles=key_metrics,
        vertical_spacing=0.08,
    )

    # Build outlier set for marking
    outlier_keys = {
        _build_run_key(
            o.get("subject", ""), o.get("session", ""),
            o.get("task", ""), o.get("run"), modality,
        )
        for o in outlier_result.get("outliers", [])
    }

    for i, metric in enumerate(key_metrics):
        row_idx = i // cols + 1
        col_idx = i % cols + 1

        # Separate normal vs outlier points
        normal_vals, normal_labels = [], []
        outlier_vals, outlier_labels = [], []

        for r in runs:
            val = r["iqms"].get(metric)
            if val is None:
                continue
            label = f'{r["subject"]}/{r["session"]}/{r["task"]}'
            if r["run"]:
                label += f'/run-{r["run"]}'
            if r["run_key"] in outlier_keys:
                outlier_vals.append(val)
                outlier_labels.append(label)
            else:
                normal_vals.append(val)
                normal_labels.append(label)

        # Box plot of all values
        all_vals = [r["iqms"].get(metric) for r in runs if r["iqms"].get(metric) is not None]
        if subject:
            group_labels = [r["session"] or "" for r in runs if r["iqms"].get(metric) is not None]
        else:
            group_labels = [r["subject"] or "" for r in runs if r["iqms"].get(metric) is not None]

        fig.add_trace(
            go.Box(
                y=all_vals, x=group_labels,
                name=metric, boxpoints="all", jitter=0.3,
                marker=dict(size=4, opacity=0.5),
                showlegend=False,
            ),
            row=row_idx, col=col_idx,
        )

        # Overlay outliers in red
        if outlier_vals:
            fig.add_trace(
                go.Scatter(
                    y=outlier_vals,
                    x=[r["subject"] if not subject else r["session"]
                       for r in runs
                       if r["run_key"] in outlier_keys and r["iqms"].get(metric) is not None],
                    mode="markers",
                    marker=dict(color="red", size=10, symbol="x"),
                    text=outlier_labels,
                    hoverinfo="text+y",
                    name="outlier",
                    showlegend=(i == 0),
                ),
                row=row_idx, col=col_idx,
            )

    fig.update_layout(
        height=300 * rows_count,
        title_text=f"IQM Distributions ({modality})",
        showlegend=True,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def _render_motion_chart(runs: list[dict], fd_threshold: float) -> str:
    """Render Plotly scatter of mean FD per run."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return "<p>Plotly not available — chart skipped.</p>"

    # Filter runs with motion data, sorted by session then run
    motion_runs = [r for r in runs if r.get("motion") and r["motion"].get("mean_fd") is not None]
    motion_runs.sort(key=lambda r: (r["subject"] or "", r["session"] or "", r["run"] or ""))

    if not motion_runs:
        return "<p>No motion data available.</p>"

    labels = []
    for r in motion_runs:
        lbl = f'{r["subject"]}/{r["session"]}'
        if r["task"]:
            lbl += f'/{r["task"]}'
        if r["run"]:
            lbl += f'/run-{r["run"]}'
        labels.append(lbl)

    fd_vals = [r["motion"]["mean_fd"] for r in motion_runs]
    colors = ["red" if r["is_outlier"] else "#3b82f6" for r in motion_runs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(fd_vals))),
        y=fd_vals,
        mode="markers",
        marker=dict(color=colors, size=8),
        text=labels,
        hoverinfo="text+y",
        name="Mean FD",
    ))
    fig.add_hline(
        y=fd_threshold, line_dash="dash", line_color="red",
        annotation_text=f"threshold={fd_threshold} mm",
    )
    fig.update_layout(
        title="Mean Framewise Displacement per Run",
        xaxis_title="Run (sorted by session)",
        yaxis_title="Mean FD (mm)",
        height=350,
        showlegend=False,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def _render_outlier_detail(runs: list[dict], mriqc_dir: Path) -> str:
    """Render a detail list for each outlier run."""
    outlier_runs = [r for r in runs if r["is_outlier"]]
    if not outlier_runs:
        return ""

    items = []
    for r in outlier_runs:
        flagged = r.get("flagged_metrics", {})
        metric_items = []
        for m, info in flagged.items():
            val = info.get("value", "?")
            direction = info.get("direction", "?")
            threshold = info.get("threshold", "?")
            metric_items.append(
                f"<li><strong>{m}</strong>: {val} ({direction}, threshold: {threshold})</li>"
            )
        metrics_list = "<ul>" + "".join(metric_items) + "</ul>" if metric_items else "<p>No details.</p>"

        label = f'sub-{r["subject"]} ses-{r["session"]} {r["task"] or ""}'
        if r["run"]:
            label += f' run-{r["run"]}'

        report_link = ""
        if r.get("report_path"):
            report_link = f' <a href="file://{r["report_path"]}" target="_blank">[MRIQC Report]</a>'

        items.append(f"""
        <details>
          <summary>{_escape(label)}{report_link}</summary>
          {metrics_list}
        </details>""")

    return "\n".join(items)


def _escape(text: str) -> str:
    """Minimal HTML escaping."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ---------------------------------------------------------------------------
# Inline CSS
# ---------------------------------------------------------------------------

_CSS = """<style>
:root {
  --bg: #ffffff; --fg: #1a1a2e;
  --ok: #22c55e; --warn: #f59e0b; --error: #ef4444;
  --pending: #94a3b8; --blue: #3b82f6;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       color: var(--fg); background: var(--bg); padding: 1.5rem; line-height: 1.5; }
header { margin-bottom: 2rem; }
h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
h2 { font-size: 1.2rem; margin: 1.5rem 0 0.75rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.25rem; }
.meta { color: #64748b; font-size: 0.85rem; margin-bottom: 1rem; }
.cards { display: flex; gap: 1rem; flex-wrap: wrap; }
.card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 0.75rem 1.25rem; font-size: 1.5rem; font-weight: 600; text-align: center; min-width: 100px; }
.card span { display: block; font-size: 0.75rem; font-weight: 400; color: #64748b; }
.card-warn { border-color: var(--warn); }
.card-ok { border-color: var(--ok); }
.card-error { border-color: var(--error); }

/* Table */
.qc-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-top: 0.5rem; }
.qc-table th { background: #f1f5f9; padding: 0.5rem 0.4rem; text-align: left;
               cursor: pointer; user-select: none; position: sticky; top: 0; white-space: nowrap; }
.qc-table th:hover { background: #e2e8f0; }
.qc-table td { padding: 0.35rem 0.4rem; border-bottom: 1px solid #e2e8f0; white-space: nowrap; }
.qc-table tbody tr:hover { background: #f8fafc; }

/* Row borders by decision */
.border-keep { border-left: 3px solid var(--ok); }
.border-exclude { border-left: 3px solid var(--error); }
.border-investigate { border-left: 3px solid var(--warn); }
.border-pending { border-left: 3px solid var(--pending); }
.row-outlier { background: #fef2f2; }

/* Cell flagged */
.cell-flagged { background: #fee2e2; font-weight: 600; }

/* Badges */
.badge { padding: 0.15rem 0.5rem; border-radius: 9999px; font-size: 0.65rem;
         font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.badge-keep { background: #dcfce7; color: #166534; }
.badge-exclude { background: #fee2e2; color: #991b1b; }
.badge-warn { background: #fef3c7; color: #92400e; }
.badge-pending { background: #f1f5f9; color: #64748b; }

/* Links */
a { color: var(--blue); text-decoration: none; }
a:hover { text-decoration: underline; }

/* Detail sections */
details { margin: 0.5rem 0; padding: 0.5rem; background: #f8fafc; border-radius: 4px; }
summary { cursor: pointer; font-weight: 500; }
details ul { margin: 0.5rem 0 0 1.5rem; }
details li { margin: 0.25rem 0; font-size: 0.85rem; }

section { margin-bottom: 2rem; }
</style>"""


# ---------------------------------------------------------------------------
# Inline JS for table sorting
# ---------------------------------------------------------------------------

_TABLE_SORT_JS = """<script>
document.querySelectorAll('.qc-table th').forEach(th => {
  th.addEventListener('click', () => {
    const table = th.closest('table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const idx = Array.from(th.parentNode.children).indexOf(th);
    const dir = th.dataset.sortDir === 'asc' ? 'desc' : 'asc';
    th.parentNode.querySelectorAll('th').forEach(h => delete h.dataset.sortDir);
    th.dataset.sortDir = dir;
    rows.sort((a, b) => {
      let av = a.children[idx]?.textContent.trim() ?? '';
      let bv = b.children[idx]?.textContent.trim() ?? '';
      const an = parseFloat(av), bn = parseFloat(bv);
      if (!isNaN(an) && !isNaN(bn)) return dir === 'asc' ? an - bn : bn - an;
      return dir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
    });
    rows.forEach(r => tbody.appendChild(r));
  });
});
</script>"""
