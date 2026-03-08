"""
Validation checks that compare the manifest (reality) against the
expectations schema (intent).

Each check function has the signature:
    check_*(conn, schema, subjects, sessions) -> list[dict]

Each returned dict has keys:
    check_name, subject, session, task, run, status, expected, actual, message
"""

import json
import sqlite3
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────

def _result(check_name, subject, session, status, *,
            task=None, run=None, expected=None, actual=None, message=""):
    return {
        "check_name": check_name,
        "subject": subject,
        "session": session,
        "task": task,
        "run": run,
        "status": status,
        "expected": str(expected) if expected is not None else None,
        "actual": str(actual) if actual is not None else None,
        "message": message,
    }


def _sessions_for_type(schema, session_type):
    """Return the list of session IDs for a given session type."""
    return schema.get("session_types", {}).get(session_type, [])


def _all_sessions_for_task(schema, task_cfg):
    """Return session IDs where a task should appear."""
    # If the task config specifies explicit sessions, use those
    if "sessions" in task_cfg:
        return task_cfg["sessions"]
    # Otherwise derive from session_type
    st = task_cfg.get("session_type")
    if st:
        return _sessions_for_type(schema, st)
    return []


# ── checks ───────────────────────────────────────────────────────────────

def check_file_presence(conn, schema, subjects, sessions):
    """Verify expected bold/scan files exist and flag unexpected ones."""
    results = []
    cur = conn.cursor()
    tasks = schema.get("tasks", {})

    for sub in subjects:
        for task_name, task_cfg in tasks.items():
            task_label = task_cfg.get("task_label", task_name)
            expected_sessions = _all_sessions_for_task(schema, task_cfg)
            datatype = task_cfg.get("datatype", "func")

            # Only check func bold and beh
            if datatype == "beh":
                suffix = "beh"
            else:
                suffix = "bold"

            for ses in expected_sessions:
                if sessions and ses not in sessions:
                    continue

                # Determine expected run count
                rps = task_cfg.get("runs_per_session", 1)
                if isinstance(rps, str):  # "variable"
                    rng = task_cfg.get("runs_range", [1, 6])
                    # Just check at least minimum runs exist
                    min_runs = rng[0]
                    max_runs = rng[1]
                else:
                    min_runs = max_runs = rps

                # For bold, only count NIfTIs (not JSON sidecars)
                # For beh, only count TSVs (not JSON sidecars)
                fmt = ".nii.gz" if suffix == "bold" else ".tsv"
                actual_count = cur.execute(
                    """SELECT COUNT(*) FROM files
                       WHERE subject=? AND session=? AND task=? AND suffix=?
                         AND format=?""",
                    (sub, ses, task_label, suffix, fmt)
                ).fetchone()[0]

                if actual_count < min_runs:
                    results.append(_result(
                        "file_presence", sub, ses, "fail",
                        task=task_label,
                        expected=f">={min_runs} {suffix} runs",
                        actual=str(actual_count),
                        message=f"Missing {suffix} runs for task {task_label}"
                    ))
                elif actual_count > max_runs:
                    results.append(_result(
                        "file_presence", sub, ses, "warn",
                        task=task_label,
                        expected=f"<={max_runs} {suffix} runs",
                        actual=str(actual_count),
                        message=f"Extra {suffix} runs for task {task_label}"
                    ))
                else:
                    results.append(_result(
                        "file_presence", sub, ses, "pass",
                        task=task_label,
                        expected=f"{min_runs}-{max_runs}" if min_runs != max_runs else str(min_runs),
                        actual=str(actual_count),
                    ))

    return results


def check_total_runs(conn, schema, subjects, sessions):
    """Verify total run count per subject matches expectations."""
    results = []
    cur = conn.cursor()
    tasks = schema.get("tasks", {})

    for sub in subjects:
        for task_name, task_cfg in tasks.items():
            task_label = task_cfg.get("task_label", task_name)
            expected_total = task_cfg.get("total_runs_per_subject")
            if expected_total is None or isinstance(expected_total, str):
                continue  # skip "variable" or unset

            datatype = task_cfg.get("datatype", "func")
            if datatype == "beh":
                suffix, fmt = "beh", ".tsv"
            else:
                suffix, fmt = "bold", ".nii.gz"

            actual_total = cur.execute(
                """SELECT COUNT(*) FROM files
                   WHERE subject=? AND task=? AND suffix=? AND format=?""",
                (sub, task_label, suffix, fmt)
            ).fetchone()[0]

            if actual_total == expected_total:
                results.append(_result(
                    "total_runs", sub, "", "pass",
                    task=task_label,
                    expected=f"{expected_total} total runs",
                    actual=str(actual_total),
                ))
            elif actual_total < expected_total:
                results.append(_result(
                    "total_runs", sub, "", "fail",
                    task=task_label,
                    expected=f"{expected_total} total runs",
                    actual=str(actual_total),
                    message=f"Missing {expected_total - actual_total} {task_label} run(s) across all sessions"
                ))
            else:
                results.append(_result(
                    "total_runs", sub, "", "warn",
                    task=task_label,
                    expected=f"{expected_total} total runs",
                    actual=str(actual_total),
                    message=f"Extra {actual_total - expected_total} {task_label} run(s) across all sessions"
                ))

    return results


def check_volume_count(conn, schema, subjects, sessions):
    """Verify NIfTI volume counts match expectations."""
    results = []
    cur = conn.cursor()

    sbref_cfg = schema.get("sbref", {})
    expected_dims = sbref_cfg.get("dimensions")  # [124, 124, 69]

    # Build task -> expected volumes mapping from schema
    tasks = schema.get("tasks", {})
    vol_expectations = {}  # task_label -> (exact, range)
    for task_name, task_cfg in tasks.items():
        task_label = task_cfg.get("task_label", task_name)
        ev = task_cfg.get("expected_volumes")
        evr = task_cfg.get("expected_volumes_range")
        if ev is not None or evr is not None:
            exact = ev if isinstance(ev, int) else None
            vol_expectations[task_label] = (exact, evr)

    rows = cur.execute(
        """SELECT f.path, f.subject, f.session, f.task, f.run, f.suffix,
                  n.nx, n.ny, n.nz, n.nt
           FROM files f
           JOIN nifti_meta n ON f.path = n.path
           WHERE f.suffix IN ('bold', 'sbref')
             AND (%s OR f.subject IN (%s))
             AND (%s OR f.session IN (%s))""" % (
            "1" if not subjects else "0",
            ",".join(f"'{s}'" for s in subjects) if subjects else "'_'",
            "1" if not sessions else "0",
            ",".join(f"'{s}'" for s in sessions) if sessions else "'_'",
        )
    ).fetchall()

    for path, sub, ses, task, run, suffix, nx, ny, nz, nt in rows:
        # Check spatial dimensions
        if expected_dims and suffix in ("bold", "sbref"):
            if [nx, ny, nz] != expected_dims:
                results.append(_result(
                    "volume_count", sub, ses, "fail",
                    task=task, run=run,
                    expected=f"dims {expected_dims}",
                    actual=f"dims [{nx},{ny},{nz}]",
                    message=f"{suffix} spatial dimensions mismatch: {path}"
                ))

        # Check sbref is single volume
        if suffix == "sbref" and nt != 1:
            results.append(_result(
                "volume_count", sub, ses, "fail",
                task=task, run=run,
                expected="1 volume (sbref)",
                actual=str(nt),
                message=f"SBRef should be single volume: {path}"
            ))

        # Check bold volume count against task expectations
        if suffix == "bold" and task in vol_expectations:
            exact, vol_range = vol_expectations[task]
            if exact is not None:
                if nt != exact:
                    results.append(_result(
                        "volume_count", sub, ses, "fail",
                        task=task, run=run,
                        expected=f"{exact} volumes",
                        actual=f"{nt} volumes",
                        message=f"Volume count mismatch: {path}"
                    ))
                else:
                    results.append(_result(
                        "volume_count", sub, ses, "pass",
                        task=task, run=run,
                        expected=f"{exact} volumes",
                        actual=f"{nt} volumes",
                    ))
            elif vol_range is not None:
                lo, hi = vol_range
                if lo <= nt <= hi:
                    results.append(_result(
                        "volume_count", sub, ses, "pass",
                        task=task, run=run,
                        expected=f"{lo}-{hi} volumes",
                        actual=f"{nt} volumes",
                    ))
                else:
                    results.append(_result(
                        "volume_count", sub, ses, "fail",
                        task=task, run=run,
                        expected=f"{lo}-{hi} volumes",
                        actual=f"{nt} volumes",
                        message=f"Volume count out of range: {path}"
                    ))

    return results


def check_events_row_count(conn, schema, subjects, sessions):
    """Verify events.tsv row counts match expectations."""
    results = []
    cur = conn.cursor()
    tasks = schema.get("tasks", {})

    for sub in subjects:
        for task_name, task_cfg in tasks.items():
            if not task_cfg.get("has_events", False):
                continue

            task_label = task_cfg.get("task_label", task_name)
            expected_rows = task_cfg.get("expected_event_rows")
            rows_range = task_cfg.get("event_rows_range")

            if expected_rows is None and rows_range is None:
                continue

            event_rows = cur.execute(
                """SELECT f.session, f.run, em.n_rows, f.path
                   FROM files f
                   JOIN events_meta em ON f.path = em.path
                   WHERE f.subject=? AND f.task=? AND f.suffix='events'
                     AND (%s OR f.session IN (%s))""" % (
                    "1" if not sessions else "0",
                    ",".join(f"'{s}'" for s in sessions) if sessions else "'_'",
                ),
                (sub, task_label)
            ).fetchall()

            for ses, run, n_rows, path in event_rows:
                if rows_range:
                    lo, hi = rows_range
                    if lo <= n_rows <= hi:
                        status = "pass"
                    else:
                        status = "fail"
                    exp_str = f"{lo}-{hi} rows"
                elif expected_rows:
                    if n_rows == expected_rows:
                        status = "pass"
                    else:
                        status = "fail"
                    exp_str = f"{expected_rows} rows"
                else:
                    continue

                results.append(_result(
                    "events_row_count", sub, ses, status,
                    task=task_label, run=run,
                    expected=exp_str,
                    actual=f"{n_rows} rows",
                    message=path if status != "pass" else "",
                ))

    return results


def check_events_columns(conn, schema, subjects, sessions):
    """Verify events.tsv files have the expected column set."""
    results = []
    cur = conn.cursor()
    tasks = schema.get("tasks", {})

    for sub in subjects:
        for task_name, task_cfg in tasks.items():
            expected_cols = task_cfg.get("event_columns")
            if not expected_cols:
                continue

            task_label = task_cfg.get("task_label", task_name)
            expected_set = set(expected_cols)

            event_rows = cur.execute(
                """SELECT f.session, f.run, em.columns, f.path
                   FROM files f
                   JOIN events_meta em ON f.path = em.path
                   WHERE f.subject=? AND f.task=? AND f.suffix='events'
                     AND (%s OR f.session IN (%s))""" % (
                    "1" if not sessions else "0",
                    ",".join(f"'{s}'" for s in sessions) if sessions else "'_'",
                ),
                (sub, task_label)
            ).fetchall()

            for ses, run, columns_json, path in event_rows:
                actual_cols = set(json.loads(columns_json))
                missing = expected_set - actual_cols
                extra = actual_cols - expected_set

                if not missing and not extra:
                    status = "pass"
                    msg = ""
                elif missing:
                    status = "fail"
                    msg = f"Missing columns: {sorted(missing)}"
                else:
                    status = "warn"
                    msg = f"Extra columns: {sorted(extra)}"

                results.append(_result(
                    "events_columns", sub, ses, status,
                    task=task_label, run=run,
                    expected=f"{len(expected_cols)} columns",
                    actual=f"{len(actual_cols)} columns",
                    message=msg,
                ))

    return results


def check_events_timing(conn, schema, subjects, sessions):
    """Verify event onsets are non-negative and events end within scan duration."""
    results = []
    cur = conn.cursor()
    meta_tr = schema.get("meta", {}).get("tr", 1.5)

    rows = cur.execute(
        """SELECT f.subject, f.session, f.task, f.run,
                  em.onset_min, em.onset_max, em.end_max, em.path
           FROM files f
           JOIN events_meta em ON f.path = em.path
           WHERE f.suffix='events'
             AND (%s OR f.subject IN (%s))
             AND (%s OR f.session IN (%s))""" % (
            "1" if not subjects else "0",
            ",".join(f"'{s}'" for s in subjects) if subjects else "'_'",
            "1" if not sessions else "0",
            ",".join(f"'{s}'" for s in sessions) if sessions else "'_'",
        )
    ).fetchall()

    for sub, ses, task, run, onset_min, onset_max, end_max, path in rows:
        if onset_min is not None and onset_min < 0:
            results.append(_result(
                "events_timing", sub, ses, "warn",
                task=task, run=run,
                expected="onset >= 0",
                actual=f"onset_min={onset_min:.2f}",
                message=f"Negative onset in {path}"
            ))

        # Check that no event extends past the end of the scan
        bold_meta = cur.execute(
            """SELECT n.nt, n.tr FROM nifti_meta n
               JOIN files f ON n.path = f.path
               WHERE f.subject=? AND f.session=? AND f.task=?
                 AND (f.run=? OR (f.run IS NULL AND ? IS NULL))
                 AND f.suffix='bold'""",
            (sub, ses, task, run, run)
        ).fetchone()

        if bold_meta:
            nt, tr = bold_meta
            tr = tr or meta_tr
            scan_dur = nt * tr

            # Prefer end_max (onset + duration) if available, fall back to onset_max
            event_end = end_max if end_max is not None else onset_max
            if event_end is not None and event_end > scan_dur:
                overshoot = event_end - scan_dur
                # Within 1 TR is a timing precision issue (warn);
                # beyond 1 TR is a genuine problem (fail)
                severity = "warn" if overshoot <= tr else "fail"
                results.append(_result(
                    "events_timing", sub, ses, severity,
                    task=task, run=run,
                    expected=f"events end <= {scan_dur:.1f}s ({nt} vols × {tr}s)",
                    actual=f"last event ends at {event_end:.1f}s (+{overshoot:.2f}s)",
                    message=f"Event extends past scan duration in {path}"
                ))
            elif event_end is not None:
                results.append(_result(
                    "events_timing", sub, ses, "pass",
                    task=task, run=run,
                    expected=f"events end <= {scan_dur:.1f}s",
                    actual=f"last event ends at {event_end:.1f}s",
                ))

    return results


def check_sbref_presence(conn, schema, subjects, sessions):
    """Verify every bold run has a matching sbref."""
    results = []
    cur = conn.cursor()

    rows = cur.execute(
        """SELECT f.subject, f.session, f.task, f.run
           FROM files f
           WHERE f.suffix='bold' AND f.format='.nii.gz'
             AND (%s OR f.subject IN (%s))
             AND (%s OR f.session IN (%s))""" % (
            "1" if not subjects else "0",
            ",".join(f"'{s}'" for s in subjects) if subjects else "'_'",
            "1" if not sessions else "0",
            ",".join(f"'{s}'" for s in sessions) if sessions else "'_'",
        )
    ).fetchall()

    for sub, ses, task, run in rows:
        has_sbref = cur.execute(
            """SELECT 1 FROM files
               WHERE subject=? AND session=? AND task=? AND suffix='sbref'
                 AND (run=? OR (run IS NULL AND ? IS NULL))""",
            (sub, ses, task, run, run)
        ).fetchone()

        status = "pass" if has_sbref else "warn"
        if not has_sbref:
            results.append(_result(
                "sbref_presence", sub, ses, status,
                task=task, run=run,
                expected="sbref exists",
                actual="missing",
                message=f"No SBRef for {task} run-{run or '(none)'}"
            ))

    return results


def check_json_sidecar(conn, schema, subjects, sessions):
    """Verify every NIfTI has a matching JSON sidecar."""
    results = []
    cur = conn.cursor()

    rows = cur.execute(
        """SELECT f.subject, f.session, f.task, f.run, f.suffix, f.path
           FROM files f
           WHERE f.format='.nii.gz'
             AND (%s OR f.subject IN (%s))
             AND (%s OR f.session IN (%s))""" % (
            "1" if not subjects else "0",
            ",".join(f"'{s}'" for s in subjects) if subjects else "'_'",
            "1" if not sessions else "0",
            ",".join(f"'{s}'" for s in sessions) if sessions else "'_'",
        )
    ).fetchall()

    for sub, ses, task, run, suffix, nii_path in rows:
        json_path = nii_path.replace(".nii.gz", ".json")
        has_json = cur.execute(
            "SELECT 1 FROM files WHERE path=?", (json_path,)
        ).fetchone()

        if not has_json:
            results.append(_result(
                "json_sidecar", sub, ses, "fail",
                task=task, run=run,
                expected="JSON sidecar",
                actual="missing",
                message=f"No JSON for {nii_path}"
            ))

    return results


def check_physio_presence(conn, schema, subjects, sessions):
    """Check expected physio channels are present for all functional sessions.

    Compares against the *ideal* dataset (every bold session should have
    physio) rather than only sessions where physio was actually collected.
    Sessions listed in ``[physio] exclude_sessions`` are skipped.
    """
    results = []
    cur = conn.cursor()
    physio_cfg = schema.get("physio", {})
    expected_recs = physio_cfg.get("recording_types", [])
    if not expected_recs:
        return results

    exclude = set(physio_cfg.get("exclude_sessions", []))

    bold_sessions = cur.execute(
        """SELECT DISTINCT subject, session FROM files
           WHERE suffix='bold'
             AND (%s OR subject IN (%s))
             AND (%s OR session IN (%s))""" % (
            "1" if not subjects else "0",
            ",".join(f"'{s}'" for s in subjects) if subjects else "'_'",
            "1" if not sessions else "0",
            ",".join(f"'{s}'" for s in sessions) if sessions else "'_'",
        )
    ).fetchall()

    for sub, ses in bold_sessions:
        if ses in exclude:
            continue

        for rec in expected_recs:
            count = cur.execute(
                """SELECT COUNT(*) FROM physio_meta pm
                   JOIN files f ON pm.path = f.path
                   WHERE f.subject=? AND f.session=? AND pm.recording=?""",
                (sub, ses, rec)
            ).fetchone()[0]

            if count == 0:
                results.append(_result(
                    "physio_presence", sub, ses, "warn",
                    expected=f"{rec} physio files",
                    actual="none",
                    message=f"No {rec} physio for {sub} {ses}"
                ))

    return results


def check_eyetracking_presence(conn, schema, subjects, sessions):
    """Check eyetracking files are present for all functional sessions.

    Compares against the *ideal* dataset (every bold session should have
    eyetracking) rather than only sessions where it was actually collected.
    Sessions listed in ``[physio.eyetracking] exclude_sessions`` are skipped.
    """
    results = []
    cur = conn.cursor()

    et_cfg = schema.get("physio", {}).get("eyetracking", {})
    exclude = set(et_cfg.get("exclude_sessions", []))

    bold_sessions = cur.execute(
        """SELECT DISTINCT subject, session FROM files
           WHERE suffix='bold'
             AND (%s OR subject IN (%s))
             AND (%s OR session IN (%s))""" % (
            "1" if not subjects else "0",
            ",".join(f"'{s}'" for s in subjects) if subjects else "'_'",
            "1" if not sessions else "0",
            ",".join(f"'{s}'" for s in sessions) if sessions else "'_'",
        )
    ).fetchall()

    for sub, ses in bold_sessions:
        if ses in exclude:
            continue
        if subjects and sub not in subjects:
            continue
        if sessions and ses not in sessions:
            continue

        count = cur.execute(
            """SELECT COUNT(*) FROM physio_meta pm
               JOIN files f ON pm.path = f.path
               WHERE f.subject=? AND f.session=? AND pm.recording='eye'""",
            (sub, ses)
        ).fetchone()[0]

        if count == 0:
            results.append(_result(
                "eyetracking_presence", sub, ses, "warn",
                expected="eyetracking files",
                actual="none",
                message=f"No eyetracking for {sub} {ses}"
            ))

    return results


def check_derivative_coverage(conn, schema, subjects, sessions):
    """Check fmriprep/mriqc outputs exist for all expected source files."""
    results = []
    cur = conn.cursor()
    deriv_cfg = schema.get("derivatives", {})

    for pipeline_key, pipe_cfg in deriv_cfg.items():
        pipeline = pipe_cfg.get("pipeline", pipeline_key)
        expected_for = pipe_cfg.get("expected_for", [])

        for source_suffix in expected_for:
            source_rows = cur.execute(
                """SELECT DISTINCT f.subject, f.session
                   FROM files f
                   WHERE f.suffix=? AND f.format='.nii.gz'
                     AND (%s OR f.subject IN (%s))
                     AND (%s OR f.session IN (%s))""" % (
                    "1" if not subjects else "0",
                    ",".join(f"'{s}'" for s in subjects) if subjects else "'_'",
                    "1" if not sessions else "0",
                    ",".join(f"'{s}'" for s in sessions) if sessions else "'_'",
                ),
                (source_suffix,)
            ).fetchall()

            for sub, ses in source_rows:
                has_deriv = cur.execute(
                    """SELECT 1 FROM derivatives
                       WHERE pipeline=? AND subject=? AND session=?""",
                    (pipeline, sub, ses)
                ).fetchone()

                if not has_deriv:
                    results.append(_result(
                        "derivative_coverage", sub, ses, "warn",
                        expected=f"{pipeline} output for {source_suffix}",
                        actual="missing",
                        message=f"No {pipeline} derivatives for {sub} {ses}"
                    ))

    return results


def check_session_metadata(conn, schema, subjects, sessions):
    """Verify sessions.tsv coverage — every session with files has metadata."""
    results = []
    cur = conn.cursor()

    file_sessions = cur.execute(
        """SELECT DISTINCT subject, session FROM files
             WHERE (%s OR subject IN (%s))
             AND (%s OR session IN (%s))""" % (
            "1" if not subjects else "0",
            ",".join(f"'{s}'" for s in subjects) if subjects else "'_'",
            "1" if not sessions else "0",
            ",".join(f"'{s}'" for s in sessions) if sessions else "'_'",
        )
    ).fetchall()

    for sub, ses in file_sessions:
        has_meta = cur.execute(
            "SELECT 1 FROM session_metadata WHERE subject=? AND session=?",
            (sub, ses)
        ).fetchone()

        if not has_meta:
            results.append(_result(
                "session_metadata", sub, ses, "warn",
                expected="session metadata row",
                actual="missing",
                message=f"No session_metadata entry for {sub} {ses}"
            ))

    return results


# ── registry ─────────────────────────────────────────────────────────────

ALL_CHECKS = {
    "file_presence": check_file_presence,
    "total_runs": check_total_runs,
    "volume_count": check_volume_count,
    "events_row_count": check_events_row_count,
    "events_columns": check_events_columns,
    "events_timing": check_events_timing,
    "sbref_presence": check_sbref_presence,
    "json_sidecar": check_json_sidecar,
    "physio_presence": check_physio_presence,
    "eyetracking_presence": check_eyetracking_presence,
    "derivative_coverage": check_derivative_coverage,
    "session_metadata": check_session_metadata,
}
