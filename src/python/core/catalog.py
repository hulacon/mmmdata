"""Read-only query helpers for the Contract A catalog (catalog.duckdb).

The catalog (built by ``scripts/catalog_sweep.py`` and its sibling
ingesters) is the answer to "what data exists, how was it processed, and
what did QC decide" — the successor to the frozen manifest.db. It lives
inside the BIDS tree at ``<bids_root>/inventory/catalog.duckdb`` and is
readable by any consumer (contracts §3.1).

Tables: files (bids2table), datasets (provenance + kind), sessions,
expected_units, expectation_exceptions, qc_decisions, files_supplemental,
sweep_meta; views: observed_units, resolution (four-state:
present / missing / excepted / surplus, with exception dispositions
accepted | pending).

Everything here opens the database read-only and returns plain Python
data structures, so it is safe to call from reports, notebooks, and
agent tools alike. Subject/session arguments accept both bare ("03")
and BIDS-prefixed ("sub-03") forms — the catalog stores bare labels.

Usage:
    from core import catalog

    cols, rows = catalog.run_select(
        db_path, "SELECT COUNT(*) AS n FROM files")
    summary = catalog.session_summary(db_path, "sub-03", "ses-04")
    report = catalog.resolution_report(db_path, subjects=["03"])
"""

from pathlib import Path

import duckdb

CATALOG_RELPATH = Path("inventory") / "catalog.duckdb"

# Statement prefixes allowed through run_select. The read-only
# connection already rejects writes; this gate exists to fail fast
# with a clear message instead of a duckdb error.
_READONLY_PREFIXES = ("SELECT", "WITH", "DESCRIBE", "SHOW", "SUMMARIZE")


def catalog_path(bids_root: Path | str) -> Path:
    """Path to the catalog database under a BIDS root."""
    return Path(bids_root) / CATALOG_RELPATH


def bare_label(value: str | None, prefix: str) -> str | None:
    """Strip a BIDS prefix ("sub-", "ses-") if present.

    The catalog stores bare labels; accepting both forms here removes
    the silent-empty-result footgun the legacy QC tools had.
    """
    if value is None:
        return None
    return value[len(prefix):] if value.startswith(prefix) else value


def connect_readonly(db_path: Path | str) -> duckdb.DuckDBPyConnection:
    """Open a read-only connection to the catalog database.

    Raises FileNotFoundError (naming the fix) if it does not exist.
    """
    db = Path(db_path)
    if not db.exists():
        raise FileNotFoundError(
            f"Catalog database not found: {db}. "
            "Build it with mmmdata/scripts/catalog_sweep.py (sbatch on "
            "Talapas), then catalog_expectations.py / "
            "catalog_qc_decisions.py / catalog_supplemental.py, using "
            "the /gpfs/projects/hulacon/shared/envs/catalog interpreter."
        )
    return duckdb.connect(str(db), read_only=True)


def run_select(
    db_path: Path | str,
    sql: str,
    params: list | tuple | None = None,
) -> tuple[list[str], list[dict]]:
    """Run a single read-only query against the catalog.

    Only SELECT/WITH/DESCRIBE/SHOW/SUMMARIZE statements are allowed;
    anything else raises ValueError before the database is opened (and
    the connection is read-only regardless).

    Returns (column_names, rows) where rows is a list of dicts keyed
    by column name.
    """
    sql_stripped = sql.strip().lstrip("(").strip()
    if not sql_stripped.upper().startswith(_READONLY_PREFIXES):
        raise ValueError(
            "Only read-only queries are allowed "
            f"(must start with one of {', '.join(_READONLY_PREFIXES)})."
        )

    conn = connect_readonly(db_path)
    try:
        cur = conn.execute(sql, list(params) if params else [])
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]
        return columns, rows
    finally:
        conn.close()


def _fetch_dicts(conn, sql: str, params: list) -> list[dict]:
    cur = conn.execute(sql, params)
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def session_summary(
    db_path: Path | str,
    subject: str,
    session: str,
) -> dict:
    """Build a summary of one subject/session from the catalog.

    Gathers: raw BIDS files with entities, derivative coverage with
    provenance (pipeline, version, from the datasets table),
    supplemental/dark file counts, QC decisions, session metadata
    (from the canonical sessions.tsv ingest), and the expectation
    resolution for the session (anything not plainly present).

    Content-tier metadata the frozen manifest carried (NIfTI headers,
    events row counts, physio channels) is out of scope of the catalog
    at schema_version 1.0 and is not reported.

    Args:
        db_path: Path to catalog.duckdb.
        subject: Subject label, "03" or "sub-03".
        session: Session label, "04" or "ses-04".
    """
    sub = bare_label(subject, "sub-")
    ses = bare_label(session, "ses-")

    conn = connect_readonly(db_path)
    try:
        raw_files = _fetch_dicts(
            conn,
            """SELECT datatype, task, run, suffix, ext, path
               FROM files
               WHERE dataset_relpath = '.' AND sub = ? AND ses = ?
               ORDER BY datatype, task, run, suffix""",
            [sub, ses],
        )

        derivatives = _fetch_dicts(
            conn,
            """SELECT f.dataset_relpath, d.pipeline, d.pipeline_version,
                      f.suffix, COUNT(*) AS n_files
               FROM files f
               LEFT JOIN datasets d ON f.dataset_relpath = d.relpath
               WHERE f.dataset_relpath <> '.' AND f.sub = ? AND f.ses = ?
               GROUP BY f.dataset_relpath, d.pipeline, d.pipeline_version,
                        f.suffix
               ORDER BY f.dataset_relpath, f.suffix""",
            [sub, ses],
        )

        supplemental = _fetch_dicts(
            conn,
            """SELECT category, COUNT(*) AS n_files
               FROM files_supplemental
               WHERE sub = ? AND ses = ?
               GROUP BY category ORDER BY category""",
            [sub, ses],
        )

        qc = _fetch_dicts(
            conn,
            """SELECT task, run, suffix, decision, reviewer, automated,
                      signed_off, reason
               FROM qc_decisions
               WHERE sub = ? AND ses = ?
               ORDER BY task, run""",
            [sub, ses],
        )

        meta_rows = _fetch_dicts(
            conn,
            "SELECT * FROM sessions WHERE sub = ? AND ses = ?",
            [sub, ses],
        )

        resolution_counts = {
            row["status"]: row["n"]
            for row in _fetch_dicts(
                conn,
                """SELECT status, COUNT(*) AS n FROM resolution
                   WHERE sub = ? AND ses = ? GROUP BY status""",
                [sub, ses],
            )
        }
        resolution_issues = _fetch_dicts(
            conn,
            """SELECT dataset_relpath, datatype, task, suffix, recording,
                      runs_min, runs_max, observed_n, status, disposition,
                      exception_notes
               FROM resolution
               WHERE sub = ? AND ses = ? AND status <> 'present'
               ORDER BY status, datatype, task""",
            [sub, ses],
        )

        return {
            "subject": f"sub-{sub}",
            "session": f"ses-{ses}",
            "n_raw_files": len(raw_files),
            "raw_files": raw_files,
            "derivatives": derivatives,
            "supplemental": supplemental,
            "qc_decisions": qc,
            "session_metadata": meta_rows[0] if meta_rows else None,
            "resolution_summary": resolution_counts,
            "resolution_issues": resolution_issues,
        }
    finally:
        conn.close()


def resolution_report(
    db_path: Path | str,
    subjects: list[str] | None = None,
    sessions: list[str] | None = None,
    statuses: list[str] | None = None,
    include_all: bool = False,
) -> dict:
    """Report expectation resolution from the catalog's resolution view.

    Every declared unit resolves to present / missing / excepted /
    surplus; excepted units carry a disposition (accepted | pending).
    By default returns only issues — missing, surplus, and
    excepted-pending — since present and excepted-accepted are the
    quiet states. Pass statuses to select explicitly, or
    include_all=True for every row.

    Returns a dict with keys: summary (status -> count over the
    filtered subjects/sessions, before status filtering), n_rows,
    rows.
    """
    where = []
    params: list = []

    subs = [bare_label(s, "sub-") for s in subjects] if subjects else None
    sess = [bare_label(s, "ses-") for s in sessions] if sessions else None

    if subs:
        where.append(f"sub IN ({', '.join('?' * len(subs))})")
        params.extend(subs)
    if sess:
        where.append(f"ses IN ({', '.join('?' * len(sess))})")
        params.extend(sess)

    scope_clause = " AND ".join(where) if where else "TRUE"

    if statuses:
        status_clause = f"status IN ({', '.join('?' * len(statuses))})"
        status_params = list(statuses)
    elif include_all:
        status_clause, status_params = "TRUE", []
    else:
        status_clause = ("(status IN ('missing', 'surplus') OR "
                         "(status = 'excepted' AND disposition = 'pending'))")
        status_params = []

    conn = connect_readonly(db_path)
    try:
        summary = {}
        for row in _fetch_dicts(
            conn,
            f"""SELECT status, disposition, COUNT(*) AS n FROM resolution
                WHERE {scope_clause} GROUP BY status, disposition""",
            params,
        ):
            key = row["status"]
            if row["disposition"]:
                key = f"{key}-{row['disposition']}"
            summary[key] = row["n"]

        rows = _fetch_dicts(
            conn,
            f"""SELECT dataset_relpath, sub, ses, datatype, task, suffix,
                       recording, runs_min, runs_max, observed_n, status,
                       disposition, exception_notes
                FROM resolution
                WHERE {scope_clause} AND {status_clause}
                ORDER BY
                  CASE status WHEN 'missing' THEN 0 WHEN 'surplus' THEN 1
                              WHEN 'excepted' THEN 2 ELSE 3 END,
                  sub, ses, datatype, task""",
            params + status_params,
        )

        return {"summary": summary, "n_rows": len(rows), "rows": rows}
    finally:
        conn.close()


def lookup_expectations(
    db_path: Path | str,
    task: str | None = None,
    session_type: str | None = None,
) -> dict:
    """Look up declared expectations from the catalog.

    The declaration itself is dataset-owned
    (``<bids_root>/expectations/dataset.toml``); this reads its
    expansion in the catalog (expected_units / sessions), so answers
    reflect exactly what the resolution view judges.

    With task: the expected unit shapes for that task (datasets,
    suffixes, run ranges, which sessions/subjects). With session_type:
    the sessions of that type and the unit shapes expected in them.
    With neither: an overview (tasks, session types, subjects,
    datasets carrying expectations).
    """
    result: dict = {}
    conn = connect_readonly(db_path)
    try:
        if task:
            units = _fetch_dicts(
                conn,
                """SELECT dataset_relpath, datatype, suffix, des, recording,
                          runs_min, runs_max,
                          COUNT(DISTINCT sub) AS n_subjects,
                          list_sort(list(DISTINCT ses)) AS sessions
                   FROM expected_units WHERE task = ?
                   GROUP BY dataset_relpath, datatype, suffix, des,
                            recording, runs_min, runs_max
                   ORDER BY dataset_relpath, datatype, suffix""",
                [task],
            )
            result["task"] = task
            if units:
                result["expected_units"] = units
            else:
                known = [r["task"] for r in _fetch_dicts(
                    conn,
                    """SELECT DISTINCT task FROM expected_units
                       WHERE task IS NOT NULL ORDER BY task""",
                    [],
                )]
                result["error"] = (
                    f"Task {task!r} has no declared expectations. "
                    f"Declared tasks: {known}"
                )

        if session_type:
            sessions = _fetch_dicts(
                conn,
                """SELECT ses, list_sort(list(DISTINCT sub)) AS subjects
                   FROM sessions WHERE session_type = ?
                   GROUP BY ses ORDER BY ses""",
                [session_type],
            )
            units = _fetch_dicts(
                conn,
                """SELECT eu.dataset_relpath, eu.datatype, eu.task,
                          eu.suffix, eu.recording, eu.runs_min, eu.runs_max,
                          COUNT(*) AS n_units
                   FROM expected_units eu
                   JOIN (SELECT DISTINCT sub, ses FROM sessions
                         WHERE session_type = ?) s
                     ON eu.sub = s.sub AND eu.ses = s.ses
                   GROUP BY eu.dataset_relpath, eu.datatype, eu.task,
                            eu.suffix, eu.recording, eu.runs_min, eu.runs_max
                   ORDER BY eu.dataset_relpath, eu.datatype, eu.task""",
                [session_type],
            )
            result["session_type"] = session_type
            result["sessions"] = sessions
            result["expected_units"] = units
            if not sessions:
                known = [r["session_type"] for r in _fetch_dicts(
                    conn,
                    """SELECT DISTINCT session_type FROM sessions
                       WHERE session_type IS NOT NULL ORDER BY 1""",
                    [],
                )]
                result["error"] = (
                    f"Session type {session_type!r} not found. "
                    f"Known types: {known}"
                )

        if not task and not session_type:
            result["available_tasks"] = [r["task"] for r in _fetch_dicts(
                conn,
                """SELECT DISTINCT task FROM expected_units
                   WHERE task IS NOT NULL ORDER BY task""",
                [],
            )]
            result["available_session_types"] = [
                r["session_type"] for r in _fetch_dicts(
                    conn,
                    """SELECT DISTINCT session_type FROM sessions
                       WHERE session_type IS NOT NULL ORDER BY 1""",
                    [],
                )
            ]
            result["subjects"] = [r["sub"] for r in _fetch_dicts(
                conn,
                "SELECT DISTINCT sub FROM expected_units ORDER BY sub",
                [],
            )]
            result["datasets_with_expectations"] = [
                r["dataset_relpath"] for r in _fetch_dicts(
                    conn,
                    """SELECT DISTINCT dataset_relpath FROM expected_units
                       ORDER BY 1""",
                    [],
                )
            ]
            result["n_expected_units"] = _fetch_dicts(
                conn, "SELECT COUNT(*) AS n FROM expected_units", []
            )[0]["n"]

        return result
    finally:
        conn.close()
