"""Read-only query helpers for the MMMData manifest database.

The manifest database (built by ``scripts/build_manifest.py``) is a
SQLite inventory of the BIDS dataset with tables: files, nifti_meta,
events_meta, physio_meta, sourcedata, derivatives, session_metadata,
and validation_results.

Everything in this module opens the database in read-only mode and
returns plain Python data structures (lists of dicts), so it is safe
to call from reports, notebooks, and agent tools alike.

Usage:
    from core import manifest_db

    cols, rows = manifest_db.run_select(
        db_path, "SELECT COUNT(*) AS n FROM files")
    summary = manifest_db.session_summary(db_path, "sub-03", "ses-04")
"""

import sqlite3
from pathlib import Path


def connect_readonly(db_path: Path | str) -> sqlite3.Connection:
    """Open a read-only connection to the manifest database.

    Uses SQLite URI mode (``mode=ro``) so writes are rejected at the
    connection level. Rows are returned as ``sqlite3.Row`` objects.

    Raises FileNotFoundError if the database does not exist.
    """
    db = Path(db_path)
    if not db.exists():
        raise FileNotFoundError(
            f"Manifest database not found: {db}. "
            "Run scripts/build_manifest.py in the mmmdata repo to create it."
        )
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def run_select(
    db_path: Path | str,
    sql: str,
    params: list | tuple | None = None,
) -> tuple[list[str], list[dict]]:
    """Run a single SELECT query against the manifest database.

    Only SELECT statements are allowed; anything else raises
    ValueError before the database is even opened. The connection is
    additionally opened read-only, so writes cannot slip through.

    Returns (column_names, rows) where rows is a list of dicts keyed
    by column name.
    """
    sql_stripped = sql.strip().lstrip("(").strip()
    if not sql_stripped.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")

    conn = connect_readonly(db_path)
    try:
        cur = conn.execute(sql, params or [])
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]
        return columns, rows
    finally:
        conn.close()


def session_summary(
    db_path: Path | str,
    subject: str,
    session: str,
) -> dict:
    """Build a comprehensive summary of one subject/session.

    Gathers, from the manifest database: all files with BIDS entities,
    NIfTI metadata (dimensions, volumes, TR), events file stats (row
    counts, onset range), physio channels, derivative coverage, and
    session metadata (date, notes, physiological measures).

    Args:
        db_path: Path to manifest.db.
        subject: Subject ID including prefix (e.g. "sub-03").
        session: Session ID including prefix (e.g. "ses-01").

    Returns a dict with keys: subject, session, n_files, files, nifti,
    events, physio, derivatives, session_metadata.
    """
    conn = connect_readonly(db_path)
    try:
        # Files
        files = [dict(r) for r in conn.execute(
            """SELECT path, datatype, task, run, suffix, format, size_bytes
               FROM files WHERE subject=? AND session=?
               ORDER BY datatype, task, run, suffix""",
            (subject, session)
        ).fetchall()]

        # NIfTI metadata
        nifti = [dict(r) for r in conn.execute(
            """SELECT f.task, f.run, f.suffix, n.nx, n.ny, n.nz, n.nt,
                      n.voxel_x, n.voxel_y, n.voxel_z, n.tr, n.acq_duration
               FROM nifti_meta n JOIN files f ON n.path = f.path
               WHERE f.subject=? AND f.session=?
               ORDER BY f.task, f.run""",
            (subject, session)
        ).fetchall()]

        # Events
        events = [dict(r) for r in conn.execute(
            """SELECT f.task, f.run, em.n_rows, em.columns,
                      em.onset_min, em.onset_max
               FROM events_meta em JOIN files f ON em.path = f.path
               WHERE f.subject=? AND f.session=?
               ORDER BY f.task, f.run""",
            (subject, session)
        ).fetchall()]

        # Physio
        physio = [dict(r) for r in conn.execute(
            """SELECT pm.recording, pm.sampling_rate, pm.n_columns
               FROM physio_meta pm JOIN files f ON pm.path = f.path
               WHERE f.subject=? AND f.session=?""",
            (subject, session)
        ).fetchall()]

        # Derivatives
        derivs = [dict(r) for r in conn.execute(
            """SELECT pipeline, source_suffix, description, COUNT(*) as n_files
               FROM derivatives
               WHERE subject=? AND session=?
               GROUP BY pipeline, source_suffix, description
               ORDER BY pipeline, source_suffix""",
            (subject, session)
        ).fetchall()]

        # Session metadata
        meta_row = conn.execute(
            "SELECT * FROM session_metadata WHERE subject=? AND session=?",
            (subject, session)
        ).fetchone()
        meta = dict(meta_row) if meta_row else None

        return {
            "subject": subject,
            "session": session,
            "n_files": len(files),
            "files": files,
            "nifti": nifti,
            "events": events,
            "physio": physio,
            "derivatives": derivs,
            "session_metadata": meta,
        }
    finally:
        conn.close()


def validation_report(
    db_path: Path | str,
    subject: str | None = None,
    status_filter: str | None = None,
) -> dict:
    """Read stored validation results from the validation_results table.

    By default returns only issues (everything except 'pass'), sorted
    by severity (fail > warn > info). Pass status_filter='pass' to see
    passing checks instead, or any other status to filter to it.

    Args:
        db_path: Path to manifest.db.
        subject: Optionally restrict to one subject (e.g. "sub-03").
        status_filter: Optionally restrict to one status
            ("pass", "fail", "warn", "info").

    Returns a dict with keys: summary (status -> count, unfiltered by
    status_filter), n_results, results (list of result dicts).
    """
    conn = connect_readonly(db_path)
    try:
        where = []
        params = []

        if subject:
            where.append("subject = ?")
            params.append(subject)

        if status_filter:
            where.append("status = ?")
            params.append(status_filter)
        else:
            where.append("status != 'pass'")

        where_clause = " AND ".join(where) if where else "1"

        rows = [dict(r) for r in conn.execute(
            f"""SELECT check_name, subject, session, task, run,
                       status, expected, actual, message, checked_at
                FROM validation_results
                WHERE {where_clause}
                ORDER BY
                  CASE status WHEN 'fail' THEN 0 WHEN 'warn' THEN 1
                              WHEN 'info' THEN 2 ELSE 3 END,
                  check_name, subject, session""",
            params
        ).fetchall()]

        # Summary counts (unfiltered by status_filter to give context)
        sub_filter = "WHERE subject = ?" if subject else ""
        sub_params = [subject] if subject else []
        counts = dict(conn.execute(
            f"""SELECT status, COUNT(*) FROM validation_results
                {sub_filter} GROUP BY status""",
            sub_params
        ).fetchall())

        return {
            "summary": counts,
            "n_results": len(rows),
            "results": rows,
        }
    finally:
        conn.close()
