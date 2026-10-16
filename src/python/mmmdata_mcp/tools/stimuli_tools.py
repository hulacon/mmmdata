"""Stimulus feature query tools for the MMMData MCP server.

Read-only access to the Contract B feature store at
``derivatives/stimuli_features/`` — the output of the feature extraction
campaign (``scripts/stimfeat_campaign.py``), covering all three MMMData stimulus sets (1,000 NSD images,
1,000 spoken words, 60 film clips) across the viz2psy / aud2psy /
word2psy model suite.

Queries go against the psytwill aggregate tables
(``derivatives/stimuli_features/psytwill/*.parquet``), one per
(stimulus set × source × granularity) **group**. Every group shares one
long-format schema — a row is one (stimulus, coordinate, model, feature)
observation:

    stimulus_id, voice, time, onset, offset, chunk_idx, word_idx,
    modality, extractor, extractor_version, model, feature,
    value, value_str

These tools pivot that back to a wide row per stimulus/coordinate, which
is what a caller comparing features across stimuli wants.

Superseded the pre-0.6.0 wide CSVs under ``stimuli/*/viz2psy_scores*``.
Those files used the column convention viz2psy 0.6.0 renamed, covered
only viz2psy, and reached neither twp1000 nor any audio, transcript,
caption, or human-annotation feature.
"""

import json
import re
from pathlib import Path
from typing import Annotated

from pydantic import Field

from ..paths import get_paths

#: Contract B embedding columns: a model prefix plus fixed-width
#: index suffixes (``clip_000``, ``ebind_1023``, the ``saliency_23_23``
#: grid). Excluded from wide output unless asked for by name — a single
#: EBind row is 1,024 columns. Named scores (``places_airfield``,
#: ``emonet_surprise``, ``yolo_zebra``, ``llstat_b_mean``) do not match.
_EMBEDDING_RE = re.compile(r"_\d+(_\d+)*$")

#: Long-table columns that locate a row rather than measure it. Kept in
#: wide output when a group populates them; dropped when all-null, so a
#: per-image group does not carry six empty timing columns.
_KEY_COLUMNS = (
    "stimulus_id", "voice", "time", "onset", "offset", "chunk_idx", "word_idx",
)

_OPS = {">": ">", "<": "<", ">=": ">=", "<=": "<=", "==": "="}

#: Key columns as SQL identifiers. Quoted because `offset` is a DuckDB
#: keyword and an unquoted ORDER BY on it is a parser error.
_KEY_SQL = ", ".join(f'"{c}"' for c in _KEY_COLUMNS)

#: Movie filename suffix the stimulus ids do not carry.
_MOVIE_SUFFIX = "_trimmed_normalized_filtered"


# ---------------------------------------------------------------------------
# Store access
# ---------------------------------------------------------------------------

def _tables_dir() -> Path:
    return get_paths().stimuli_features_tables_dir


def _group_path(group: str) -> Path:
    """Parquet backing one group id (the filename minus ``_features``)."""
    return _tables_dir() / f"{group}_features.parquet"


def _available_groups() -> list[str]:
    d = _tables_dir()
    if not d.exists():
        return []
    return sorted(p.name.removesuffix("_features.parquet")
                  for p in d.glob("*_features.parquet"))


def _resolve_group(group: str | None, default: str) -> tuple[str, dict | None]:
    """(group, error) — validate a group id against what the store holds."""
    group = group or default
    if _group_path(group).exists():
        return group, None
    groups = _available_groups()
    if not groups:
        return group, {
            "error": "Contract B feature store not found",
            "expected_path": str(_tables_dir()),
            "hint": ("Build the aggregates: "
                     "stimfeat_campaign.py aggregate (mmmdata/scripts)."),
        }
    return group, {
        "error": f"Unknown feature group: {group!r}",
        "available_groups": groups,
    }


def _connect():
    import duckdb
    con = duckdb.connect()
    # The progress bar writes to stdout, which for an MCP tool is noise
    # interleaved with nothing anyone reads.
    con.execute("SET enable_progress_bar = false")
    return con


def _quote(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _is_embedding(feature: str) -> bool:
    return bool(_EMBEDDING_RE.search(feature))


def _movie_stem(name: str) -> str:
    """'Adventure_Time_trimmed_normalized_filtered.mov' -> 'Adventure_Time'."""
    stem = Path(name).stem
    return stem.removesuffix(_MOVIE_SUFFIX)


# ---------------------------------------------------------------------------
# Long -> wide
# ---------------------------------------------------------------------------

def _wide_query(
    path: Path,
    where: list[str],
    params: list,
    limit: int,
) -> str:
    """SQL returning the long rows of the first `limit` distinct stimuli/rows.

    Ranking distinct key tuples before the limit is what keeps a row whole:
    limiting the long rows directly would truncate one stimulus mid-feature
    and silently return a partial observation.
    """
    clause = " AND ".join(where) if where else "TRUE"
    return f"""
        WITH filtered AS (
            SELECT * FROM read_parquet({_quote(path)}) WHERE {clause}
        ), ranked AS (
            SELECT *, DENSE_RANK() OVER (ORDER BY {_KEY_SQL}) AS _row
            FROM filtered
        )
        SELECT * FROM ranked WHERE _row <= {int(limit)}
    """


def _pivot(df) -> tuple[list[str], list[dict]]:
    """Long rows -> (columns, wide records), dropping all-null key columns."""
    if df.empty:
        return [], []
    keys = [c for c in _KEY_COLUMNS if df[c].notna().any()]
    df = df.copy()
    # Not `_value`: itertuples renames leading-underscore columns positionally.
    df["feature_value"] = df["value"].where(df["value"].notna(), df["value_str"])

    records: dict[tuple, dict] = {}
    order: list[tuple] = []
    for row in df.itertuples(index=False):
        ident = tuple(getattr(row, k) for k in keys)
        rec = records.get(ident)
        if rec is None:
            rec = {k: getattr(row, k) for k in keys}
            records[ident] = rec
            order.append(ident)
        rec[row.feature] = row.feature_value

    out = [records[i] for i in order]
    # Feature columns sorted, so the same query returns the same column
    # order every time; parquet row order within a stimulus is not stable
    # enough to read as an ordering.
    features = sorted({c for rec in out for c in rec} - set(keys))
    return list(keys) + features, out


def _feature_filters(
    models: list[str] | None,
    features: list[str] | None,
) -> tuple[list[str], list]:
    """WHERE fragments for a model/feature request.

    With neither, embeddings are excluded — the default answer is the
    interpretable scores. Naming a feature or an embedding model opts in.
    """
    where: list[str] = []
    params: list = []
    if models:
        where.append(f"model IN ({', '.join('?' for _ in models)})")
        params.extend(models)
    if features:
        where.append(f"feature IN ({', '.join('?' for _ in features)})")
        params.extend(features)
    elif not models:
        where.append(r"NOT regexp_matches(feature, '_\d+(_\d+)*$')")
    return where, params


def _run_wide(
    group: str,
    where: list[str],
    params: list,
    limit: int,
    extra: dict | None = None,
) -> str:
    path = _group_path(group)
    con = _connect()
    try:
        df = con.execute(_wide_query(path, where, params, limit), params).df()
    finally:
        con.close()
    columns, records = _pivot(df)
    body = {"group": group, "rows": len(records), "columns": columns,
            "data": records}
    if extra:
        body.update(extra)
    return json.dumps(body, indent=2, default=str)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def list_stimulus_sets() -> str:
    """List the three MMMData stimulus sets with their raw file counts, and
    every feature group in the Contract B store with its row count, stimulus
    count, and models. Group ids are what the other tools' `group` parameter
    takes.
    """
    paths = get_paths()
    stim = paths.stimuli_dir
    tables = _tables_dir()

    result: dict = {
        "feature_store": str(paths.stimuli_features_dir),
        "sets": {
            "shared1000": {
                "description": "1,000 NSD images (trial-based encoding)",
                "path": str(stim / "shared1000"),
                "n_files": len(list((stim / "shared1000" / "images").glob("*.png")))
                if (stim / "shared1000" / "images").exists() else 0,
            },
            "twp1000": {
                # One subdirectory per synthesized voice; the word->voice
                # pairing is frozen across subjects, so only 1,000 of the
                # 4,000 files were ever presented.
                "description": "1,000 spoken words x 4 voices (word audio)",
                "path": str(stim / "twp1000"),
                "voices": sorted(d.name for d in (stim / "twp1000").iterdir()
                                 if d.is_dir()) if (stim / "twp1000").exists() else [],
                "n_files": len(list((stim / "twp1000").glob("*/*.mp3"))),
            },
            "movies": {
                "description": "60 short films (naturalistic encoding) + cue images",
                "path": str(stim / "movies"),
                "n_files": len(list((stim / "movies" / "movie_files").glob("*.mov")))
                if (stim / "movies" / "movie_files").exists() else 0,
                "n_cues": len(list((stim / "movies" / "movie_cues").glob("*.jpg")))
                if (stim / "movies" / "movie_cues").exists() else 0,
            },
        },
        "groups": {},
    }

    if not tables.exists():
        result["error"] = "Contract B feature store not found"
        result["expected_path"] = str(tables)
        return json.dumps(result, indent=2)

    con = _connect()
    try:
        for group in _available_groups():
            path = _group_path(group)
            row = con.execute(
                f"""SELECT COUNT(*), COUNT(DISTINCT stimulus_id),
                           COUNT(DISTINCT model)
                    FROM read_parquet({_quote(path)})"""
            ).fetchone()
            models = [r[0] for r in con.execute(
                f"SELECT DISTINCT model FROM read_parquet({_quote(path)}) "
                f"ORDER BY model"
            ).fetchall()]
            result["groups"][group] = {
                "rows": row[0],
                "n_stimuli": row[1],
                "models": models,
            }
    finally:
        con.close()
    return json.dumps(result, indent=2, default=str)


def get_image_features(
    stimulus_ids: Annotated[
        list[str] | None,
        Field(description=(
            "Filter by stimulus_id, e.g. 'shared0001_nsd02951'. "
            "Default: all images."
        )),
    ] = None,
    nsd_ids: Annotated[
        list[int] | None,
        Field(description="Filter by NSD image IDs (0-based)."),
    ] = None,
    group: Annotated[
        str,
        Field(description=(
            "Feature group. 'shared1000_image' is the viz2psy image suite; "
            "'shared1000_caption_chunks' and 'shared1000_humancap_chunks' are "
            "the machine and human caption text features. See "
            "list_stimulus_sets for all groups."
        )),
    ] = "shared1000_image",
    models: Annotated[
        list[str] | None,
        Field(description=(
            "Restrict to these models, e.g. ['resmem', 'emonet']. Naming an "
            "embedding model ('clip', 'dinov2', 'ebind') opts into its columns."
        )),
    ] = None,
    features: Annotated[
        list[str] | None,
        Field(description=(
            "Specific feature columns. Default: all non-embedding features."
        )),
    ] = None,
    limit: Annotated[
        int,
        Field(description="Maximum stimuli to return (default 20)."),
    ] = 20,
) -> str:
    """Query features for the shared1000 images. Returns one wide row per
    image, carrying the interpretable scores (memorability, emotion, scene
    categories, objects, low-level statistics, caption text) by default and
    excluding embeddings. Request an embedding model or named features to
    include them.
    """
    group, err = _resolve_group(group, "shared1000_image")
    if err:
        return json.dumps(err, indent=2)

    where, params = _feature_filters(models, features)
    if stimulus_ids:
        where.append(f"stimulus_id IN ({', '.join('?' for _ in stimulus_ids)})")
        params.extend(stimulus_ids)
    if nsd_ids:
        # Filenames carry a 1-based nsdId; the parameter is 0-based.
        tags = [f"%nsd{n + 1:05d}%" for n in nsd_ids]
        where.append("(" + " OR ".join("stimulus_id LIKE ?" for _ in tags) + ")")
        params.extend(tags)
    return _run_wide(group, where, params, limit)


def get_movie_features(
    movie_name: Annotated[
        str,
        Field(description=(
            "Movie stimulus_id, e.g. 'adventure-time'. Matched "
            "case-insensitively; the trimmed/normalized filename suffix is "
            "not part of the id."
        )),
    ],
    group: Annotated[
        str,
        Field(description=(
            "Feature group. 'movies_frames' is the 0.5 s visual grid; "
            "'movies_audio_frames' the audio grid; 'movies_audio_transcript' "
            "the speech transcript; 'movies_annot_chunks' the human scene "
            "annotations. See list_stimulus_sets."
        )),
    ] = "movies_frames",
    time_start: Annotated[
        float | None,
        Field(description="Start time in seconds (inclusive)."),
    ] = None,
    time_end: Annotated[
        float | None,
        Field(description="End time in seconds (inclusive)."),
    ] = None,
    models: Annotated[
        list[str] | None,
        Field(description="Restrict to these models."),
    ] = None,
    features: Annotated[
        list[str] | None,
        Field(description="Specific feature columns."),
    ] = None,
    limit: Annotated[
        int,
        Field(description="Maximum rows (time points / chunks) to return."),
    ] = 50,
) -> str:
    """Query within-movie features over time. Returns one wide row per
    coordinate — a 0.5 s window for the frame grids, a segment for the
    transcript and annotation groups. Use time_start/time_end to focus on a
    stretch of a film.
    """
    group, err = _resolve_group(group, "movies_frames")
    if err:
        return json.dumps(err, indent=2)

    path = _group_path(group)
    con = _connect()
    try:
        match = con.execute(
            f"""SELECT stimulus_id FROM read_parquet({_quote(path)})
                WHERE lower(stimulus_id) = lower(?) LIMIT 1""",
            [_movie_stem(movie_name)],
        ).fetchone()
        if match is None:
            available = [r[0] for r in con.execute(
                f"SELECT DISTINCT stimulus_id FROM read_parquet({_quote(path)}) "
                f"ORDER BY stimulus_id"
            ).fetchall()]
            return json.dumps({
                "error": f"No features for movie {movie_name!r} in {group!r}",
                "available_stimuli": available,
            }, indent=2)
        resolved = match[0]
    finally:
        con.close()

    where, params = _feature_filters(models, features)
    where.append("stimulus_id = ?")
    params.append(resolved)
    if time_start is not None:
        where.append("time >= ?")
        params.append(time_start)
    if time_end is not None:
        where.append("time <= ?")
        params.append(time_end)
    return _run_wide(group, where, params, limit, extra={"movie": resolved})


def get_movie_summary(
    movie_names: Annotated[
        list[str] | None,
        Field(description="Filter by movie stimulus_id. Default: all."),
    ] = None,
    group: Annotated[
        str,
        Field(description=(
            "Feature group to summarize over. Default 'movies_frames' "
            "(the 0.5 s visual grid)."
        )),
    ] = "movies_frames",
    models: Annotated[
        list[str] | None,
        Field(description="Restrict to these models."),
    ] = None,
    features: Annotated[
        list[str] | None,
        Field(description="Specific feature columns."),
    ] = None,
    stats: Annotated[
        list[str] | None,
        Field(
            description="Statistics to compute. Default ['mean', 'sd'].",
            json_schema_extra={"items": {"enum": ["mean", "sd", "min", "max"]}},
        ),
    ] = None,
) -> str:
    """Aggregate a movie feature group to one row per film — the per-movie
    view for comparing films to each other, as opposed to get_movie_features
    for temporal structure within one. Computed on the fly from the group's
    rows, so it follows whatever the store currently holds.
    """
    group, err = _resolve_group(group, "movies_frames")
    if err:
        return json.dumps(err, indent=2)

    stats = stats or ["mean", "sd"]
    agg = {"mean": "AVG(value)", "sd": "STDDEV_SAMP(value)",
           "min": "MIN(value)", "max": "MAX(value)"}
    unknown = [s for s in stats if s not in agg]
    if unknown:
        return json.dumps({"error": f"Unknown statistic(s): {unknown}",
                           "available": list(agg)}, indent=2)

    where, params = _feature_filters(models, features)
    # String features have no mean; excluded rather than silently nulled.
    where.append("value IS NOT NULL")
    if movie_names:
        placeholders = ", ".join("?" for _ in movie_names)
        where.append(f"lower(stimulus_id) IN ({placeholders})")
        params.extend(m.lower() for m in movie_names)

    selects = ", ".join(f"{agg[s]} AS {s}" for s in stats)
    sql = f"""
        SELECT stimulus_id, feature, {selects}
        FROM read_parquet({_quote(_group_path(group))})
        WHERE {' AND '.join(where)}
        GROUP BY stimulus_id, feature
        ORDER BY stimulus_id, feature
    """
    con = _connect()
    try:
        df = con.execute(sql, params).df()
    finally:
        con.close()

    records: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        rec = records.setdefault(row.stimulus_id, {"stimulus_id": row.stimulus_id})
        for stat in stats:
            rec[f"{row.feature}_{stat}"] = getattr(row, stat)
    data = list(records.values())
    columns = list(data[0]) if data else []
    return json.dumps({"group": group, "stats": stats, "rows": len(data),
                       "columns": columns, "data": data},
                      indent=2, default=str)


def search_stimuli(
    feature: Annotated[
        str,
        Field(description=(
            "Feature column name, e.g. 'resmem_memorability', "
            "'aesthetics_score', 'emonet_awe'. Names carry their model prefix "
            "(Contract B); describe_features lists them."
        )),
    ],
    op: Annotated[
        str,
        Field(
            description="Comparison operator.",
            json_schema_extra={"enum": [">", "<", ">=", "<=", "=="]},
        ),
    ],
    value: Annotated[
        float,
        Field(description="Threshold value for comparison."),
    ],
    group: Annotated[
        str,
        Field(description=(
            "Feature group to search. Default 'shared1000_image'. Searching "
            "a per-time group like 'movies_frames' matches individual "
            "windows; use get_movie_summary for whole-film comparisons."
        )),
    ] = "shared1000_image",
    limit: Annotated[
        int,
        Field(description="Maximum matches to return (default 20)."),
    ] = 20,
) -> str:
    """Find stimuli whose feature crosses a threshold — the most memorable
    images, the most aesthetic movie frames, the highest-arousal words.
    Returns matches sorted by the feature, strongest first for > and >=.
    """
    if op not in _OPS:
        return json.dumps({"error": f"Invalid operator: {op!r}",
                           "available": list(_OPS)}, indent=2)
    group, err = _resolve_group(group, "shared1000_image")
    if err:
        return json.dumps(err, indent=2)

    path = _group_path(group)
    descending = op in (">", ">=")
    sql = f"""
        SELECT {_KEY_SQL}, model, value, value_str
        FROM read_parquet({_quote(path)})
        WHERE feature = ? AND value {_OPS[op]} ?
        ORDER BY value {'DESC' if descending else 'ASC'}
        LIMIT {int(limit)}
    """
    con = _connect()
    try:
        df = con.execute(sql, [feature, value]).df()
        if df.empty:
            known = [r[0] for r in con.execute(
                f"""SELECT DISTINCT feature FROM read_parquet({_quote(path)})
                    WHERE feature = ? LIMIT 1""", [feature]).fetchall()]
            if not known:
                sample = [r[0] for r in con.execute(
                    f"""SELECT DISTINCT feature FROM read_parquet({_quote(path)})
                        WHERE NOT regexp_matches(feature, '_\\d+(_\\d+)*$')
                        ORDER BY feature LIMIT 60""").fetchall()]
                return json.dumps({
                    "error": f"Feature {feature!r} not in group {group!r}",
                    "available_features_sample": sample,
                }, indent=2)
    finally:
        con.close()

    if df.empty:
        # The feature exists (an unknown one returned above) — the threshold
        # simply matched nothing. Said explicitly, because "0 matches" and
        # "no such feature" are different answers and read the same.
        return json.dumps({
            "group": group,
            "query": f"{feature} {op} {value}",
            "matches": 0,
            "columns": [],
            "data": [],
            "note": (f"{feature!r} exists in this group but no stimulus "
                     f"satisfies the threshold."),
        }, indent=2)

    keep = [c for c in _KEY_COLUMNS if df[c].notna().any()] + ["model", "value"]
    return json.dumps({
        "group": group,
        "query": f"{feature} {op} {value}",
        "matches": len(df),
        "columns": keep,
        "data": df[keep].to_dict(orient="records"),
    }, indent=2, default=str)


def describe_features(
    group: Annotated[
        str,
        Field(description=(
            "Feature group to describe. Default 'shared1000_image'. See "
            "list_stimulus_sets for the full list."
        )),
    ] = "shared1000_image",
) -> str:
    """Describe what a feature group contains: which models contributed,
    their checkpoints and versions, how many features each carries, and the
    named (non-embedding) feature columns. Reads the group's Contract B
    sidecar plus the table itself.
    """
    group, err = _resolve_group(group, "shared1000_image")
    if err:
        return json.dumps(err, indent=2)

    path = _group_path(group)
    con = _connect()
    try:
        rows = con.execute(
            f"""SELECT model, extractor, ANY_VALUE(extractor_version),
                       ANY_VALUE(modality), COUNT(DISTINCT feature)
                FROM read_parquet({_quote(path)})
                GROUP BY model, extractor ORDER BY model"""
        ).fetchall()
        named = {}
        for model, *_ in rows:
            cols = [r[0] for r in con.execute(
                f"""SELECT DISTINCT feature FROM read_parquet({_quote(path)})
                    WHERE model = ? ORDER BY feature""", [model]).fetchall()]
            named[model] = [c for c in cols if not _is_embedding(c)]
    finally:
        con.close()

    # Checkpoints live in the psytwill sidecar's per-input provenance, one
    # entry per extractor CSV that fed the aggregate.
    checkpoints: dict[str, set] = {}
    sidecar_path = path.with_suffix(".meta.json")
    sidecar = {}
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text())
        for entry in sidecar.get("inputs", []):
            for model, info in (entry.get("models") or {}).items():
                ckpt = (info or {}).get("checkpoint")
                if ckpt:
                    checkpoints.setdefault(model, set()).add(ckpt)

    models = {}
    for model, extractor, version, modality, n_features in rows:
        entry = {
            "extractor": extractor,
            "extractor_version": version,
            "modality": modality,
            "n_features": n_features,
        }
        if model in checkpoints:
            ckpts = sorted(checkpoints[model])
            entry["checkpoint"] = ckpts[0] if len(ckpts) == 1 else ckpts
        if named[model]:
            entry["features"] = named[model]
        else:
            entry["output_type"] = "embedding"
        models[model] = entry

    return json.dumps({
        "group": group,
        "table": str(path),
        "schema_version": sidecar.get("schema_version"),
        "aggregator": sidecar.get("extractor"),
        "aggregator_version": sidecar.get("extractor_version"),
        "rows": sidecar.get("output", {}).get("rows"),
        "key_columns": sidecar.get("output", {}).get("key_columns"),
        "models": models,
    }, indent=2, default=str)


TOOL_FUNCTIONS = [
    list_stimulus_sets,
    get_image_features,
    get_movie_features,
    get_movie_summary,
    search_stimuli,
    describe_features,
]
