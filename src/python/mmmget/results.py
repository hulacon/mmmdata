"""write_result(): a results table + its Vega-Lite spec, mmmview-valid, with provenance."""

from __future__ import annotations

import copy
import datetime
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from resultsview import page

from .provenance import provenance_records
from ._version import __version__


def _frame(table: pd.DataFrame | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    df = table if isinstance(table, pd.DataFrame) else pd.DataFrame(list(table))
    if df.empty or not len(df.columns):
        raise page.SpecError("a results table needs at least one column and one row")
    bad_cols = [c for c in df.columns if any(ch in str(c) for ch in "\t\n\r")]
    text = df.select_dtypes(include=["object", "string"])
    bad_cells = text.apply(lambda s: s.astype(str).str.contains(r"[\t\n\r]", regex=True)).any()
    bad = bad_cols + [c for c, b in bad_cells.items() if b]
    if bad:
        raise page.SpecError(
            f"tab or newline inside column(s) {bad}; a TSV cell cannot hold one")
    return df


def provenance_block() -> dict[str, Any]:
    """What the getters recorded since the last reset, summarised.

    ``standard`` is True only when at least one getter fed the result and
    every one of them stayed on the standard path; ``non_standard`` lists
    the reasons when not.
    """
    records = provenance_records()
    reasons = [f"{r['getter']}: {r['reason']}" for r in records if not r["standard"]]
    if not records:
        reasons.append("no mmmget getter was called, so nothing vouches for the inputs")
    return {
        "mmmget_version": __version__,
        "standard": not reasons,
        "non_standard": reasons,
        "getters": records,
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def write_result(
    table: pd.DataFrame | Iterable[Mapping[str, Any]],
    spec: Mapping[str, Any],
    out_dir: Path | str,
    stem: str,
) -> tuple[Path, Path]:
    """Write ``<stem>.tsv`` + ``<stem>.vl.json``, valid under mmmview's page contract.

    *spec* is a Vega-Lite spec with no data of its own and a
    ``usermeta.mmmview`` block naming ``err_over`` (``resultsview/page.py``
    has the contract; ``schema_version`` is filled in when absent). The
    getters' provenance is stamped into ``usermeta.mmmview.provenance``:
    the mmmget version, every getter call recorded since
    :func:`~mmmget.reset_provenance`, and whether all of them were standard.

    Both files are validated with mmmview's own loader before either lands
    in *out_dir*, so a contract violation raises
    :class:`resultsview.page.SpecError` and leaves *out_dir* untouched.

    Returns
    -------
    (tsv_path, spec_path)
    """
    if not stem or "/" in stem or stem.endswith((page.TABLE_EXT, page.SPEC_EXT)):
        raise ValueError(f"stem {stem!r} must be a bare name without an extension")
    df = _frame(table)
    spec = copy.deepcopy(dict(spec))
    meta = spec.setdefault("usermeta", {}).setdefault("mmmview", {})
    if not isinstance(meta, dict):
        raise page.SpecError("usermeta.mmmview must be an object")
    meta.setdefault("schema_version", page.SCHEMA_VERSION)
    table_name = stem + page.TABLE_EXT
    if meta.get("table", table_name) != table_name:
        raise page.SpecError(
            f"usermeta.mmmview.table is {meta['table']!r}, but write_result "
            f"writes {table_name!r}; drop the key or match the stem")
    meta["provenance"] = provenance_block()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv, vl = out_dir / table_name, out_dir / (stem + page.SPEC_EXT)
    with tempfile.TemporaryDirectory(dir=out_dir, prefix=".mmmget-") as tmp:
        t_tsv, t_vl = Path(tmp) / tsv.name, Path(tmp) / vl.name
        df.to_csv(t_tsv, sep="\t", index=False, na_rep="n/a")
        t_vl.write_text(json.dumps(spec, indent=2, allow_nan=False) + "\n")
        page.load_spec(t_vl)  # raises SpecError naming the fix
        os.replace(t_tsv, tsv)
        os.replace(t_vl, vl)
    return tsv, vl
