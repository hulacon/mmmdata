"""Shared paths, names and readers for the nat_encret derivative.

``derivatives/nat_encret/`` holds NAT encoding->retrieval coding: which parts
of each spoken film recall express which film events. It is organised by data
type, not by method: the matcher is a ``desc-`` value (``desc-rmatch``; a
human coder would be ``desc-human``), not a folder.

Layout (one pair per harmonised ``task-NATretrieval`` word table in ``beh/``)::

    nat_encret/
      dataset_description.json
      task-NATretrieval_recallunits.json            column descriptions
      task-NATretrieval_desc-<m>_recallmap.json     column descriptions
      sub-##/ses-##/beh/
        sub-##_ses-##_task-NATretrieval[_run-##]_recallunits.tsv
        sub-##_ses-##_task-NATretrieval[_run-##]_desc-<m>_recallmap.tsv
        sub-##_ses-##_task-NATretrieval[_run-##]_desc-<m>_recallmap.json   run provenance
      group/
        desc-<m>_eventrecall.tsv   participant x SEG-C event, first recalls
        desc-<m>_eventnorms.tsv    SEG-C event, first recalls pooled over participants

No file here holds transcript text: a unit is a row range into its ``beh.tsv``
(``behavioral.recall_units``).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src" / "python", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _ru():
    # mmmdata packages are imported on use: the behavioral and core package __init__s pull in
    # pybids, which the GPU env that runs rmatch_run.py vgenerate does not carry (and vgenerate
    # needs neither the units code nor the config).
    from behavioral import recall_units
    return recall_units


TASK = "NATretrieval"
BEH_GLOB = f"sub-*/ses-*/beh/sub-*_task-{TASK}*_beh.tsv"
UNIT_KEY = ["subject", "session", "run", "trial_num", "unit_idx"]
DECIMALS = 3  # the word tables carry millisecond times


def paths() -> dict:
    from core.config import load_config
    cfg = load_config(config_dir=REPO / "config")["paths"]
    return dict(bids=Path(cfg["bids_project_dir"]), derivatives=Path(cfg["output_dir"]))


def default_out() -> Path:
    return paths()["derivatives"] / "nat_encret"


def entities(name: str) -> dict:
    """BIDS entities of a file name (``sub-##_ses-##_..._suffix.ext``)."""
    return dict(p.split("-", 1) for p in name.split("_")[:-1] if "-" in p)


def stem(ent: dict) -> str:
    run = f"_run-{ent['run']}" if ent.get("run") else ""
    return f"sub-{ent['sub']}_ses-{ent['ses']}_task-{TASK}{run}"


def beh_dir(root: Path, ent: dict) -> Path:
    return root / f"sub-{ent['sub']}" / f"ses-{ent['ses']}" / "beh"


def units_path(out: Path, ent: dict) -> Path:
    return beh_dir(out, ent) / f"{stem(ent)}_recallunits.tsv"


def map_path(out: Path, ent: dict, matcher: str) -> Path:
    return beh_dir(out, ent) / f"{stem(ent)}_desc-{matcher}_recallmap.tsv"


def beh_path(bids: Path, ent: dict) -> Path:
    return beh_dir(bids, ent) / f"{stem(ent)}_beh.tsv"


def name_to_id(bids: Path) -> dict:
    """movie_name (and each listed variant) -> registry stimulus_id."""
    reg = pd.read_csv(bids / "stimuli" / "stimulus_registry" / "movies.tsv", sep="\t",
                      dtype=str, keep_default_na=False)
    out = dict(zip(reg.movie_name, reg.stimulus_id))
    for _, r in reg.iterrows():
        for v in filter(None, re.split(r"[|;]", r.movie_name_variants)):
            out[v.strip()] = r.stimulus_id
    return out


def build_units(bids: Path) -> pd.DataFrame:
    """Recall units of every NATretrieval word table under ``bids``.

    Columns: subject, session, run ('' when the file has no run entity),
    trial_num, unit_idx, stimulus_id, recall_rank, first_row, last_row,
    n_words, onset, offset. Raises when no table is found or a film name is
    not in the registry.
    """
    files = sorted(bids.glob(BEH_GLOB))
    if not files:
        raise FileNotFoundError(f"no {TASK} word tables under {bids} ({BEH_GLOB})")
    ru = _ru()
    n2i = name_to_id(bids)
    parts = []
    for f in files:
        ent = entities(f.name)
        u = ru.beh_units(ru.read_beh(f))
        unknown = sorted(set(u.movie_name) - set(n2i))
        if unknown:
            raise ValueError(f"{f.name}: movie_name not in the registry: {unknown}")
        parts.append(u.assign(subject=ent["sub"], session=ent["ses"], run=ent.get("run", ""),
                              stimulus_id=u.movie_name.map(n2i)).drop(columns="movie_name"))
    U = ru.add_recall_rank(pd.concat(parts, ignore_index=True))
    cols = ["subject", "session", "run", "trial_num", "unit_idx", "stimulus_id", "recall_rank",
            "first_row", "last_row", "n_words", "onset", "offset"]
    return U[cols].sort_values(UNIT_KEY).reset_index(drop=True)


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", na_values=["n/a"], keep_default_na=False,
                       dtype={"subject": str, "session": str, "run": str})


def read_units(out: Path, subjects=None) -> pd.DataFrame:
    """Every unit table in the derivative, with its entity columns."""
    files = sorted(out.glob(f"sub-*/ses-*/beh/sub-*_task-{TASK}*_recallunits.tsv"))
    if not files:
        raise FileNotFoundError(f"no recallunits tables under {out}: run `build_nat_encret.py units` first")
    parts = []
    for f in files:
        ent = entities(f.name)
        if subjects and ent["sub"] not in subjects:
            continue
        parts.append(read_tsv(f).assign(subject=ent["sub"], session=ent["ses"], run=ent.get("run", "")))
    return pd.concat(parts, ignore_index=True)


def with_text(bids: Path, units: pd.DataFrame) -> pd.DataFrame:
    """``units`` plus a ``text`` column, rebuilt from each unit's beh.tsv rows."""
    ru = _ru()
    parts = []
    for (s, ses, run), g in units.groupby(["subject", "session", "run"], sort=False):
        beh = ru.read_beh(beh_path(bids, dict(sub=s, ses=ses, run=run)))
        parts.append(g.assign(text=ru.rejoin(beh, g)))
    return pd.concat(parts).loc[units.index]


def annotations() -> pd.DataFrame:
    """The parsed SEG-B/SEG-C table (parse_movie_annotations); raises on any parse problem."""
    import parse_movie_annotations as pma

    ann, problems, _, _ = pma.load_annotations()
    hard = [p for p in problems if "MISSING FILE" in p or "REFUSED" in p or "parsed 0" in p]
    if hard:
        raise RuntimeError("annotation parse problems:\n  " + "\n  ".join(hard))
    return ann


def segc_events(ann: pd.DataFrame | None = None) -> pd.DataFrame:
    """SEG-C events, one row each: film, seg_number, onset, offset, text.

    In time order within film; this order is the numbering rMatch's prompt
    shows the matcher (1..n). Rows with an empty description are dropped.
    """
    ann = annotations() if ann is None else ann
    c = ann[ann.level == "C"]
    ev = pd.DataFrame(dict(film=c.stimulus_id, seg_number=c.seg_number, onset=c.onset, offset=c.offset,
                           text=c.description.fillna("").astype(str).str.strip()))
    ev = ev[ev.text != ""]
    return ev.sort_values(["film", "onset"]).reset_index(drop=True)


def segb_spans(ann: pd.DataFrame | None = None) -> pd.DataFrame:
    ann = annotations() if ann is None else ann
    b = ann[ann.level == "B"]
    return pd.DataFrame(dict(film=b.stimulus_id, seg_number=b.seg_number, onset=b.onset, offset=b.offset))


def code_version() -> dict:
    """mmmdata HEAD and whether the tree differs from it (HEAD alone does not say)."""
    def git(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    return dict(commit=git("rev-parse", "--short", "HEAD"), dirty=bool(git("status", "--porcelain", "--", ".")))


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, na_rep="n/a")


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def round_times(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = np.round(df[c].astype(float), DECIMALS)
    return df
