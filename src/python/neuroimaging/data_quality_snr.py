"""Tier 2 of the data-quality collection: the SNR view over tier 1 (registry T2.8).

A view, not a new measurement: it reads only the tier-1 tables
(``tier1_runs.tsv`` for tSNR and temporal DOF, T1.1/T1.2; ``tier1_motion.tsv``
for FD, T1.7; ``tier1_parcels.tsv`` for parcel coverage) and summarises them
two ways. Design record: mmmdata-agents ``docs/archive/workbench/data-quality/``.

* **Per task** (``task_summary.tsv``): per scope x task x regime, the median and
  IQR over runs of each run's in-mask median tSNR, and the run DOF (median and
  minimum residual DOF, median DOF loss, median regressor count). ``task`` is
  also ``all``. Scopes are every subject, ``pooled`` and, when some subjects are
  provisional (fMRIPrep due a rerun), ``pooled_confirmed`` without them.
* **Per session** (``sessions.tsv``): per subject x session x regime, the
  30-session longitudinal view. A session's runs are different tasks, and tSNR
  differs by task, so besides the raw median it carries ``tsnr_rel_median``:
  each run's tSNR over its subject x task x regime median, then the median over
  the session. That is the column to read for drift across sessions. A task the
  subject ran in only one session has no reference outside that session, so
  its runs are n/a there, not 1; ``n_runs_rel`` counts the runs that enter the
  median. FD and coverage do not depend on the regime and repeat on every
  regime's row.

Coverage per run, from ``tier1_parcels.tsv`` (identical in every regime, read
from ``none``): the median Schaefer-400 parcel coverage, the number of parcels
in either atlas below ``COVERAGE_FLOOR``, and the lower of the two HOSPA
hippocampi.

Declared-absent cells (``tier1_runs.tsv`` ``absent``) are counted in
``n_absent`` and left out of every median; they are never filled.

Outputs, under ``<tree>/tier2/snr/``::

    task_summary.tsv    scope x task x regime
    sessions.tsv        subject x session x regime
    provenance.json
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data_quality_connectivity import HIPPOCAMPUS, SEG, SUBCORTICAL
from .data_quality_tier2 import FLOAT_FORMAT, SCHEMA_VERSION, TIER2_DIR, file_sha256

PART = "snr"
#: A parcel below this fraction of its atlas voxels inside the run's brain mask counts as poorly covered.
COVERAGE_FLOOR = 0.8
#: tier1_parcels.tsv regime coverage is read from (it is the same in every regime).
COVERAGE_REGIME = "none"
KEYS = ["sub", "ses", "task", "run"]
ALL_TASKS = "all"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_coverage(tree_root: Path) -> pd.DataFrame:
    """Per run: median Schaefer coverage, parcels below the floor (both atlases), worse hippocampus."""
    path = Path(tree_root) / "tier1_parcels.tsv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; build tier 1 first (`tier1.py run` then `tier1.py collect`)")
    p = pd.read_csv(path, sep="\t", dtype={"sub": str, "ses": str, "run": str}, keep_default_na=False,
                    usecols=KEYS + ["atlas", "regime", "parcel", "coverage"])
    p = p[p["regime"] == COVERAGE_REGIME]
    if p.empty:
        raise ValueError(f"{path} has no regime '{COVERAGE_REGIME}' rows to read coverage from")
    p["run"] = p["run"].replace("n/a", "")
    g = p.groupby(KEYS, sort=True)
    out = pd.DataFrame({
        "cov_median": p[p["atlas"] == SEG].groupby(KEYS)["coverage"].median(),
        "n_cov_below_floor": g["coverage"].apply(lambda s: int((s < COVERAGE_FLOOR).sum())),
    })
    hipp = p[(p["atlas"] == SUBCORTICAL) & p["parcel"].isin(HIPPOCAMPUS.values())]
    out["hipp_cov_min"] = hipp.groupby(KEYS)["coverage"].min()
    return out.reset_index()


def run_table(tier1_runs: pd.DataFrame, motion: pd.DataFrame, coverage: pd.DataFrame,
              provisional: Iterable[str] = ()) -> pd.DataFrame:
    """One row per run x regime with its tSNR, DOF, FD and coverage; a run missing an input is an error."""
    cols = ["regime", "absent", "n_vol", "n_regressors", "dof_resid", "dof_loss", "tsnr_median_mask",
            "mask_n_voxels"]
    runs = tier1_runs[KEYS + cols].copy()
    runs["run"] = runs["run"].replace("n/a", "")
    for c in cols[2:]:
        runs[c] = pd.to_numeric(runs[c].replace("n/a", np.nan))
    # mask_n_voxels is regime-free; an absent cell leaves it n/a, so take it from the run's other cells
    runs["mask_n_voxels"] = runs.groupby(KEYS)["mask_n_voxels"].transform("max")

    out = runs.merge(motion[KEYS + ["fd_mean", "fdf_mean", "fd_frac_gt_0.2"]], on=KEYS, how="left",
                     validate="many_to_one")
    out = out.merge(coverage, on=KEYS, how="left", validate="many_to_one")
    for col, source in (("fd_mean", "tier1_motion.tsv (`tier1.py motion`)"),
                        ("cov_median", "tier1_parcels.tsv (`tier1.py collect`)")):
        missing = out[out[col].isna()].drop_duplicates(KEYS)
        if len(missing):
            raise KeyError(f"{len(missing)} runs have no {source} row, e.g. {missing[KEYS].iloc[0].to_dict()}; "
                           "rebuild it")
    out["provisional"] = out["sub"].isin(set(provisional))
    ok = out[~out["absent"]]
    by_task = ok.groupby(["sub", "task", "regime"])
    recurs = by_task["ses"].transform("nunique") >= 2
    out["tsnr_rel"] = (ok["tsnr_median_mask"] / by_task["tsnr_median_mask"].transform("median")).where(recurs)
    return out.sort_values(KEYS + ["regime"], ignore_index=True)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _scopes(runs: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """``(scope, row mask)``: every subject, all subjects, and all confirmed ones when some are provisional."""
    out = [(f"sub-{s}", runs["sub"] == s) for s in sorted(runs["sub"].unique())]
    out.append(("pooled", pd.Series(True, index=runs.index)))
    if runs["provisional"].any():
        out.append(("pooled_confirmed", ~runs["provisional"]))
    return out


def _task_row(g: pd.DataFrame) -> dict:
    ok = g[~g["absent"]]
    t = ok["tsnr_median_mask"]
    return {
        "n_runs": len(g),
        "n_absent": int(g["absent"].sum()),
        "tsnr_median": t.median(),
        "tsnr_q25": t.quantile(0.25),
        "tsnr_q75": t.quantile(0.75),
        "dof_resid_median": ok["dof_resid"].median(),
        "dof_resid_min": ok["dof_resid"].min(),
        "dof_loss_median": ok["dof_loss"].median(),
        "n_regressors_median": ok["n_regressors"].median(),
    }


def task_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, mask in _scopes(runs):
        scoped = runs[mask]
        provisional = bool(scoped["provisional"].any())
        for regime, by_regime in scoped.groupby("regime", sort=True):
            groups = [(ALL_TASKS, by_regime)] + list(by_regime.groupby("task", sort=True))
            for task, g in groups:
                rows.append({"scope": scope, "task": task, "regime": regime, "provisional": provisional,
                             **_task_row(g)})
    return pd.DataFrame(rows)


def sessions(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sub, ses, regime), g in runs.groupby(["sub", "ses", "regime"], sort=True):
        ok = g[~g["absent"]]
        rel = ok["tsnr_rel"].dropna()
        rows.append({
            "sub": sub, "ses": ses, "regime": regime, "provisional": bool(g["provisional"].any()),
            "tasks": ",".join(sorted(g["task"].unique())),
            "n_runs": len(g),
            "n_absent": int(g["absent"].sum()),
            "tsnr_median": ok["tsnr_median_mask"].median(),
            "tsnr_rel_median": rel.median() if len(rel) else np.nan,
            "n_runs_rel": len(rel),
            "dof_loss_median": ok["dof_loss"].median(),
            "fd_mean_median": g["fd_mean"].median(),
            "fd_mean_max": g["fd_mean"].max(),
            "fdf_mean_median": g["fdf_mean"].median(),
            "fd_frac_gt_0.2_max": g["fd_frac_gt_0.2"].max(),
            "mask_n_voxels_median": g["mask_n_voxels"].median(),
            "cov_median_min": g["cov_median"].min(),
            "n_cov_below_floor_max": int(g["n_cov_below_floor"].max()),
            "hipp_cov_min": g["hipp_cov_min"].min(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SnrResult:
    task_summary: pd.DataFrame
    sessions: pd.DataFrame


def compute(runs: pd.DataFrame) -> SnrResult:
    return SnrResult(task_summary=task_summary(runs), sessions=sessions(runs))


def out_dir(tree_root: Path) -> Path:
    return Path(tree_root) / TIER2_DIR / PART


TABLES = ("task_summary", "sessions")


def write(result: SnrResult, dest: Path, provenance: dict) -> list[Path]:
    """Every table in a fixed float format, so a rebuild is byte-identical."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for name in TABLES:
        path = dest / f"{name}.tsv"
        getattr(result, name).to_csv(path, sep="\t", index=False, na_rep="n/a", float_format=FLOAT_FORMAT)
        written.append(path)
    prov = dict(provenance, schema_version=SCHEMA_VERSION,
                parameters={"coverage_floor": COVERAGE_FLOOR, "coverage_regime": COVERAGE_REGIME, "seg": SEG,
                            "subcortical": SUBCORTICAL, "hippocampus": HIPPOCAMPUS})
    path = dest / "provenance.json"
    path.write_text(json.dumps(prov, indent=2, default=str) + "\n")
    written.append(path)
    return written


def diff(a: Path, b: Path) -> list[str]:
    """Differences between two snr trees; empty means identical."""
    a, b = Path(a), Path(b)
    problems = []
    for name in TABLES:
        pa, pb = a / f"{name}.tsv", b / f"{name}.tsv"
        if not (pa.exists() and pb.exists()):
            problems.append(f"{name}.tsv missing in {'a' if not pa.exists() else 'b'}")
        elif file_sha256(pa) != file_sha256(pb):
            problems.append(f"{name}.tsv differs")
    return problems
