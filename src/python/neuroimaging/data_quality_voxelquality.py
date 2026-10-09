"""Tier 2 of the data-quality collection: the ``alignment`` and ``voxelmaps`` parts.

Views over the voxel-quality tier-1 tables, never a voxel: ``tier1_alignment.tsv``
(:mod:`data_quality_alignment`) and ``tier1_voxelmaps_{sessions,parcels,surfvol}.tsv``
(:mod:`data_quality_voxelmaps`). Design record: mmmdata-agents
``docs/workbench/voxel-quality/``.

``alignment`` (``<tree>/tier2/alignment/``)::

    runs.tsv        every run's row, with ``provisional``
    sessions.tsv    subject x session: median / max displacement, flagged runs, edge r, Dice
    subjects.tsv    subject: displacement quantiles, flagged runs, lowest edge r

``voxelmaps`` (``<tree>/tier2/voxelmaps/``)::

    sessions.tsv          subject x session x space(+hemi): dropped / uncovered / lost
                          fractions, relative tSNR, ``examine``
    parcel_sessions.tsv   subject x space x hemi x atlas x parcel: sessions in which most
                          of the parcel is lost (frac_lost > PARCEL_LOST)
    surfvol.tsv           subject x hemi: surface ÷ sampled-volume ratio, parcel ρ,
                          ribbon in-mask, depth profile, ``adequate``
    verdict.tsv           subject: the Settles-when 2-4 readings in one row
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data_quality_alignment as dqa
from . import data_quality_voxelmaps as dqv
from .data_quality_tier2 import FLOAT_FORMAT, SCHEMA_VERSION, TIER2_DIR, file_sha256

#: A parcel counts as lost in a session when more than this fraction of it is lost.
PARCEL_LOST = 0.5


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `tier1.py collect` after the voxel-quality cells")
    return pd.read_csv(path, sep="\t", na_values=["n/a"], dtype={"sub": str, "ses": str, "run": str})


def _mark(df: pd.DataFrame, provisional: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["provisional"] = df["sub"].isin(provisional)
    return df


def _write(tables: dict[str, pd.DataFrame], dest: Path, provenance: dict, parameters: dict) -> list[Path]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for name, df in tables.items():
        p = dest / f"{name}.tsv"
        df.to_csv(p, sep="\t", index=False, na_rep="n/a", float_format=FLOAT_FORMAT)
        written.append(p)
    p = dest / "provenance.json"
    p.write_text(json.dumps(dict(provenance, schema_version=SCHEMA_VERSION, parameters=parameters),
                            indent=2, default=str) + "\n")
    written.append(p)
    return written


def _diff(a: Path, b: Path, names: tuple[str, ...]) -> list[str]:
    problems = []
    for name in names:
        pa, pb = Path(a) / f"{name}.tsv", Path(b) / f"{name}.tsv"
        if not (pa.exists() and pb.exists()):
            problems.append(f"{name}.tsv missing in {'a' if not pa.exists() else 'b'}")
        elif file_sha256(pa) != file_sha256(pb):
            problems.append(f"{name}.tsv differs")
    return problems


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------

class alignment:  # noqa: N801 - a namespace with the part interface tier2.py expects
    PART = "alignment"
    TABLES = ("runs", "sessions", "subjects")
    INPUTS = (f"{dqa.TABLE_NAME}.tsv",)

    @staticmethod
    def compute(tree_root: Path, provisional: list[str]) -> dict[str, pd.DataFrame]:
        runs = _mark(_read(Path(tree_root) / f"{dqa.TABLE_NAME}.tsv"), provisional)
        runs = runs.sort_values(["sub", "ses", "task", "run"], na_position="first").reset_index(drop=True)
        g = runs.groupby(["sub", "ses"], sort=True)
        sessions = g.agg(n_runs=("disp_mean_mm", "size"), disp_mean_median=("disp_mean_mm", "median"),
                         disp_mean_max=("disp_mean_mm", "max"), disp_max_max=("disp_max_mm", "max"),
                         n_flagged=("flag", "sum"), edge_r_median=("edge_r", "median"),
                         edge_r_min=("edge_r", "min"), mask_dice_median=("mask_dice", "median"),
                         off_grid=("off_grid", "any"), provisional=("provisional", "first")).reset_index()
        s = runs.groupby("sub", sort=True)
        subjects = s.agg(n_runs=("disp_mean_mm", "size"), n_sessions=("ses", "nunique"),
                         n_flagged=("flag", "sum"),
                         disp_mean_p50=("disp_mean_mm", "median"),
                         disp_mean_p95=("disp_mean_mm", lambda x: float(np.percentile(x, 95))),
                         disp_mean_max=("disp_mean_mm", "max"), edge_r_min=("edge_r", "min"),
                         mask_dice_min=("mask_dice", "min"), provisional=("provisional", "first")).reset_index()
        return {"runs": runs, "sessions": sessions, "subjects": subjects}

    @staticmethod
    def write(tables: dict[str, pd.DataFrame], dest: Path, provenance: dict) -> list[Path]:
        return _write(tables, dest, provenance, {"displacement_flag_mm": dqa.DISPLACEMENT_FLAG_MM,
                                                 "edge_sigma_mm": dqa.EDGE_SIGMA_MM,
                                                 "mask_majority": dqa.MASK_MAJORITY})

    @staticmethod
    def diff(a: Path, b: Path) -> list[str]:
        return _diff(a, b, alignment.TABLES)


# ---------------------------------------------------------------------------
# voxelmaps
# ---------------------------------------------------------------------------

def parcel_sessions(parcels: pd.DataFrame) -> pd.DataFrame:
    """Per subject x space x hemi x atlas x parcel: sessions where the parcel is mostly lost."""
    p = parcels[parcels["ses"].notna() & parcels["frac_lost"].notna()].copy()
    p["hemi"] = p["hemi"].fillna("n/a")
    p["lost"] = p["frac_lost"] > PARCEL_LOST
    keys = ["sub", "space", "hemi", "atlas", "parcel"]
    out = p.groupby(keys, sort=True).agg(
        n_sessions=("ses", "nunique"), n_sessions_lost=("lost", "sum"),
        frac_lost_median=("frac_lost", "median"), frac_lost_max=("frac_lost", "max"),
        sessions_lost=("ses", lambda s: ",".join(sorted(s[p.loc[s.index, "lost"]])) or "n/a"),
    ).reset_index()
    return out


def verdict(sessions: pd.DataFrame, surfvol: pd.DataFrame, align: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for sub in sorted(set(sessions["sub"]) | set(surfvol["sub"])):
        s = sessions[sessions["sub"] == sub]
        sv = surfvol[surfvol["sub"] == sub]
        vol = s[s["space"] == dqv.VOLUME_SPACE]
        row = {"sub": sub, "n_sessions": int(vol["ses"].nunique()),
               "sessions_examine_T1w": ",".join(sorted(vol.loc[vol["examine"], "ses"])) or "n/a",
               "sessions_examine_fsnative": ",".join(sorted(set(s.loc[(s["space"] == dqv.SURFACE_SPACE) & s["examine"], "ses"]))) or "n/a",
               "frac_lost_T1w_median": float(vol["frac_lost"].median()) if len(vol) else np.nan}
        for h in dqv.HEMIS:
            r = sv[sv["hemi"] == h]
            row[f"surfvol_mean_median_{h}"] = float(r["surfvol_mean_median"].iloc[0]) if len(r) else np.nan
            row[f"rho_parcel_sampled_{h}"] = float(r["rho_parcel_sampled"].iloc[0]) if len(r) else np.nan
            row[f"rho_parcel_ribbon_{h}"] = float(r["rho_parcel_ribbon"].iloc[0]) if len(r) else np.nan
            row[f"frac_ribbon_below_floor_{h}"] = float(r["frac_ribbon_below_floor"].iloc[0]) if len(r) else np.nan
        row["surface_adequate"] = bool(len(sv) == len(dqv.HEMIS) and sv["adequate"].astype(bool).all())
        if align is not None and not align.empty:
            a = align[align["sub"] == sub]
            row["n_runs_aligned"] = int(len(a))
            row["n_runs_flagged"] = int(a["flag"].astype(bool).sum())
        rows.append(row)
    return pd.DataFrame(rows)


class voxelmaps:  # noqa: N801
    PART = "voxelmaps"
    TABLES = ("sessions", "parcel_sessions", "surfvol", "verdict")
    INPUTS = (f"{dqv.SESSIONS_TABLE}.tsv", f"{dqv.PARCELS_TABLE}.tsv", f"{dqv.SURFVOL_TABLE}.tsv")

    @staticmethod
    def compute(tree_root: Path, provisional: list[str]) -> dict[str, pd.DataFrame]:
        tree_root = Path(tree_root)
        sessions = _mark(_read(tree_root / f"{dqv.SESSIONS_TABLE}.tsv"), provisional)
        parcels = _read(tree_root / f"{dqv.PARCELS_TABLE}.tsv")
        surfvol = _mark(_read(tree_root / f"{dqv.SURFVOL_TABLE}.tsv"), provisional)
        sessions["examine"] = sessions["examine"].astype(bool)
        align_path = tree_root / f"{dqa.TABLE_NAME}.tsv"
        align = _read(align_path) if align_path.exists() else None
        ps = _mark(parcel_sessions(parcels), provisional)
        v = _mark(verdict(sessions, surfvol, align), provisional)
        sort = ["sub", "ses", "space", "hemi"]
        return {"sessions": sessions.sort_values(sort, na_position="first").reset_index(drop=True),
                "parcel_sessions": ps, "surfvol": surfvol.sort_values(["sub", "hemi"]).reset_index(drop=True),
                "verdict": v}

    @staticmethod
    def write(tables: dict[str, pd.DataFrame], dest: Path, provenance: dict) -> list[Path]:
        return _write(tables, dest, provenance, {"drop_floor": dqv.DROP_FLOOR, "consensus": dqv.CONSENSUS,
                                                 "examine_multiple": dqv.EXAMINE_MULTIPLE,
                                                 "examine_floor": dqv.EXAMINE_FLOOR,
                                                 "ribbon_floor": dqv.RIBBON_FLOOR, "parcel_lost": PARCEL_LOST,
                                                 "surface_adequate": "median ratio and every depth_*_rel in "
                                                 f"{list(dqv.ADEQUATE_BAND)}, rho_parcel_ribbon >= {dqv.ADEQUATE_RHO}"})

    @staticmethod
    def diff(a: Path, b: Path) -> list[str]:
        return _diff(a, b, voxelmaps.TABLES)


PARTS = {alignment.PART: alignment, voxelmaps.PART: voxelmaps}


def out_dir(tree_root: Path, part: str) -> Path:
    return Path(tree_root) / TIER2_DIR / part
