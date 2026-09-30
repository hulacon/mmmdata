"""Tier 2 of the data-quality collection: the univariate part (registry T1.5/T1.8 summaries, T2.13).

Reads only tier-1 outputs: ``tier1_glm.tsv`` (T1.5, `tier1.py glm` then
`collect`), ``tier1_motion.tsv`` (T1.8, `tier1.py motion`) and, for the
split-half, the localizer cells' ``_betas.nii.gz`` / ``_sigmasquared.nii.gz`` and
their sidecars. Design record: mmmdata-agents ``docs/archive/workbench/data-quality/``
(D1–D4 DECIDED 2026-09-29).

* **Task R²** (``task_r2.tsv``): per scope x task x regime, the median and IQR
  over runs of the **fraction of in-mask voxels with task-F p < .001** (the T1.5
  headline: its null level is .001 at any dof, though iid-noise-based), of each
  run's in-mask median and p99 adjusted partial R² (an effect size), and the raw
  median (for reference only: it ranks regimes by regressor count), the median
  residual DOF, and T1.8's motion–task |r| (median and max over runs; regime-free,
  repeated on every regime's row). Scopes as in the SNR part.
* **Split-half** (``split_half.tsv``, T2.13): per subject x localizer task x
  contrast x regime, the glm-bake-off protocol on this collection's per-run
  cells. Runs sorted by (session, run), odd positions form half 1 and even half 2;
  each half is a precision-weighted fixed effect of the runs' contrast, inside
  the intersection of every run's brain mask for that subject and task; the
  score is :func:`neuroimaging.glm.harness.score_halves` on the two t maps (``r``
  plus the pre-registered Dice sets where the bake-off has them). A run's mask is
  read from its own ``sigma²`` map (non-zero or NaN inside, 0 outside), so this
  never opens an fMRIPrep file. Voxels below a run's percent-signal-change floor
  are NaN in that run and so drop out of the half and the score.

Declared-absent cells are counted and left out; nothing is filled.

Outputs, under ``<tree>/tier2/univariate/``::

    task_r2.tsv      scope x task x regime
    split_half.tsv   subject x task x contrast x regime
    provenance.json
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .data_quality_glm import LOCALIZER_MODELS, TABLE_NAME
from .data_quality_tier2 import FLOAT_FORMAT, SCHEMA_VERSION, TIER2_DIR, file_sha256
from .glm.harness import N_SETS, Z_THRESHOLD, score_halves, split_runs

PART = "univariate"
KEYS = ["sub", "ses", "task", "run"]
ALL_TASKS = "all"
TABLES = ("task_r2", "split_half")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_glm(tree_root: Path) -> pd.DataFrame:
    path = Path(tree_root) / f"{TABLE_NAME}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; build it first (`tier1.py glm` then `tier1.py collect`)")
    df = pd.read_csv(path, sep="\t", dtype={"sub": str, "ses": str, "run": str}, keep_default_na=False)
    df["run"] = df["run"].replace("n/a", "")
    df["absent"] = df["absent"].astype(str).str.lower() == "true"
    for c in ("task_frac_p001", "task_r2adj_median", "task_r2adj_p99", "task_r2_median", "dof_resid", "n_regressors"):
        df[c] = pd.to_numeric(df[c].replace("n/a", np.nan))
    return df


def run_table(glm: pd.DataFrame, motion: pd.DataFrame, provisional: Iterable[str] = ()) -> pd.DataFrame:
    """One row per task run x regime with its T1.5 summaries and T1.8; a run missing T1.8 is an error."""
    need = ["motion_task_r_max"]
    if not set(need) <= set(motion.columns):
        raise KeyError("tier1_motion.tsv has no motion_task_r_max column; rebuild it (`tier1.py motion`)")
    m = motion[KEYS + need].copy()
    m["motion_task_r_max"] = pd.to_numeric(m["motion_task_r_max"].replace("n/a", np.nan))
    out = glm[KEYS + ["regime", "absent", "task_frac_p001", "task_r2adj_median", "task_r2adj_p99", "task_r2_median", "dof_resid",
                      "n_regressors"]].merge(m, on=KEYS, how="left", validate="many_to_one")
    missing = out[out["motion_task_r_max"].isna()].drop_duplicates(KEYS)
    if len(missing):
        raise KeyError(f"{len(missing)} task runs have no T1.8 value in tier1_motion.tsv, e.g. "
                       f"{missing[KEYS].iloc[0].to_dict()}; rebuild it (`tier1.py motion`)")
    out["provisional"] = out["sub"].isin(set(provisional))
    return out.sort_values(KEYS + ["regime"], ignore_index=True)


# ---------------------------------------------------------------------------
# Task R²
# ---------------------------------------------------------------------------

def _scopes(runs: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    out = [(f"sub-{s}", runs["sub"] == s) for s in sorted(runs["sub"].unique())]
    out.append(("pooled", pd.Series(True, index=runs.index)))
    if runs["provisional"].any():
        out.append(("pooled_confirmed", ~runs["provisional"]))
    return out


def _r2_row(g: pd.DataFrame) -> dict:
    ok = g[~g["absent"]]
    med, p99, frac = ok["task_r2adj_median"], ok["task_r2adj_p99"], ok["task_frac_p001"]
    return {
        "n_runs": len(g),
        "n_absent": int(g["absent"].sum()),
        "frac_p001_median": frac.median(),
        "frac_p001_q25": frac.quantile(0.25),
        "frac_p001_q75": frac.quantile(0.75),
        "r2adj_median": med.median(),
        "r2adj_median_q25": med.quantile(0.25),
        "r2adj_median_q75": med.quantile(0.75),
        "r2adj_p99": p99.median(),
        "r2adj_p99_q25": p99.quantile(0.25),
        "r2adj_p99_q75": p99.quantile(0.75),
        "r2raw_median": ok["task_r2_median"].median(),
        "dof_resid_median": ok["dof_resid"].median(),
        "n_regressors_median": ok["n_regressors"].median(),
        "motion_task_r_median": g["motion_task_r_max"].median(),
        "motion_task_r_max": g["motion_task_r_max"].max(),
    }


def task_r2(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, mask in _scopes(runs):
        scoped = runs[mask]
        provisional = bool(scoped["provisional"].any())
        for regime, by_regime in scoped.groupby("regime", sort=True):
            for task, g in [(ALL_TASKS, by_regime)] + list(by_regime.groupby("task", sort=True)):
                rows.append({"scope": scope, "task": task, "regime": regime, "provisional": provisional,
                             **_r2_row(g)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Split-half (T2.13)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _Cell:
    ses: str
    run: str
    meta: dict
    betas: np.ndarray  # (x, y, z, n_conditions)
    sigma2: np.ndarray  # (x, y, z); 0 outside the run's mask, NaN below its PSC floor


def _load_cell(json_path: Path) -> _Cell:
    import nibabel as nib

    meta = json.loads(json_path.read_text())
    stem = json_path.name[: -len("_glm.json")]
    betas = np.asarray(nib.load(str(json_path.with_name(f"{stem}_betas.nii.gz"))).dataobj, dtype=np.float64)
    s2 = np.asarray(nib.load(str(json_path.with_name(f"{stem}_sigmasquared.nii.gz"))).dataobj, dtype=np.float64)
    return _Cell(ses=meta["ses"], run=meta["run"] or "", meta=meta, betas=betas, sigma2=s2)


def _half_t(cells: list[_Cell], name: str, mask: np.ndarray) -> np.ndarray:
    """Precision-weighted fixed-effects t for contrast ``name`` over ``cells``, inside ``mask``."""
    num = np.zeros(int(mask.sum()))
    prec = np.zeros(int(mask.sum()))
    for c in cells:
        w = np.asarray(c.meta["contrasts"][name], dtype=np.float64)
        eff = np.tensordot(c.betas, w, axes=([3], [0]))[mask]
        var = c.sigma2[mask] * float(w @ np.asarray(c.meta["cov_unscaled"]) @ w)
        num += eff / var
        prec += 1.0 / var
    with np.errstate(divide="ignore", invalid="ignore"):
        return (num / prec) * np.sqrt(prec)  # effect / sqrt(variance), variance = 1 / precision


def split_half(tree_root: Path, glm: pd.DataFrame, regimes: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """T2.13 per subject x localizer task x contrast x regime; subjects with < 2 runs are skipped."""
    loc = glm[glm["task"].isin(LOCALIZER_MODELS) & ~glm["absent"]]
    if regimes is not None:
        loc = loc[loc["regime"].isin(set(regimes))]
    rows = []
    for (sub, task, regime), g in loc.groupby(["sub", "task", "regime"], sort=True):
        g = g.sort_values(["ses", "run"])
        if len(g) < 2:
            continue
        cells = []
        for r in g.itertuples(index=False):
            prefix = f"sub-{r.sub}_ses-{r.ses}_task-{r.task}" + (f"_run-{r.run}" if r.run else "")
            hits = sorted((Path(tree_root) / f"sub-{r.sub}" / f"ses-{r.ses}" / "func").glob(f"{prefix}_space-*_desc-{regime}_glm.json"))
            if len(hits) != 1:
                raise FileNotFoundError(f"expected one GLM sidecar for {prefix} desc-{regime}, found {len(hits)}")
            cells.append(_load_cell(hits[0]))
        mask = np.logical_and.reduce([c.sigma2 != 0 for c in cells])
        h1, h2 = split_runs(len(cells))
        n_set = N_SETS.get(LOCALIZER_MODELS[task], ())
        for name in cells[0].meta["contrasts"]:
            t1 = _half_t([cells[i] for i in h1], name, mask)
            t2 = _half_t([cells[i] for i in h2], name, mask)
            s = score_halves(t1, t2, np.ones_like(t1, dtype=bool), n_set)
            n_z = s.pop(f"n@z{Z_THRESHOLD}")
            rows.append({
                "sub": sub, "task": task, "contrast": name, "regime": regime,
                "n_runs": len(cells), "n_half1": len(h1), "n_half2": len(h2),
                "runs_half1": ",".join(f"{cells[i].ses}:{cells[i].run}" for i in h1),
                "n_mask": int(mask.sum()), "n_valid": s.pop("n_valid"), "r": s.pop("r"),
                f"dice@z{Z_THRESHOLD}": s.pop(f"dice@z{Z_THRESHOLD}"),
                f"n@z{Z_THRESHOLD}_half1": n_z[0], f"n@z{Z_THRESHOLD}_half2": n_z[1],
                **{k: v for k, v in s.items() if k.startswith("dice@")},
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class UnivariateResult:
    task_r2: pd.DataFrame
    split_half: pd.DataFrame


def compute(tree_root: Path, runs: pd.DataFrame, glm: pd.DataFrame,
            regimes: Optional[Iterable[str]] = None) -> UnivariateResult:
    return UnivariateResult(task_r2=task_r2(runs), split_half=split_half(tree_root, glm, regimes))


def out_dir(tree_root: Path) -> Path:
    return Path(tree_root) / TIER2_DIR / PART


def write(result: UnivariateResult, dest: Path, provenance: dict) -> list[Path]:
    """Every table in a fixed float format, so a rebuild is byte-identical."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for name in TABLES:
        path = dest / f"{name}.tsv"
        getattr(result, name).to_csv(path, sep="\t", index=False, na_rep="n/a", float_format=FLOAT_FORMAT)
        written.append(path)
    prov = dict(provenance, schema_version=SCHEMA_VERSION,
                parameters={"localizer_models": LOCALIZER_MODELS, "n_sets": {k: list(v) for k, v in N_SETS.items()},
                            "z_threshold": Z_THRESHOLD,
                            "halves": "runs sorted by (session, run); odd positions = half 1, even = half 2; "
                                      "precision-weighted fixed effects within a half; t maps scored",
                            "mask": "intersection of the subject's run masks for the task (sigma² != 0)"})
    path = dest / "provenance.json"
    path.write_text(json.dumps(prov, indent=2, default=str) + "\n")
    written.append(path)
    return written


def diff(a: Path, b: Path) -> list[str]:
    """Differences between two univariate trees; empty means identical."""
    a, b = Path(a), Path(b)
    problems = []
    for name in TABLES:
        pa, pb = a / f"{name}.tsv", b / f"{name}.tsv"
        if not (pa.exists() and pb.exists()):
            problems.append(f"{name}.tsv missing in {'a' if not pa.exists() else 'b'}")
        elif file_sha256(pa) != file_sha256(pb):
            problems.append(f"{name}.tsv differs")
    return problems
