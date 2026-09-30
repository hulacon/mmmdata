"""GLMsingle-native rows of the data-quality collection: registry T2.10, T2.11, T2.12.

GLMsingle fits do their own denoising (FitHRF, GLMdenoise, fracridge), so their
rows carry ``regime = glmsingle`` and never enter the (runs x regimes) count.
The fits are the retrieval-modeling product, ``derivatives/glmsingle_tb/sub-##/<arm>``
(siloed arms ``enc``, ``ret-image``, ``ret-word``; the retained ``-tbonly`` arms are
not read). Design record: mmmdata-agents ``docs/workbench/data-quality/`` (D1–D3
DECIDED 2026-09-29).

**Tier 1, one cell per subject x arm** (``tier1.py glmsingle``). Each fit's three
beta dicts (``TYPEB_FITHRF``, ``TYPEC_FITHRF_GLMDENOISE``,
``TYPED_FITHRF_GLMDENOISE_RR``; 5–15 GB pickles that must be loaded whole, so
this runs under sbatch) are read once, one at a time, and reduced to:

* **T2.11, fit QC:** per beta type, GLMsingle's ``R2`` (percent); from TYPEB the
  ``HRFindex`` (a distribution only, never physiology: it is unstable across
  fits); from TYPEC/TYPED ``pcnum`` and the ``noisepool`` (how much of it lies
  outside the brain mask: the fits are unmasked); from TYPED ``FRACvalue``.
* **T2.12, noise ceiling** (NSD, Allen et al. 2022): per beta type, betas
  z-scored per voxel within each session over the arm's trials; the noise
  variance is the mean over conditions of the across-repeat variance
  (``ddof=1``), the signal variance ``max(0, 1 - noise)``, and
  ``ncsnr = sqrt(signal) / sqrt(noise)``. The noise ceiling for an average of
  ``n`` trials is ``100 * ncsnr² / (ncsnr² + 1/n)``, reported at ``n = 1`` and at
  ``n`` = the set's repeat count. Two condition sets (DECIDED D1 2026-09-29):
  ``repeat``, every non-anchor condition presented more than once (all with
  one repeat count, else a loud error), and ``anchor``, the ``sharedId == 1``
  super-repeat items, reported apart. **Where the repeats fall differs by arm**
  and is recorded as ``rep_scope``: the enc arm's three encodings of an item lie
  in one session (often one run), so its ``repeat`` ncsnr is a within-session
  reliability and reads high against NSD's cross-session repeats; the ret arms'
  two presentations are a TB retrieval and the ses-30 final recall, months
  apart; the anchors span every session. TYPEC/TYPED were tuned on these same
  repeats, so their ncsnr is tuning-exposed: read TYPEB beside them (D2).
* **The ncsnr null** (added 2026-09-29): the estimator has a positive floor.
  With no shared signal about half the voxels get ncsnr 0 (``max(0, ·)``) and
  the rest a positive value, so the median sits on the edge of the zero mass and
  reads positive (.06 for 120 Gaussian pairs). A session z-score taken over many
  more trials than the repeats, as in these arms, pushes the positive share above
  half (.55 synthetic), and heavy tails add a little more. The measured size of
  the floor on the real fits is in the design record. So every cell also computes
  :data:`N_NULL` label shuffles: each repeat's later presentations are permuted
  across items **within their own run**, which keeps session and run structure
  (and enc's same-run proximity) and breaks only item identity. The **headline
  is ``frac_exceed``**, the fraction of voxels whose matched ncsnr beats every
  draw (null level .05), beside the null's median and the matched − null excess.
  The raw NSD ncsnr and noise ceilings are kept for reference only; they are not
  comparable to NSD's without this floor.

Voxels: the intersection of the fit runs' fMRIPrep brain masks, then a floor on
GLMsingle's ``meanvol`` (the time-averaged EPI, the denominator of its percent
betas) at :data:`MEANVOL_FLOOR_FRACTION` of the median, taken over the whole
mask for whole-mask summaries and within each parcel for parcel summaries (the
retrieval-modeling ``clean_voxels`` rule). The floor marks tissue signal, not
response. Nothing below it is filled; it is left out.

Per cell, under ``<tree>/sub-##/func/``::

    sub-##_arm-<arm>_space-S_desc-glmsingle_stat.nii.gz   4D maps, volume names in the sidecar
    sub-##_arm-<arm>_space-S_desc-glmsingle_stat.json     whole-mask summaries per beta type, provenance
    sub-##_arm-<arm>_space-S_desc-glmsingle_parcels.tsv   per parcel x beta type

``collect`` flattens them into ``tier1_glmsingle.tsv`` (subject x arm x beta type)
and ``tier1_glmsingle_parcels.tsv``.

**Tier 2, part ``glmsingle``** (``tier2.py build --parts glmsingle``) reads only
those two tables, ``tier1_runs.tsv`` (which subjects have TB runs, so a subject
without a fit gets a declared-absent row) and the retrieval-modeling benchmark
tables:

* ``fit_summary.tsv``: the tier-1 whole-mask rows, plus absent rows.
* ``networks.tsv``: per subject x arm x beta type x region (Schaefer 17 network
  and hemisphere, or HOSPA structure), medians over parcels of the parcel medians.
* ``benchmark_6cell.tsv`` (T2.10, adopt): the settled 6-cell reinstatement
  tables, concatenated, not recomputed; their hashes are in the provenance.
"""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from . import data_quality as dq
from .fmriprep_layout import space_part

SCHEMA_VERSION = "1.1"  # 1.1: label-shuffle null for ncsnr (2026-09-29)
REGIME = "glmsingle"
PART = "glmsingle"
TABLE_NAME = "tier1_glmsingle"
PARCELS_TABLE_NAME = "tier1_glmsingle_parcels"
FIT_TREE = "glmsingle_tb"
BENCHMARK_TREE = "pattern_similarity/results/retrieval_modeling"

#: The final siloed arms (TB + FIN retrieval by cue). The ``-tbonly`` arms are not read.
ARMS: tuple[str, ...] = ("enc", "ret-image", "ret-word")

#: beta type -> GLMsingle output file.
BETA_FILES: dict[str, str] = {
    "B": "TYPEB_FITHRF.npy",
    "C": "TYPEC_FITHRF_GLMDENOISE.npy",
    "D": "TYPED_FITHRF_GLMDENOISE_RR.npy",
}

#: A voxel counts when its meanvol is at least this fraction of the median (mask or parcel).
MEANVOL_FLOOR_FRACTION = 0.25

#: GLMsingle's default HRF library size.
N_HRFS = 20

CONDITION_SETS: tuple[str, ...] = ("repeat", "anchor")

#: Label-shuffle draws per set and beta type. A voxel "exceeds" when its matched ncsnr beats
#: every draw, so the null level of ``frac_exceed`` is 1 / (N_NULL + 1) = .05.
N_NULL = 19


# ---------------------------------------------------------------------------
# Estimators (pure; tested on synthetic data)
# ---------------------------------------------------------------------------

def zscore_by_session(betas: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    """Z-score each voxel's betas within each session (``ddof=1``), as NSD does. Returns float32.

    ``betas`` is (voxels, trials); ``sessions`` labels the trials. A voxel with no
    variance in a session gets NaN there.
    """
    out = np.empty(betas.shape, dtype=np.float32)
    sessions = np.asarray(sessions)
    for ses in np.unique(sessions):
        idx = np.flatnonzero(sessions == ses)
        if idx.size < 2:
            raise ValueError(f"session {ses!r} has {idx.size} trial(s); z-scoring needs at least 2")
        x = betas[:, idx].astype(np.float64)
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, ddof=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, idx] = np.where(sd > 0, (x - mu) / sd, np.nan)
    return out


def ncsnr(z: np.ndarray, groups: np.ndarray, chunk: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    """NSD noise-ceiling SNR per voxel from session-z-scored betas.

    ``z`` is (voxels, trials); ``groups`` is (conditions, n) trial columns, one row
    per condition, all with the same repeat count ``n >= 2``. Returns
    ``(ncsnr, noise_var)``: noise variance = mean over conditions of the
    across-repeat variance (``ddof=1``), signal variance ``max(0, 1 - noise)``.
    """
    groups = np.asarray(groups)
    if groups.ndim != 2 or groups.shape[1] < 2 or groups.shape[0] < 1:
        raise ValueError(f"groups must be (conditions, n >= 2), got shape {groups.shape}")
    n_vox = z.shape[0]
    noise = np.empty(n_vox, dtype=np.float64)
    for s in range(0, n_vox, chunk):
        x = z[s:s + chunk][:, groups].astype(np.float64)  # (v, C, n)
        noise[s:s + chunk] = x.var(axis=2, ddof=1).mean(axis=1)
    signal = np.clip(1.0 - noise, 0.0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.sqrt(signal) / np.sqrt(noise)
    return snr.astype(np.float32), noise.astype(np.float32)


def shuffle_groups(groups: np.ndarray, block: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Re-pair repeats across items: column 0 stays; each later column is permuted among the rows
    whose trial in that column shares a ``block`` label (the run), so no trial changes run."""
    g = np.array(groups, copy=True)
    for col in range(1, g.shape[1]):
        key = block[g[:, col]]
        for b in np.unique(key):
            rows = np.flatnonzero(key == b)
            g[rows, col] = g[rng.permutation(rows), col]
    return g


def ncsnr_null(z: np.ndarray, groups: np.ndarray, block: np.ndarray, n_draws: int, seed: int,
               chunk: int = 20_000) -> np.ndarray:
    """(n_draws, voxels) ncsnr under label shuffles (:func:`shuffle_groups`), float32."""
    rng = np.random.default_rng(seed)
    return np.stack([ncsnr(z, shuffle_groups(groups, block, rng), chunk)[0] for _ in range(n_draws)])


def null_seed(*parts: str) -> int:
    """A reproducible seed from names (Python's ``hash`` is salted per process)."""
    return zlib.crc32("|".join(parts).encode())


def noise_ceiling(snr: np.ndarray, n: int) -> np.ndarray:
    """NSD noise ceiling (percent variance explainable) for an average of ``n`` trials."""
    s2 = np.asarray(snr, dtype=np.float64) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return (100.0 * s2 / (s2 + 1.0 / n)).astype(np.float32)


def condition_sets(trial_info: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """The ``repeat`` and ``anchor`` condition sets of one arm, as trial-column groups.

    ``trial_info`` rows are in ``betasmd`` column order. Anchors are the
    ``sharedId == 1`` conditions; ``repeat`` is every other condition seen more than
    once. Each set must have one repeat count: GLMsingle arms are built that way,
    and a mixture would need a different estimator, so it is refused, not averaged.
    """
    ti = trial_info.reset_index(drop=True)
    cond = ti["condition_id"].astype(str)
    anchor_ids = set(cond[ti["sharedId"] == 1])
    counts = cond.value_counts()
    out: dict[str, dict[str, Any]] = {}
    for name, ids in (("repeat", [c for c in counts.index if c not in anchor_ids and counts[c] > 1]),
                      ("anchor", sorted(anchor_ids))):
        if not ids:
            raise ValueError(f"no {name} conditions in this arm")
        reps = sorted(set(int(counts[c]) for c in ids))
        if len(reps) != 1:
            raise ValueError(f"{name} conditions have mixed repeat counts {reps}; the NSD estimator needs one")
        ids = sorted(ids, key=lambda c: (len(c), c))
        groups = np.stack([np.flatnonzero(cond.to_numpy() == c) for c in ids])
        ses = ti["session"].astype(str).to_numpy()[groups]
        runs = (ti["session"].astype(str) + "/" + ti["run"].astype(str)).to_numpy()[groups]
        n_ses = np.array([len(set(r)) for r in ses])
        scope = ("within-session" if (n_ses == 1).all() else
                 "cross-session" if (n_ses > 1).all() else "mixed")
        # How far the within-run shuffle can move this set: later presentations alone in their run
        # stay put, and small blocks often permute to themselves, so the null is conservative there.
        block = (ti["session"].astype(str) + "/" + ti["run"].astype(str)).to_numpy()
        sizes = []
        for col in range(1, groups.shape[1]):
            u, c = np.unique(block[groups[:, col]], return_counts=True)
            sizes.extend(c.tolist())
        sizes = np.repeat(sizes, sizes)  # one entry per trial
        out[name] = {
            "groups": groups, "n_conditions": int(groups.shape[0]), "n_reps": int(reps[0]),
            "rep_scope": scope,
            "frac_conditions_one_run": float(np.mean([len(set(r)) == 1 for r in runs])),
            "null_block_median": float(np.median(sizes)),
            "null_frac_unshufflable": float(np.mean(sizes == 1)),
        }
    return out


def meanvol_floor(meanvol: np.ndarray, fraction: float = MEANVOL_FLOOR_FRACTION) -> np.ndarray:
    """Boolean: ``meanvol >= fraction * median(meanvol)`` over the finite values given."""
    mv = np.asarray(meanvol, dtype=np.float64)
    med = np.nanmedian(mv) if np.isfinite(mv).any() else np.nan
    with np.errstate(invalid="ignore"):
        return np.isfinite(mv) & (mv >= fraction * med)


def _q(x: np.ndarray, q: float) -> float:
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else float("nan")


# ---------------------------------------------------------------------------
# Tier 1: inputs and layout
# ---------------------------------------------------------------------------

def fit_dir(glmsingle_root: Path, subject: str, arm: str) -> Path:
    return Path(glmsingle_root) / f"sub-{subject}" / arm


def find_fits(glmsingle_root: Path, subject: Optional[str] = None, arm: Optional[str] = None) -> list[tuple[str, str]]:
    """``(subject, arm)`` for every fit on disk with a trial table, in :data:`ARMS` only."""
    out = []
    for sd in sorted(Path(glmsingle_root).glob("sub-*")):
        sub = sd.name.removeprefix("sub-")
        if subject and sub != subject:
            continue
        for a in ARMS:
            if arm and a != arm:
                continue
            if (sd / a / "trial_info.csv").exists():
                out.append((sub, a))
    return out


def fit_keys(fdir: Path) -> dict[str, Any]:
    """Identity of a fit without hashing 40 GB: size + mtime of each beta file, sha256 of the tables."""
    fdir = Path(fdir)
    keys: dict[str, Any] = {}
    for bt, fname in BETA_FILES.items():
        p = fdir / "glmsingle_outputs" / fname
        if not p.exists():
            raise FileNotFoundError(f"GLMsingle output missing: {p}")
        st = p.stat()
        keys[f"TYPE{bt}"] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    for name in ("trial_info.csv", "run_metadata.json"):
        keys[name] = dq.file_sha256(fdir / name)
    return keys


def cell_stem(tree_root: Path, subject: str, arm: str, space: str) -> Path:
    return Path(tree_root) / f"sub-{subject}" / "func" / f"sub-{subject}_arm-{arm}{space_part(space)}_desc-{REGIME}"


def cell_paths(tree_root: Path, subject: str, arm: str, space: str) -> dict[str, Path]:
    stem = cell_stem(tree_root, subject, arm, space)
    return {"maps": stem.with_name(stem.name + "_stat.nii.gz"),
            "sidecar": stem.with_name(stem.name + "_stat.json"),
            "parcels": stem.with_name(stem.name + "_parcels.tsv")}


def cell_exists(tree_root: Path, subject: str, arm: str, space: str) -> bool:
    return all(p.exists() for p in cell_paths(tree_root, subject, arm, space).values())


def is_current(tree_root: Path, subject: str, arm: str, space: str, keys: dict, atlases_sha: str) -> bool:
    if not cell_exists(tree_root, subject, arm, space):
        return False
    side = json.loads(cell_paths(tree_root, subject, arm, space)["sidecar"].read_text())
    return (side.get("schema_version") == SCHEMA_VERSION and side.get("fit_keys") == keys
            and side.get("input_atlases_sha256") == atlases_sha)


def fit_runs(trial_info: pd.DataFrame) -> list[tuple[str, str, str]]:
    """``(session, task, run)`` bare labels of every run in the fit, in trial order."""
    seen: dict[tuple[str, str, str], None] = {}
    for r in trial_info[["session", "task", "run"]].astype(str).itertuples(index=False):
        seen[(r.session.removeprefix("ses-"), r.task, r.run.removeprefix("run-"))] = None
    return list(seen)


def load_beta_dict(fdir: Path, beta_type: str) -> dict:
    p = Path(fdir) / "glmsingle_outputs" / BETA_FILES[beta_type]
    return np.load(str(p), allow_pickle=True).item()


def _need(d: dict, key: str, beta_type: str) -> Any:
    if key not in d:
        raise KeyError(f"TYPE{beta_type} has no {key!r}; keys are {sorted(d)}")
    return d[key]


# ---------------------------------------------------------------------------
# Tier 1: one cell
# ---------------------------------------------------------------------------

def reduce_beta_type(d: dict, beta_type: str, mask: np.ndarray, trial_info: pd.DataFrame,
                     sets: dict[str, dict[str, Any]], seed_label: str = "",
                     n_null: int = N_NULL) -> dict[str, Any]:
    """Everything tier 1 keeps from one beta dict, restricted to the mask (C order)."""
    betas = _need(d, "betasmd", beta_type)
    if betas.ndim != 4 or betas.shape[:3] != mask.shape:
        raise ValueError(f"TYPE{beta_type} betasmd shape {betas.shape} does not match the mask grid {mask.shape}")
    if betas.shape[-1] != len(trial_info):
        raise ValueError(f"TYPE{beta_type} betasmd has {betas.shape[-1]} trials but trial_info has {len(trial_info)} "
                         "rows; the per-trial layout is violated")
    out: dict[str, Any] = {
        "R2": np.asarray(_need(d, "R2", beta_type), dtype=np.float32)[mask],
        "meanvol": np.asarray(_need(d, "meanvol", beta_type), dtype=np.float32)[mask],
    }
    z = zscore_by_session(betas[mask], trial_info["session"].astype(str).to_numpy())
    block = (trial_info["session"].astype(str) + "/" + trial_info["run"].astype(str)).to_numpy()
    for name, s in sets.items():
        matched, _ = ncsnr(z, s["groups"])
        null = ncsnr_null(z, s["groups"], block, n_null, null_seed(seed_label, beta_type, name))
        out[f"ncsnr_{name}"] = matched
        out[f"ncsnrnull_{name}"] = np.median(null, axis=0).astype(np.float32)
        with np.errstate(invalid="ignore"):
            out[f"exceed_{name}"] = np.isfinite(matched) & (matched > np.nanmax(null, axis=0))
    del z
    if beta_type == "B":
        out["HRFindex"] = np.asarray(_need(d, "HRFindex", beta_type))[mask].astype(np.int16)
    # GLMdenoise's noise pool and PC count belong to TYPEC; TYPED repeats them when present.
    if beta_type == "C" or (beta_type == "D" and "noisepool" in d and "pcnum" in d):
        pool = np.asarray(_need(d, "noisepool", beta_type)).astype(bool)
        if pool.shape != mask.shape:
            raise ValueError(f"TYPE{beta_type} noisepool shape {pool.shape} is not the grid {mask.shape}")
        out["noisepool"] = pool[mask]
        out["noisepool_n_total"] = int(pool.sum())
        out["noisepool_n_outside_mask"] = int((pool & ~mask).sum())
        out["pcnum"] = int(np.asarray(_need(d, "pcnum", beta_type)))
    if beta_type == "D":
        out["FRACvalue"] = np.asarray(_need(d, "FRACvalue", beta_type), dtype=np.float32)[mask]
    return out


def summaries(red: dict[str, Any], keep: np.ndarray, sets: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Whole-mask summaries of one beta type over the floored voxels."""
    row: dict[str, float] = {}
    r2 = red["R2"][keep]
    row.update(r2_median=_q(r2, .5), r2_p90=_q(r2, .9), r2_p99=_q(r2, .99),
               frac_r2_gt0=float(np.mean(r2[np.isfinite(r2)] > 0)) if np.isfinite(r2).any() else float("nan"))
    for name, s in sets.items():
        snr = red[f"ncsnr_{name}"][keep]
        null = red[f"ncsnrnull_{name}"][keep]
        row[f"{name}_frac_exceed"] = float(red[f"exceed_{name}"][keep].mean())
        row[f"{name}_null_ncsnr_median"] = _q(null, .5)
        row[f"{name}_excess_median"] = _q(snr - null, .5)
        row[f"{name}_ncsnr_median"] = _q(snr, .5)
        row[f"{name}_ncsnr_p90"] = _q(snr, .9)
        row[f"{name}_ncsnr_p99"] = _q(snr, .99)
        row[f"{name}_frac_ncsnr_gt0"] = float(np.mean(snr[np.isfinite(snr)] > 0)) if np.isfinite(snr).any() else float("nan")
        row[f"{name}_nc1_median"] = _q(noise_ceiling(snr, 1), .5)
        row[f"{name}_ncn_median"] = _q(noise_ceiling(snr, s["n_reps"]), .5)
        row[f"{name}_ncn_p90"] = _q(noise_ceiling(snr, s["n_reps"]), .9)
    if "noisepool" in red:
        row["pcnum"] = red["pcnum"]
        row["noisepool_n_total"] = red["noisepool_n_total"]
        row["noisepool_frac_outside_mask"] = (red["noisepool_n_outside_mask"] / red["noisepool_n_total"]
                                              if red["noisepool_n_total"] else float("nan"))
        row["noisepool_frac_of_floor"] = float(red["noisepool"][keep].mean())
    if "FRACvalue" in red:
        fr = red["FRACvalue"][keep]
        fin = fr[np.isfinite(fr)]
        row.update(frac_median=_q(fr, .5), frac_p10=_q(fr, .1), frac_p90=_q(fr, .9),
                   frac_at_min=float(np.mean(fin <= fin.min() + 1e-6)) if fin.size else float("nan"),
                   frac_min=float(fin.min()) if fin.size else float("nan"))
    return row


def hrf_histogram(hrfindex: np.ndarray) -> list[int]:
    h = np.asarray(hrfindex).astype(int)
    if h.size and (h.min() < 0 or h.max() >= N_HRFS):
        raise ValueError(f"HRFindex outside 0..{N_HRFS - 1}: [{h.min()}, {h.max()}]")
    return np.bincount(h, minlength=N_HRFS).tolist()


def parcel_rows(red: dict[str, Any], parc: dq.Parcellation, sets: dict[str, dict[str, Any]],
                beta_type: str) -> list[dict]:
    rows = []
    labels = parc.labels_in_mask
    for rec in parc.table.itertuples(index=False):
        sel = labels == int(rec.index)
        n_mask = int(sel.sum())
        row = {"beta_type": beta_type, "atlas": parc.name, "parcel": int(rec.index), "name": str(rec.name).strip(),
               "n_voxels_atlas": int(rec.n_voxels_atlas), "n_voxels_mask": n_mask}
        if n_mask:
            keep = meanvol_floor(red["meanvol"][sel])
            row["n_voxels_floor"] = int(keep.sum())
            r2 = red["R2"][sel][keep]
            row["r2_median"], row["r2_p90"] = _q(r2, .5), _q(r2, .9)
            for name, s in sets.items():
                snr = red[f"ncsnr_{name}"][sel][keep]
                row[f"{name}_frac_exceed"] = float(red[f"exceed_{name}"][sel][keep].mean()) if keep.any() else float("nan")
                row[f"{name}_excess_median"] = _q(snr - red[f"ncsnrnull_{name}"][sel][keep], .5)
                row[f"{name}_ncsnr_median"] = _q(snr, .5)
                row[f"{name}_ncsnr_p90"] = _q(snr, .9)
                row[f"{name}_nc1_median"] = _q(noise_ceiling(snr, 1), .5)
                row[f"{name}_ncn_median"] = _q(noise_ceiling(snr, s["n_reps"]), .5)
            if "noisepool" in red:
                row["noisepool_frac"] = float(red["noisepool"][sel][keep].mean()) if keep.any() else float("nan")
            if "FRACvalue" in red:
                row["frac_median"] = _q(red["FRACvalue"][sel][keep], .5)
        else:
            row["n_voxels_floor"] = 0
        rows.append(row)
    return rows


def write_cell(tree_root: Path, subject: str, arm: str, space: str, fdir: Path, mask: np.ndarray,
               affine: np.ndarray, atlases_dir: Path, keys: dict, provenance: dict,
               log=print) -> dict:
    """Load the three beta dicts one at a time, reduce, and write the cell. Returns the sidecar."""
    import gc
    import time

    import nibabel as nib

    ti = pd.read_csv(Path(fdir) / "trial_info.csv")
    sets = condition_sets(ti)
    parcs = [dq.load_parcellation(name, atlases_dir, mask, affine) for name in dq.PARCELLATIONS]
    volumes: dict[str, np.ndarray] = {}
    per_type: dict[str, dict] = {}
    parcel_table: list[dict] = []
    hist = None
    meanvol = None
    for bt in BETA_FILES:
        t0 = time.time()
        d = load_beta_dict(fdir, bt)
        red = reduce_beta_type(d, bt, mask, ti, sets, seed_label=f"sub-{subject}|{arm}")
        del d
        gc.collect()
        if meanvol is None:
            meanvol = red["meanvol"]
            keep = meanvol_floor(meanvol)
        elif not np.allclose(red["meanvol"], meanvol, equal_nan=True, rtol=1e-4):
            raise ValueError(f"TYPE{bt} meanvol differs from TYPEB's; the three types should share one")
        per_type[bt] = summaries(red, keep, sets)
        for p in parcs:
            parcel_table.extend(parcel_rows(red, p, sets, bt))
        volumes[f"R2_{bt}"] = red["R2"]
        for name in sets:
            volumes[f"ncsnr_{name}_{bt}"] = red[f"ncsnr_{name}"]
            volumes[f"ncsnrnull_{name}_{bt}"] = red[f"ncsnrnull_{name}"]
            volumes[f"exceed_{name}_{bt}"] = red[f"exceed_{name}"].astype(np.float32)
        if bt == "B":
            hist = hrf_histogram(red["HRFindex"][keep])
            volumes["HRFindex"] = red["HRFindex"].astype(np.float32)
        if bt == "C":
            volumes["noisepool"] = red["noisepool"].astype(np.float32)
        if bt == "D":
            volumes["FRACvalue"] = red["FRACvalue"]
        log(f"sub-{subject} {arm} TYPE{bt}: loaded + reduced in {time.time() - t0:.0f} s; "
            f"R2 med {per_type[bt]['r2_median']:.2f}; repeat frac_exceed {per_type[bt]['repeat_frac_exceed']:.3f} "
            f"(ncsnr med {per_type[bt]['repeat_ncsnr_median']:.3f} vs null {per_type[bt]['repeat_null_ncsnr_median']:.3f}); "
            f"anchor frac_exceed {per_type[bt]['anchor_frac_exceed']:.3f}")
    volumes["meanvol"] = meanvol
    volumes["floor"] = keep.astype(np.float32)

    paths = cell_paths(tree_root, subject, arm, space)
    paths["maps"].parent.mkdir(parents=True, exist_ok=True)
    names = list(volumes)
    data = np.full(mask.shape + (len(names),), np.nan, dtype=np.float32)
    for i, n in enumerate(names):
        data[..., i][mask] = volumes[n]
    nib.save(nib.Nifti1Image(data, affine), str(paths["maps"]))
    pd.DataFrame(parcel_table).to_csv(paths["parcels"], sep="\t", index=False, na_rep="n/a", float_format="%.6g")
    sidecar = {
        "schema_version": SCHEMA_VERSION, "regime": REGIME, "subject": subject, "arm": arm, "space": space,
        "volumes": names,
        "n_trials": int(len(ti)), "n_sessions": int(ti["session"].nunique()),
        "n_voxels_mask": int(mask.sum()), "n_voxels_floor": int(keep.sum()),
        "meanvol_floor_fraction": MEANVOL_FLOOR_FRACTION,
        "n_null": N_NULL, "null": "later presentations permuted across items within their run",
        "condition_sets": {k: {kk: vv for kk, vv in v.items() if kk != "groups"} for k, v in sets.items()},
        "hrfindex_histogram": hist,
        "beta_types": per_type,
        "fit_dir": str(fdir), "fit_keys": keys,
        **provenance,
    }
    paths["sidecar"].write_text(json.dumps(sidecar, indent=2, default=float) + "\n")
    return sidecar


def collect(tree_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten every cell into (subject x arm x beta type) and parcel tables."""
    rows, parcels = [], []
    for js in sorted(Path(tree_root).glob(f"sub-*/func/*_desc-{REGIME}_stat.json")):
        side = json.loads(js.read_text())
        base = {"sub": side["subject"], "arm": side["arm"], "regime": REGIME}
        for bt, summ in side["beta_types"].items():
            row = dict(base, beta_type=bt, n_trials=side["n_trials"], n_sessions=side["n_sessions"],
                       n_voxels_mask=side["n_voxels_mask"], n_voxels_floor=side["n_voxels_floor"])
            for name, s in side["condition_sets"].items():
                row[f"{name}_n_conditions"] = s["n_conditions"]
                row[f"{name}_n_reps"] = s["n_reps"]
                row[f"{name}_rep_scope"] = s["rep_scope"]
                row[f"{name}_null_block_median"] = s["null_block_median"]
            row.update(summ)
            h = np.asarray(side["hrfindex_histogram"], dtype=float)
            p = h / h.sum() if h.sum() else h
            row["hrf_mode"] = int(np.argmax(h))
            row["hrf_mode_frac"] = float(p.max()) if h.sum() else float("nan")
            row["hrf_entropy_bits"] = float(-(p[p > 0] * np.log2(p[p > 0])).sum()) if h.sum() else float("nan")
            row.update(schema_version=side["schema_version"], code_version=side.get("code_version"),
                       fmriprep_version=side.get("fmriprep_version"))
            rows.append(row)
        pt = pd.read_csv(js.with_name(js.name.replace("_stat.json", "_parcels.tsv")), sep="\t", na_values=["n/a"])
        pt.insert(0, "arm", side["arm"])
        pt.insert(0, "sub", side["subject"])
        parcels.append(pt)
    return pd.DataFrame(rows), (pd.concat(parcels, ignore_index=True) if parcels else pd.DataFrame())


# ---------------------------------------------------------------------------
# Tier 2
# ---------------------------------------------------------------------------

TABLES = ("fit_summary", "networks", "benchmark_6cell")
_SCHAEFER_NET = re.compile(r"^17Networks_(LH|RH)_([A-Za-z]+)_")


def region_of(atlas: str, name: str) -> str:
    """Schaefer parcel -> ``<hemi>_<network>`` (e.g. ``LH_VisCent``); HOSPA structure -> its name."""
    if atlas.startswith("Schaefer"):
        m = _SCHAEFER_NET.match(name)
        if not m:
            raise ValueError(f"Schaefer parcel name {name!r} does not parse as 17Networks_<hemi>_<network>_...")
        return f"{m.group(1)}_{m.group(2)}"
    return name.strip()


def load_tables(tree_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = []
    for name in (TABLE_NAME, PARCELS_TABLE_NAME):
        p = Path(tree_root) / f"{name}.tsv"
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing; run `tier1.py glmsingle` for every fit, then `tier1.py collect`")
        out.append(pd.read_csv(p, sep="\t", na_values=["n/a"], dtype={"sub": str}))
    return out[0], out[1]


def fit_summary(fits: pd.DataFrame, expected_subjects: Iterable[str], provisional: Iterable[str] = ()) -> pd.DataFrame:
    """Tier-1 rows plus one declared-absent row per expected subject x arm with no fit."""
    fits = fits.copy()
    fits["absent"] = False
    fits["absent_reason"] = None
    have = set(zip(fits["sub"], fits["arm"]))
    extra = [{"sub": s, "arm": a, "regime": REGIME, "absent": True,
              "absent_reason": f"no {FIT_TREE} fit"}
             for s in sorted(set(expected_subjects)) for a in ARMS if (s, a) not in have]
    out = pd.concat([fits, pd.DataFrame(extra)], ignore_index=True) if extra else fits
    out["provisional"] = out["sub"].isin(set(provisional))
    out["beta_type"] = out["beta_type"].fillna("")
    lead = ["sub", "arm", "beta_type", "regime", "absent", "absent_reason", "provisional"]
    return out[lead + [c for c in out.columns if c not in lead]].sort_values(["sub", "arm", "beta_type"]).reset_index(drop=True)


def networks(parcels: pd.DataFrame) -> pd.DataFrame:
    """Median over parcels of each parcel-median measure, per subject x arm x beta type x region."""
    p = parcels.copy()
    p["region"] = [region_of(a, n) for a, n in zip(p["atlas"], p["name"])]
    measures = [c for c in p.columns if c.endswith(("_median", "_frac_exceed")) or c in ("noisepool_frac",)]
    g = p.groupby(["sub", "arm", "beta_type", "atlas", "region"], sort=True)
    out = g[measures].median()
    out.insert(0, "n_voxels_floor", g["n_voxels_floor"].sum())
    out.insert(0, "n_parcels", g.size())
    return out.reset_index()


def benchmark_6cell(benchmark_root: Path, subjects: Iterable[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    """The settled 6-cell tables, concatenated (T2.10 adopt). Returns the table and each source's sha256."""
    frames, shas = [], {}
    for s in sorted(set(subjects)):
        p = Path(benchmark_root) / f"sub-{s}" / f"sub-{s}_6cell.tsv"
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing: a subject with a GLMsingle fit has no 6-cell benchmark table")
        frames.append(pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False))
        shas[str(p)] = dq.file_sha256(p)
    return pd.concat(frames, ignore_index=True), shas


def out_dir(tree_root: Path) -> Path:
    from .data_quality_tier2 import TIER2_DIR
    return Path(tree_root) / TIER2_DIR / PART


def write(tables: dict[str, pd.DataFrame], dest: Path, provenance: dict) -> list[Path]:
    """Every table in a fixed float format, so a rebuild is byte-identical."""
    from .data_quality_tier2 import FLOAT_FORMAT, SCHEMA_VERSION as T2_SCHEMA

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for name in TABLES:
        path = dest / f"{name}.tsv"
        fmt = None if name == "benchmark_6cell" else FLOAT_FORMAT  # the join is copied verbatim
        tables[name].to_csv(path, sep="\t", index=False, na_rep="n/a", float_format=fmt)
        written.append(path)
    prov = dict(provenance, schema_version=T2_SCHEMA,
                parameters={"meanvol_floor_fraction": MEANVOL_FLOOR_FRACTION, "arms": list(ARMS), "n_null": N_NULL,
                            "networks": "median over parcels of parcel medians; Schaefer by hemisphere x "
                                        "17-network, HOSPA by structure"})
    path = dest / "provenance.json"
    path.write_text(json.dumps(prov, indent=2, default=str) + "\n")
    written.append(path)
    return written


def diff(a: Path, b: Path) -> list[str]:
    """Differences between two glmsingle trees; empty means identical."""
    a, b = Path(a), Path(b)
    problems = []
    for name in TABLES:
        pa, pb = a / f"{name}.tsv", b / f"{name}.tsv"
        if not (pa.exists() and pb.exists()):
            problems.append(f"{name}.tsv missing in {'a' if not pa.exists() else 'b'}")
        elif dq.file_sha256(pa) != dq.file_sha256(pb):
            problems.append(f"{name}.tsv differs")
    return problems

