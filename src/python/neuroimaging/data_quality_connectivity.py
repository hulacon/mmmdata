"""Tier 2 of the data-quality collection: connectivity measures pooled from tier-1 caches.

Reads only the tier-1 parcel series of the rest-like runs, the tier-1 motion
table (``tier1_motion.tsv``, registry T1.7) and the Schaefer atlas (for parcel
centroids) -- never voxels. Design record: mmmdata-agents
``docs/archive/workbench/data-quality/`` (log 2026-09-29 for the three decisions below).

Measures, per confound regime:

* **FC** (registry T2.5, input): Pearson r between every pair of Schaefer
  parcels over a run's steady-state volumes, Fisher-z. A parcel with any
  non-finite value there is n/a for the run (never imputed); non-steady-state
  volumes are all-n/a rows in the cache and are dropped.
* **QC-FC** (T2.5): across runs, the correlation of each edge's FC with the
  run's mean FD, **within subject**: FC and FD are centred on the subject's
  mean before correlating, so a subject who both moves more and has different
  FC does not read as a motion effect (DECIDED 2026-09-29, the analogue of
  Parkes et al. 2018's age/sex partial correlation). p-values use
  ``n_runs - n_subjects - 1`` degrees of freedom. Summaries: median |QC-FC|,
  fraction of edges with p < .05, and the **distance dependence**, the Spearman
  rho between QC-FC and the Euclidean distance between parcel centroids (Ciric
  et al. 2017). Computed for raw and respiration-filtered FD.
* **Hippocampal FC profile** (T2.6): Fisher-z r of a hippocampus seed (HOSPA
  left, right, and both pooled voxel-weighted, as the colleague's panel) with
  every Schaefer parcel, averaged over the subject's rest runs; and the
  colleague's statistic, the Pearson r between two regimes' profiles.
* **Fingerprinting** (T2.7), Finn et al. 2015: a run's FC vector is matched to
  the most correlated other run.
  - ``subject``: identified correctly when that run is the same subject's.
    With few subjects the rate saturates; the margin (best same-subject r
    minus best other-subject r) is the informative number.
  - ``session``: each run is split into its first and second halves of
    steady-state volumes (one rest run per session, so a second run does not
    exist; DECIDED 2026-09-29). A half is matched among all of the subject's
    other-half FC vectors; correct when it picks its own run. The halves share
    head position and scanner state, so this is a noise fingerprint and should
    fall as a regime removes more noise.

Pooling scopes: every subject alone (at least ``MIN_RUNS_PER_SUBJECT`` runs),
and ``pooled`` over all subjects. Subjects named provisional (their fMRIPrep is
due a rerun) mark any row that includes them, and add a ``pooled_confirmed``
scope without them.

Outputs, under ``<tree>/tier2/connectivity/``::

    rest_runs.tsv                          the runs, in FC row order, with mean FD
    parcels.tsv                            Schaefer parcels in edge order, centroid (mm)
    fc/seg-<atlas>_desc-<regime>_fc.npy    (runs, edges) float32 Fisher-z, upper triangle
    qcfc.tsv                               regime x FD kind x scope summaries
    fingerprint.tsv                        regime x test x scope: ID rate, chance, margin
    hipp_profile.tsv                       subject x regime x seed x parcel
    hipp_similarity.tsv                    subject x seed x regime pair
    provenance.json
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .data_quality_tier2 import FLOAT_FORMAT, SCHEMA_VERSION, TIER2_DIR, cross_corr, file_sha256, load_series

PART = "connectivity"
SEG = "Schaefer17n400"
SUBCORTICAL = "HOSPA"
REST_TASKS = ("INITresting", "TBresting", "NATresting", "FINresting")
HIPPOCAMPUS = {"left": "Left Hippocampus", "right": "Right Hippocampus"}
#: Fewest runs a subject needs for its own QC-FC or fingerprint row.
MIN_RUNS_PER_SUBJECT = 10
#: Mean-FD column of tier1_motion.tsv per FD kind.
FD_KINDS = {"raw": "fd_mean", "filtered": "fdf_mean"}
SCHAEFER_DSEG = "tpl-MNI152NLin2009cAsym/anat/tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_seg-17n_scale-400_res-2_dseg"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_motion(tree_root: Path) -> pd.DataFrame:
    path = Path(tree_root) / "tier1_motion.tsv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; build it first (`tier1.py motion`)")
    df = pd.read_csv(path, sep="\t", dtype={"sub": str, "ses": str, "run": str}, keep_default_na=False,
                     na_values=["n/a"])
    df["run"] = df["run"].fillna("")
    return df


def rest_runs(tier1_runs: pd.DataFrame, motion: pd.DataFrame, provisional: Iterable[str] = ()) -> pd.DataFrame:
    """One row per rest-like run, sorted, with its mean FD; a run missing from the motion table is an error."""
    runs = (tier1_runs[tier1_runs["task"].isin(REST_TASKS)]
            .drop_duplicates(["sub", "ses", "task", "run"])
            [["sub", "ses", "task", "run", "space", "n_vol", "n_nss"]].copy())
    if runs.empty:
        raise ValueError(f"tier1_runs.tsv has no {REST_TASKS} runs")
    runs["run"] = runs["run"].replace("n/a", "")
    runs["n_vol"] = runs["n_vol"].astype(float).astype(int)
    runs["n_nss"] = runs["n_nss"].astype(float).astype(int)
    keys = ["sub", "ses", "task", "run"]
    out = runs.merge(motion[keys + list(FD_KINDS.values())], on=keys, how="left", validate="one_to_one")
    missing = out[out["fd_mean"].isna()]
    if len(missing):
        raise KeyError(f"{len(missing)} rest runs have no tier1_motion.tsv row, e.g. "
                       f"{missing[keys].iloc[0].to_dict()}; rebuild it (`tier1.py motion`)")
    out["provisional"] = out["sub"].isin(set(provisional))
    return out.sort_values(keys, ignore_index=True)


def series_path(tree_root: Path, r, regime: str, seg: str) -> Path:
    run = f"_run-{r.run}" if r.run else ""
    name = f"sub-{r.sub}_ses-{r.ses}_task-{r.task}{run}_space-{r.space}_seg-{seg}_desc-{regime}_timeseries.tsv"
    return Path(tree_root) / f"sub-{r.sub}" / f"ses-{r.ses}" / "func" / name


def hippocampus_weights(tsv_path: Path) -> dict[str, float]:
    """In-mask voxel counts of the two hippocampus parcels, from the HOSPA series' sidecar."""
    meta = json.loads(Path(str(tsv_path).replace(".tsv", ".json")).read_text())
    parcels = meta["parcels"]
    return {side: float(parcels[name]["n_voxels_mask"]) for side, name in HIPPOCAMPUS.items()}


def parcel_centroids(atlases_dir: Path) -> pd.DataFrame:
    """Schaefer parcels (index order) with their centroid in world mm."""
    import nibabel as nib

    stem = Path(atlases_dir) / SCHAEFER_DSEG
    table = pd.read_csv(f"{stem}.tsv", sep="\t")
    img = nib.load(f"{stem}.nii.gz")
    lab = np.asarray(img.dataobj).astype(int)
    ijk = np.argwhere(lab > 0)
    labels = lab[tuple(ijk.T)]
    xyz = nib.affines.apply_affine(img.affine, ijk)
    rows = []
    for idx, name in zip(table["index"], table["name"]):
        sel = labels == idx
        if not sel.any():
            raise ValueError(f"{stem}.nii.gz has no voxel labelled {idx} ({name})")
        c = xyz[sel].mean(axis=0)
        rows.append({"index": int(idx), "name": str(name), "x": c[0], "y": c[1], "z": c[2]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------

def steady_mask(x: np.ndarray) -> np.ndarray:
    """Rows of a parcel-series array that are not all n/a (i.e. not non-steady-state volumes)."""
    return np.isfinite(np.asarray(x, dtype=float)).any(axis=1)


def fisher_z(r: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.arctanh(r)


def fc_vector(x: np.ndarray) -> np.ndarray:
    """``(N, P)`` steady-state series -> ``(P*(P-1)/2,)`` Fisher-z FC, upper triangle, row-major."""
    r = cross_corr(x, x)
    return fisher_z(r[np.triu_indices(r.shape[0], 1)])


def centre_within(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each group's mean along axis 0 (NaN propagates: a NaN edge stays NaN for its group)."""
    x = np.asarray(x, dtype=float).copy()
    for g in np.unique(groups):
        sel = groups == g
        x[sel] -= x[sel].mean(axis=0)
    return x


def qcfc(fc: np.ndarray, fd: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Within-group QC-FC per edge: ``(r, p, dof)``. ``fc`` is ``(runs, edges)``, ``fd`` ``(runs,)``."""
    from scipy import stats

    fc = centre_within(fc, groups)
    fd = centre_within(np.asarray(fd, float)[:, None], groups)[:, 0]
    dof = len(fd) - len(np.unique(groups)) - 1
    if dof < 2:
        raise ValueError(f"QC-FC needs more runs than subjects + 2; got {len(fd)} runs, {len(np.unique(groups))} subjects")
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (fc * fd[:, None]).sum(axis=0) / np.sqrt((fc ** 2).sum(axis=0) * (fd ** 2).sum())
        t = r * np.sqrt(dof / (1 - r ** 2))
    p = 2 * stats.t.sf(np.abs(t), dof)
    return r, p, dof


def qcfc_summary(r: np.ndarray, p: np.ndarray, distance: np.ndarray) -> dict:
    from scipy import stats

    ok = np.isfinite(r)
    rho = stats.spearmanr(r[ok], distance[ok]).statistic if ok.sum() > 2 else np.nan
    return {"n_edges": int(ok.sum()), "median_abs_qcfc": float(np.median(np.abs(r[ok]))) if ok.any() else np.nan,
            "frac_p05": float((p[ok] < 0.05).mean()) if ok.any() else np.nan, "distance_dependence": float(rho)}


def row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(A, E), (B, E) -> (A, B)`` Pearson r between rows over the edges finite in every row of both."""
    ok = np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)
    return cross_corr(a[:, ok].T, b[:, ok].T)


def identify_subjects(fc: np.ndarray, subjects: np.ndarray) -> pd.DataFrame:
    """Per run: is its most correlated other run the same subject's, and by what margin."""
    c = row_corr(fc, fc)
    np.fill_diagonal(c, -np.inf)
    rows = []
    for i, s in enumerate(subjects):
        same, other = subjects == s, subjects != s
        same[i] = False
        if not same.any() or not other.any():
            continue
        best_same, best_other = c[i, same].max(), c[i, other].max()
        rows.append({"target": i, "sub": s, "correct": bool(best_same > best_other),
                     "margin": float(best_same - best_other)})
    return pd.DataFrame(rows)


def identify_sessions(half_a: np.ndarray, half_b: np.ndarray) -> pd.DataFrame:
    """One subject's runs: is each half's most correlated other-half its own run's (both directions)."""
    c = row_corr(half_a, half_b)
    rows = []
    n = c.shape[0]
    for direction, m in (("a->b", c), ("b->a", c.T)):
        for i in range(n):
            others = np.delete(m[i], i)
            rows.append({"target": i, "direction": direction, "correct": bool(m[i, i] > others.max()),
                         "margin": float(m[i, i] - others.max())})
    return pd.DataFrame(rows)


def profile_similarity(profiles: dict[str, np.ndarray]) -> list[dict]:
    """Pearson r between every pair of regimes' profiles, over parcels finite in both."""
    names = sorted(profiles)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y = profiles[a], profiles[b]
            ok = np.isfinite(x) & np.isfinite(y)
            r = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else np.nan
            out.append({"regime_a": a, "regime_b": b, "n_parcels": int(ok.sum()), "r": r})
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ConnectivityResult:
    runs: pd.DataFrame
    parcels: pd.DataFrame
    fc: dict[str, np.ndarray]
    qcfc: pd.DataFrame
    fingerprint: pd.DataFrame
    hipp_profile: pd.DataFrame
    hipp_similarity: pd.DataFrame
    skipped: list[dict]


def _scopes(runs: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    """``(scope, row mask)``: each subject with enough runs, all subjects, and all confirmed ones."""
    out = []
    for sub, n in runs["sub"].value_counts().sort_index().items():
        if n >= MIN_RUNS_PER_SUBJECT:
            out.append((f"sub-{sub}", (runs["sub"] == sub).to_numpy()))
    out.append(("pooled", np.ones(len(runs), bool)))
    if runs["provisional"].any():
        out.append(("pooled_confirmed", (~runs["provisional"]).to_numpy()))
    return out


def compute(
    tree_root: Path,
    runs: pd.DataFrame,
    regimes: list[str],
    centroids: pd.DataFrame,
    absent: set[tuple],
    rename: Optional[dict[str, str]] = None,
) -> ConnectivityResult:
    """Every measure for every regime; each run's two caches are read once per regime."""
    names = centroids["name"].tolist()
    iu = np.triu_indices(len(names), 1)
    xyz = centroids[["x", "y", "z"]].to_numpy(float)
    distance = np.linalg.norm(xyz[iu[0]] - xyz[iu[1]], axis=1)
    subjects = runs["sub"].to_numpy()
    fc_out, qc_rows, fp_rows, prof_rows, sim_rows, skipped = {}, [], [], [], [], []

    for regime in regimes:
        missing = [r for r in runs.itertuples(index=False)
                   if (r.sub, r.ses, r.task, r.run or "n/a", regime) in absent]
        if missing:
            skipped.append({"regime": regime, "reason": f"{len(missing)} rest runs declared absent under it"})
            continue
        fc = np.empty((len(runs), len(distance)))
        half_a, half_b = np.empty_like(fc), np.empty_like(fc)
        seeds: dict[str, list[np.ndarray]] = {side: [] for side in (*HIPPOCAMPUS, "both")}
        for i, r in enumerate(runs.itertuples(index=False)):
            ts = load_series(series_path(tree_root, r, regime, SEG), r.n_vol, rename)
            if list(ts.columns) != names:
                raise ValueError(f"{series_path(tree_root, r, regime, SEG)}: columns are not the atlas table's, in order")
            steady = steady_mask(ts.to_numpy(float))
            x = ts.to_numpy(float)[steady]
            if len(x) != r.n_vol - r.n_nss:
                raise ValueError(f"sub-{r.sub} ses-{r.ses} {r.task}: {len(x)} steady rows, expected {r.n_vol - r.n_nss}")
            fc[i] = fc_vector(x)
            h = len(x) // 2
            half_a[i], half_b[i] = fc_vector(x[:h]), fc_vector(x[h:2 * h])

            sub_path = series_path(tree_root, r, regime, SUBCORTICAL)
            sc = load_series(sub_path, r.n_vol)
            w = hippocampus_weights(sub_path)
            hip = {side: sc[name].to_numpy(float)[steady] for side, name in HIPPOCAMPUS.items()}
            hip["both"] = (hip["left"] * w["left"] + hip["right"] * w["right"]) / (w["left"] + w["right"])
            for side, y in hip.items():
                seeds[side].append(fisher_z(cross_corr(y[:, None], x)[0]))
        fc_out[regime] = fc.astype(np.float32)

        # QC-FC
        for fd_kind, col in FD_KINDS.items():
            for scope, sel in _scopes(runs):
                r_e, p_e, dof = qcfc(fc[sel], runs.loc[sel, col].to_numpy(float), subjects[sel])
                qc_rows.append({"regime": regime, "fd_kind": fd_kind, "scope": scope,
                                "subjects": ",".join(sorted(set(subjects[sel]))), "n_runs": int(sel.sum()),
                                "dof": dof, **qcfc_summary(r_e, p_e, distance),
                                "provisional": bool(runs.loc[sel, "provisional"].any())})

        # Fingerprinting
        sess_parts = []
        for sub in sorted(set(subjects)):
            sel = subjects == sub
            if sel.sum() < 2:
                continue
            part = identify_sessions(half_a[sel], half_b[sel])
            part["sub"], part["chance"] = sub, 1.0 / sel.sum()
            sess_parts.append(part)
        sess = pd.concat(sess_parts, ignore_index=True)
        for scope, sel in _scopes(runs):
            in_scope = set(subjects[sel])
            prov = bool(runs.loc[sel, "provisional"].any())
            s = identify_subjects(fc[sel], subjects[sel]) if len(in_scope) > 1 else pd.DataFrame(
                columns=["correct", "margin"])
            fp_rows.append({"regime": regime, "test": "subject", "scope": scope,
                            "subjects": ",".join(sorted(in_scope)), "n_targets": len(s),
                            "chance": np.nan, "id_rate": s["correct"].mean(),
                            "margin_median": s["margin"].median(), "provisional": prov})
            e = sess[sess["sub"].isin(in_scope)]
            fp_rows.append({"regime": regime, "test": "session", "scope": scope,
                            "subjects": ",".join(sorted(in_scope)), "n_targets": len(e),
                            "chance": e["chance"].mean(), "id_rate": e["correct"].mean(),
                            "margin_median": e["margin"].median(), "provisional": prov})

        # Hippocampal FC profile
        for side, z in seeds.items():
            z = np.stack(z)
            for sub in sorted(set(subjects)):
                sel = subjects == sub
                mean = z[sel].mean(axis=0)  # a parcel n/a in any run stays n/a
                for name, value in zip(names, mean):
                    prof_rows.append({"sub": sub, "regime": regime, "seed": side, "parcel": name,
                                      "z_mean": value, "n_runs": int(sel.sum())})

    prof = pd.DataFrame(prof_rows)
    for (sub, side), g in (prof.groupby(["sub", "seed"], sort=True) if len(prof) else []):
        profiles = {reg: gg["z_mean"].to_numpy(float) for reg, gg in g.groupby("regime")}
        for row in profile_similarity(profiles):
            sim_rows.append({"sub": sub, "seed": side, **row})

    parcels = centroids.copy()
    return ConnectivityResult(
        runs=runs, parcels=parcels, fc=fc_out, qcfc=pd.DataFrame(qc_rows), fingerprint=pd.DataFrame(fp_rows),
        hipp_profile=prof, hipp_similarity=pd.DataFrame(sim_rows), skipped=skipped,
    )


def out_dir(tree_root: Path) -> Path:
    return Path(tree_root) / TIER2_DIR / PART


TABLES = ("rest_runs", "parcels", "qcfc", "fingerprint", "hipp_profile", "hipp_similarity")


def write(result: ConnectivityResult, dest: Path, provenance: dict) -> list[Path]:
    """Every table (fixed float format) and FC array, so a rebuild is byte-identical."""
    dest = Path(dest)
    (dest / "fc").mkdir(parents=True, exist_ok=True)
    frames = {"rest_runs": result.runs, "parcels": result.parcels, "qcfc": result.qcfc,
              "fingerprint": result.fingerprint, "hipp_profile": result.hipp_profile,
              "hipp_similarity": result.hipp_similarity}
    written = []
    for name, df in frames.items():
        path = dest / f"{name}.tsv"
        df.to_csv(path, sep="\t", index=False, na_rep="n/a", float_format=FLOAT_FORMAT)
        written.append(path)
    for regime, arr in sorted(result.fc.items()):
        path = dest / "fc" / f"seg-{SEG}_desc-{regime}_fc.npy"
        np.save(path, arr)
        written.append(path)
    prov = dict(provenance, schema_version=SCHEMA_VERSION, skipped=result.skipped,
                parameters={"seg": SEG, "subcortical": SUBCORTICAL, "rest_tasks": list(REST_TASKS),
                            "hippocampus": HIPPOCAMPUS, "min_runs_per_subject": MIN_RUNS_PER_SUBJECT,
                            "fd_kinds": FD_KINDS})
    path = dest / "provenance.json"
    path.write_text(json.dumps(prov, indent=2, default=str) + "\n")
    written.append(path)
    return written


def diff(a: Path, b: Path) -> list[str]:
    """Differences between two connectivity trees (every table and FC array); empty means identical."""
    a, b = Path(a), Path(b)
    problems = []
    for name in TABLES:
        pa, pb = a / f"{name}.tsv", b / f"{name}.tsv"
        if not (pa.exists() and pb.exists()):
            problems.append(f"{name}.tsv missing in {'a' if not pa.exists() else 'b'}")
        elif file_sha256(pa) != file_sha256(pb):
            problems.append(f"{name}.tsv differs")
    ma = sorted(p.name for p in (a / "fc").glob("*.npy"))
    mb = sorted(p.name for p in (b / "fc").glob("*.npy"))
    if ma != mb:
        problems.append(f"fc/ file sets differ ({len(ma)} vs {len(mb)})")
    for name in sorted(set(ma) & set(mb)):
        if not np.array_equal(np.load(a / "fc" / name), np.load(b / "fc" / name), equal_nan=True):
            problems.append(f"fc/{name} differs")
    return problems
