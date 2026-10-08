#!/usr/bin/env python3
"""Score the functional-space routes on the held-out test data, and decide H1/H2.

Pre-registration §9 (metrics, inference), §3.2 (test data), §6 (routes);
scoring choices DECIDED 2026-09-30 in mmmdata-agents
``docs/archive/workbench/functional-space/``. The metrics themselves are
``scoring.py``'s; this module loads data, builds each job's models and runs
them.

Models, per partition job. Each maps the template subjects' held-out data
into the target's space (``scoring.carry``: ``x_s[:, cols_s] A_s A_T'`` per
piece), and the template side of every metric is their mean:

  anatomical            fsaverage6 vertex identity (§6)
  combined              the stacked model at the job's tuned (w, λ)
  combined-minus-<b>    leave-one-route-out (three-block jobs only)
  cha, stimulus-<space>, response
                        the single routes, i.e. the tuned one-block faces
  stimulus-vgg19        tuned alone (λ only), on its own Grams
  srm, pca              the SRM comparator and its PCA control at the tuned k (s > 0)

Faces (DECIDED #1). Every face of the combined job's tuning table (the full
model, leave-one-route-out, the single routes) is refitted in memory from
the routes' Grams at its weights, on the combined model's common columns,
and taken at its tuned λ. The full model must reproduce the combined job's
saved crosses; a mismatch is an error. So the Grams must still be on disk.

Columns (DECIDED #2). Per job and film set, every model is scored on one
column set: the columns finite in the target's data and in every model's
projection over every film.

Arms (``--arm`` on ``score`` and ``m4``). ``ebind`` (default) is the
pre-registered run above. ``psytwill`` (DECIDED 2026-10-07) is the later arm,
psytwill-space in place of EBind: anatomical, every face of the
combined-psytwill job (its combined faces named ``combined-psytwill*``), and
the combined-EBind job's full model and stimulus face as ``combined-ebind``
and ``stimulus-ebind``, all on one column set so psytwill and EBind pair on
identical columns. No VGG19 or SRM/PCA. Written under
``scores/psytwill-arm/``; M4 there covers psytwill only.

Film sets (§3.2): ``heldout``, the 12 unique films of ses-23 and ses-26
(primary); ``repeat``, the repeated films' held-out showings, scored
separately. Every subject's window is paired to the target's grid by nearest
film time (``films.paired_slices``) and z-scored per column before
projection.

Metrics here: M2a (per film), M2b (per segment), M3 (TB items; the 294 images
with three exposures in every subject, all-item and triplet-free foil pools,
DECIDED 2026-09-30) and M1 (localizer maps: fLoc t, motor between-effector
effect, pRF angle from projected Cartesian components). M4 is a GPU step of
its own (verb ``m4``). Floors (TB meanvol/|beta|, pRF R^2 and radius) decide which columns
are SCORED; projection inputs are never blanked, since one NaN column would
spoil its whole piece.

Verbs:

  score    one job -> <derivatives>/functional_space/scores/<scenario>/pct-<pct>/
           draw-<draw>/target-<sub>/ (m2b.parquet, m2a.tsv, m3.tsv, m1.tsv, score.json)
  m4       one 0% job, on a GPU: each subject's encoders refitted, their predictions
           removed from its own held-out series, M2a recomputed (raw and residual)
           for anatomical and each stimulus route -> m4.tsv, m4.json beside the scores
  mni      the MNI voxel-identity baseline for one target (job-independent) ->
           scores/anatomical-mni/target-<sub>/
  decide   H1, then H2 only if H1 is a go, from the primary 0% scores ->
           scores/decision.json
  families one job's §10 secondary families and §6 smoothed-anatomical control ->
           <the job's score dir>/families/ (m2b.parquet, m2a.tsv, m3.tsv, m1.tsv, smoothing.tsv,
           families.json); needs `families.py cache-tb` for every subject first

Usage:
    python score_route.py score --pct 0 --draw 0 --target 03 --n-jobs 16
    python score_route.py m4 --pct 0 --draw 0 --target 03
    python score_route.py score --arm psytwill --pct 0 --draw 0 --target 03 --n-jobs 16
    python score_route.py mni --target 03
    python score_route.py decide
    python score_route.py families --pct 0 --draw 0 --target 03 --n-jobs 16
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "src" / "python", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import procrustes as pr  # noqa: E402
import scoring as sc  # noqa: E402

FAMILY_A = "schaefer7n"
FILM_SETS = {"heldout": "heldout", "heldout_repeat": "repeat"}  # window role -> film set
PRIMARY_SET = "heldout"
H1_JOBS = ("primary", 0)  # scenario, level
SPACE = "ebind"  # the combined model's stimulus block (§6)
MNI_STEM = ("space-fsaverage6", "space-MNI152NLin2009cAsym_res-2")
FULL_REPRO_RTOL = 1e-6
#: Where M1 is read, per family-A network (M1 floor DECIDED 2026-09-29: ceiling >= .4).
M1_COMPONENTS = {"Vis": ("floc", "motor", "prf"), "SomMot": ("motor",), "DorsAttn": ("floc", "motor"),
                 "SalVentAttn": ("motor",), "Cont": ("motor",)}
#: Motor's between-effector contrasts with the mouth contrasts excluded (DECIDED 2026-09-29).
MOTOR_CONTRASTS = ("handVsFootDerived", "saccadeVsOthersDerived")
PRF_R2_FLOOR = 10.0  # percent (DECIDED 2026-09-29)
#: TB beta floor, as retrieval_modeling/benchmark_6cell.py:clean_voxels (meanvol-floor reference).
MEANVOL_FRAC, BETA_CAP = 0.25, 100.0


#: Scoring arms. ``ebind`` is the pre-registered run (§6, combined = CHA + EBind + response). ``psytwill`` is
#: the later arm (DECIDED 2026-10-07): psytwill-space replaces EBind in the combined model, and EBind's
#: combined model and stimulus route are scored beside it on the same columns; its own tree.
ARMS = ("ebind", "psytwill")
M4_SPACES_BY_ARM = {"ebind": ("ebind", "vgg19"), "psytwill": ("psytwill",)}


def scores_root(derivatives: Path, arm: str = "ebind") -> Path:
    root = Path(derivatives) / "functional_space" / "scores"
    return root if arm == "ebind" else root / f"{arm}-arm"


def out_dir(derivatives: Path, scenario: str, pct: int, draw: int, target: str, arm: str = "ebind") -> Path:
    return scores_root(derivatives, arm) / scenario / f"pct-{pct:03d}" / f"draw-{draw:02d}" / f"target-{target}"


def mni_dir(derivatives: Path, target: str) -> Path:
    return scores_root(derivatives) / "anatomical-mni" / f"target-{target}"


# ---------------------------------------------------------------------------
# test data
# ---------------------------------------------------------------------------

def network_labels(table: pd.DataFrame, atlases: Path) -> np.ndarray:
    """Family-A network per grayordinate ('' outside the 7 Schaefer networks)."""
    import grayordinates as go
    import localizer_ceiling as lc

    cortex = np.flatnonzero(table["piece"].to_numpy() == "cortex")
    want = np.concatenate([np.arange(go.FSAVERAGE6_N)] * 2)
    hemi = np.repeat(["L", "R"], go.FSAVERAGE6_N)
    if (not np.array_equal(cortex, np.arange(want.size))
            or not np.array_equal(table.loc[cortex, "vertex"].to_numpy(), want)
            or not np.array_equal(table.loc[cortex, "hemi"].to_numpy(), hemi)):
        raise ValueError("grayordinate cortex rows are not fsaverage6 L then R vertices in order")
    lab = np.full(len(table), "", dtype=object)
    for (fam, net), mask in lc.load_rois(Path(atlases)).items():
        if fam == FAMILY_A:
            lab[cortex[mask]] = net
    return lab


def mni_network_labels(voxels: pd.DataFrame) -> np.ndarray:
    """Family-A network per MNI voxel, from the MNI Schaefer 7-network atlas (§6)."""
    names = voxels["schaefer7n"].fillna("").astype(str).to_numpy()
    return np.array([n.split("_")[2] if n.startswith("7Networks_") else "" for n in names], dtype=object)


def film_rows(windows: pd.DataFrame, subs: list[str], role: str) -> dict[str, pd.DataFrame]:
    """Each subject's windows of one role, indexed by film key (the same keys for every subject)."""
    out = {}
    for s in subs:
        r = windows[(windows["sub"] == s) & (windows["role"] == role)].copy()
        r["key"] = r["stimulus_id"] if role == "heldout" else r["stimulus_id"] + "@" + r["showing"].astype(str)
        r = r.set_index("key")
        if r.index.duplicated().any():
            raise ValueError(f"sub-{s}: duplicate {role} film keys")
        out[s] = r
    keys = sorted(out[subs[0]].index)
    if not keys or any(sorted(v.index) != keys for v in out.values()):
        raise ValueError(f"{role}: subjects' film keys differ or are empty")
    return out


def load_films(rows: dict[str, pd.DataFrame], subs: list[str], target: str, read) -> dict[str, dict[str, np.ndarray]]:
    """Every subject's paired, per-column z-scored window of each film, on the target's grid.

    ``read(row, cache)`` returns one showing's full window (``films.film_series``
    or its MNI counterpart). Subjects are loaded one at a time so only one
    subject's runs sit in the cache.
    """
    import films as fm

    keys = sorted(rows[target].index)
    slices = {k: fm.paired_slices({s: rows[s].loc[k] for s in subs}, target) for k in keys}
    out = {}
    for s in subs:
        cache: dict = {}
        out[s] = {k: sc.zscore_columns(read(rows[s].loc[k], cache)[slices[k][s]]).astype(np.float32) for k in keys}
        cache.clear()
    return out


def mni_reader(cleaned: Path, n_voxels: int):
    """``read(row, cache)`` for the MNI series of a window row (same runs, same volumes)."""
    import nibabel as nib

    def read(row, cache):
        rel = str(row.series)
        if MNI_STEM[0] not in rel:
            raise ValueError(f"{rel}: not an fsaverage6 series path")
        path = Path(cleaned) / rel.replace(*MNI_STEM)
        if path not in cache:
            data = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
            if data.shape[1] != n_voxels:
                raise ValueError(f"{path.name}: {data.shape[1]} voxels, index has {n_voxels}")
            cache[path] = data
        x = cache[path][int(row.start): int(row.start) + int(row.n)]
        if np.isnan(x).all(axis=1).any():
            raise ValueError(f"{path.name}: window {row.start}+{row.n} contains a non-steady-state row")
        return x

    return read


def fs6_columns(v: np.ndarray, n: int) -> np.ndarray:
    """An L+R fsaverage6 vertex vector as a grayordinate row (cortex columns come first, in order)."""
    import grayordinates as go

    if v.shape != (2 * go.FSAVERAGE6_N,):
        raise ValueError(f"expected {2 * go.FSAVERAGE6_N} fsaverage6 values, got {v.shape}")
    out = np.full(n, np.nan, dtype=np.float32)
    out[: v.size] = v
    return out


def load_items(derivatives: Path, subs: list[str], n: int, networks: np.ndarray
               ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str], np.ndarray]:
    """M3 inputs (deviation DECIDED 2026-09-30): the images with exactly three exposures in every subject.

    Returns per subject the item patterns (items x columns: GLMsingle TYPED
    ``betasmd`` averaged over the three exposures, then z-scored per column
    across items) and the floor (bool per column: finite in every trial,
    meanvol >= MEANVOL_FRAC x its network's median, median |beta| <= BETA_CAP);
    the item ids; and the triplet-free foil pool (not enCon 3 in any subject).
    """
    patterns, base, meanvol, items, foils = item_inputs(derivatives, subs, n)
    return patterns, {s: item_floor(base[s], meanvol[s], networks) for s in subs}, items, foils


def item_inputs(derivatives: Path, subs: list[str], n: int, exposures: int = 3, subcortex: pd.DataFrame | None = None
                ) -> tuple[dict, dict, dict, list[str], np.ndarray]:
    """Per subject the item patterns, the label-free floor (finite, |beta| cap) and meanvol, per column; the item
    ids (exactly ``exposures`` exposures in every subject) and the triplet-free foil pool.

    Cortex columns come from the fsaverage6 refit. With ``subcortex`` (the
    grayordinate table), the subcortical columns come from the cached MNI
    res-2 fit (``families.load_tb_subcortex``), whose trials are the same.
    """
    import grayordinates as go

    root = Path(derivatives) / "functional_space" / "glmsingle_tb_fsaverage6"
    trials = {}
    for s in subs:
        t = pd.read_csv(root / f"sub-{s}" / "enc" / "trial_info.csv", dtype={"mmmId": str})
        if not t.groupby(["session", "run"], sort=False)["onset"].apply(lambda o: o.is_monotonic_increasing).all():
            raise ValueError(f"sub-{s}: trial_info onsets are not increasing within runs (betas are in time order)")
        trials[s] = t
    have = [set(c[c == exposures].index) for c in (t.groupby("mmmId").size() for t in trials.values())]
    items = sorted(set.intersection(*have))
    triplet = set().union(*(set(t.loc[t["enCon"] == 3, "mmmId"]) for t in trials.values()))
    foils = np.array([i not in triplet for i in items])
    patterns, base, meanvol = {}, {}, {}
    for s in subs:
        d = root / f"sub-{s}" / "enc"
        fit = np.load(d / "glmsingle_outputs" / "TYPED_FITHRF_GLMDENOISE_RR.npy", allow_pickle=True).item()
        betas = fit["betasmd"].reshape(fit["betasmd"].shape[0], -1)
        mv = fit["meanvol"].reshape(-1)
        if betas.shape[1] != len(trials[s]):
            raise ValueError(f"sub-{s}: {betas.shape[1]} betas for {len(trials[s])} trials")
        vi = pd.read_csv(d / "vertex_index.tsv", sep="\t")
        if not np.array_equal(vi["row"].to_numpy(), np.arange(len(vi))) or len(vi) != betas.shape[0]:
            raise ValueError(f"sub-{s}: vertex_index does not match the beta rows")
        cols = np.where(vi["hemi"].to_numpy() == "R", go.FSAVERAGE6_N, 0) + vi["vertex"].to_numpy()
        del fit
        blocks = [(cols, betas, mv)]
        if subcortex is not None:
            import families as fam

            if not trials[s].equals(pd.read_csv(fam.tb_cache_dir(derivatives, s) / "trial_info.csv",
                                                dtype={"mmmId": str})):
                raise ValueError(f"sub-{s}: the subcortical cache's trials differ from the fsaverage6 fit's")
            blocks.append(fam.load_tb_subcortex(derivatives, s, subcortex))
        ids = trials[s]["mmmId"].to_numpy()
        x = np.full((len(items), n), np.nan, dtype=np.float32)
        ok = np.zeros(n, dtype=bool)
        mvol = np.full(n, np.nan, dtype=np.float64)
        for c, b, m in blocks:
            x[:, c] = np.stack([b[:, ids == i].mean(axis=1) for i in items])
            good = np.isfinite(b).all(axis=1)
            with np.errstate(invalid="ignore"):
                good &= np.nanmedian(np.abs(b), axis=1) <= BETA_CAP
            ok[c] = good
            mvol[c] = m
        patterns[s] = sc.zscore_columns(x).astype(np.float32)
        base[s], meanvol[s] = ok, mvol
        del betas, blocks
    return patterns, base, meanvol, items, foils


def item_floor(base: np.ndarray, meanvol: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The M3 floor for one label set: ``base`` and meanvol >= MEANVOL_FRAC x the median of the column's label."""
    ok = base.copy()
    for name in sorted(set(labels) - {""}):
        sel = labels == name
        ok[sel] &= meanvol[sel] >= MEANVOL_FRAC * np.nanmedian(meanvol[sel])
    return ok


def load_maps(derivatives: Path, subs: list[str], n: int) -> tuple[dict[str, np.ndarray], list[tuple], dict]:
    """M1 inputs: per subject a (maps x columns) array; the row keys; each subject's pRF keep mask.

    Rows: fLoc full-data t maps (all contrasts), motor between-effector effect
    maps (MOTOR_CONTRASTS, the mean over runs of each run's derived effect,
    as the ceiling built them), and the pRF centre as Cartesian components
    (x, y), projected as such (§9).
    """
    import nibabel as nib
    import localizer_ceiling as lc

    tree = Path(derivatives) / "functional_space" / "localizer_splithalf"
    prf_root = Path(derivatives) / "functional_space" / "prf_splithalf"
    rows_by_sub, keys_by_sub, keep = {}, {}, {}
    for s in subs:
        func = tree / f"sub-{s}" / "func"
        floc = sorted(p.name.split("contrast-")[1].split("_")[0] for p in
                      func.glob(f"sub-{s}_task-floc_hemi-L_space-fsaverage6_contrast-*_stat-t_desc-referenceOLS_statmap.func.gii"))
        if not floc:
            raise FileNotFoundError(f"{func}: no full-data fLoc t maps")
        vecs, keys = [], []
        for c in floc:
            vecs.append(np.concatenate([np.asarray(nib.load(str(
                func / f"sub-{s}_task-floc_hemi-{h}_space-fsaverage6_contrast-{c}_stat-t_desc-referenceOLS_statmap.func.gii"
            )).darrays[0].data, dtype=np.float64).ravel() for h in lc.HEMIS]))
            keys.append(("floc", c))
        meta = json.loads(next(func.glob(f"sub-{s}_task-motor_space-fsaverage6_desc-*Splithalf_statmap.json")).read_text())
        runs = [lc._sr(prefix) for prefix in meta["runs"]]
        for c in MOTOR_CONTRASTS:
            w = lc.DERIVED["motor"][c]
            vecs.append(np.mean([sum(wt * lc._run_effect(tree, s, "motor", ses, run, base, meta["desc"])
                                     for base, wt in w.items()) for ses, run in runs], axis=0))
            keys.append(("motor", c))
        root = prf_root / f"sub-{s}"
        angle, ecc, r2 = (lc._prf_map(root, f"sub-{s}", prm) for prm in ("angle", "eccentricity", "R2"))
        radius = float(json.loads((Path(derivatives) / "prf" / f"sub-{s}" /
                                   f"sub-{s}_task-prf_space-T1w_prf.json").read_text())["StimulusRadiusDeg"])
        px, py = sc.cartesian(angle, ecc)
        vecs += [px, py]
        keys += [("prf", "x"), ("prf", "y")]
        with np.errstate(invalid="ignore"):
            keep[s] = fs6_columns(((r2 > PRF_R2_FLOOR) & (ecc <= radius)).astype(np.float32), n) == 1
        rows_by_sub[s] = np.stack([fs6_columns(v, n) for v in vecs])
        keys_by_sub[s] = keys
    keys = keys_by_sub[subs[0]]
    if any(k != keys for k in keys_by_sub.values()):
        raise ValueError("subjects' localizer maps differ in contrasts")
    return rows_by_sub, keys, keep


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def face_name(face: str, names: list[str], space: str) -> str:
    """Model name for a combined-table face: combined, combined-minus-<b>, or the single route."""
    blocks = face.split("+")
    if len(blocks) == len(names):
        return "combined"
    if len(blocks) == 1:
        return f"stimulus-{space}" if blocks[0] == "stimulus" else blocks[0]
    return "combined-minus-" + "+".join(b for b in names if b not in blocks)


def face_models(job: dict, template_subs: list[str], n: int, n_jobs: int, log=print, space: str = SPACE,
                only: tuple[str, ...] | None = None, rename=None) -> tuple[dict, dict]:
    """Faces of the combined job's tuning table (combined-<space>), refitted from the Grams at its (w, λ).

    ``only`` keeps the named faces (model names as ``face_name`` gives them);
    ``rename(name)`` gives the name a face is scored under.
    """
    import combined as cb
    import stimulus_route as sr
    from joblib import Parallel, delayed

    target = job["target"]
    cdir = cb.out_dir(job["derivatives"], space, job["scenario"], job["pct"], job["draw"], target)
    side = json.loads((cdir / "combined.json").read_text())
    specs = cb.job_blocks(job["derivatives"], job["parts"], job["windows"], space, job["scenario"], job["pct"],
                          job["draw"], target)
    names = [nm for nm, _, _ in specs]
    if names != list(side["blocks"]):
        raise ValueError(f"combined job's blocks {list(side['blocks'])} differ from this job's {names}")
    raw = [cb.load_block(nm, d, prts, template_subs, target) for nm, d, prts in specs]
    common = cb.common_columns(raw)
    blocks = [cb.restrict(b, common) for b in raw]
    del raw
    coefs = [cb.energy_scale(b, template_subs) for b in blocks]
    for nm, c in zip(names, coefs):
        if not np.isclose(c, side["blocks"][nm]["scale_c"], rtol=1e-9):
            raise ValueError(f"{nm}: energy scale {c} differs from the combined job's {side['blocks'][nm]['scale_c']}")
    labs = sorted(common.template, key=lambda lab: -common.template[lab].size)
    tcols = {lab: common.template[lab][pos] for lab, pos in common.target.items()}
    subs = [*template_subs, target]
    models, info = {}, {}
    for face, spec in side["selection"].items():
        name = face_name(face, names, space)
        if only is not None and name not in only:
            continue
        w = [spec["weights"][nm] * c for nm, c in zip(names, coefs)]
        fitted = Parallel(n_jobs=n_jobs)(
            delayed(cb.fit_piece)([{pair: b.tpl[pair][lab] for pair in b.tpl} for b in blocks],
                                  [{pair: b.tgt[pair][lab] for pair in b.tgt} for b in blocks]
                                  if lab in common.target else [], w, template_subs, target,
                                  common.target.get(lab))
            for lab in labs)
        crosses = {s: sr.as_cross({lab: f[s] for lab, f in zip(labs, fitted) if s in f},
                                  tcols if s == target else common.template) for s in subs}
        name = rename(name) if rename else name
        models[name] = {s: pr.transform_from_cross(crosses[s], n, spec["lam"]) for s in subs}
        info[name] = {"source": "combined face", "face": face, "weights": spec["weights"], "lam": spec["lam"],
                      "objective": spec["objective"]}
        if space != SPACE:
            info[name]["combined_job"] = f"combined-{space}"
        if face == side["full_model"]:
            info[name]["reproduces_saved_crosses"] = _check_full(crosses, cdir, subs)
    if only is not None and set(only) - {face_name(f, names, space) for f in side["selection"]}:
        raise ValueError(f"combined-{space} has no face {sorted(set(only) - set(models))}")
    label = "combined" if space == SPACE else f"combined-{space}"
    log(f"{label} faces: {sorted(models)} ({len(common.template)} pieces)")
    return models, info


def _check_full(crosses: dict, cdir: Path, subs: list[str]) -> float:
    """Max relative difference between the refitted full model and the combined job's saved crosses."""
    import cha

    worst = 0.0
    for s in subs:
        saved = cha.load_cross(cdir / f"cross_sub-{s}.npz")
        mine = {str(lab): v for lab, v in crosses[s].items()}
        if saved.keys() != mine.keys():
            raise ValueError(f"sub-{s}: refitted full model covers other pieces than the saved crosses")
        for lab, (cols, m) in saved.items():
            c2, m2 = mine[lab]
            if not np.array_equal(cols, c2):
                raise ValueError(f"sub-{s} {lab}: refitted columns differ from the saved crosses")
            worst = max(worst, float(np.abs(m2 - m).max() / max(np.abs(m).max(), 1e-300)))
    if worst > FULL_REPRO_RTOL:
        raise ValueError(f"refitted full model differs from the saved crosses (max relative {worst:.2e})")
    return worst


def tuned_alone(job: dict, space: str, template_subs: list[str], y: dict, n: int, n_jobs: int) -> tuple[dict, dict]:
    """A stimulus route outside the combined model, tuned alone (λ only) by the §3.4 objective."""
    import cha
    import combined as cb
    import stimulus_route as sr
    from joblib import Parallel, delayed

    target = job["target"]
    d = sr.out_dir(job["derivatives"], space, job["scenario"], job["pct"], job["draw"], target)
    blk = cb.load_block("stimulus", d, sr.BLOCKS, template_subs, target)
    labs = list(blk.pcols.template)
    res = Parallel(n_jobs=n_jobs)(
        delayed(cb.tune_piece)([{pair: blk.tpl[pair][lab] for pair in blk.tpl}], [1.0], [(1.0,)], template_subs,
                               {s: y[s][:, blk.pcols.template[lab]] for s in template_subs})
        for lab in labs)
    results = {lab: r for lab, r in zip(labs, res) if r is not None}
    table = cb.tuning_table(results, [(1.0,)], ["stimulus"])
    best = table.loc[table["objective"].idxmax()]
    lam = float(best["lam"])
    subs = [*template_subs, target]
    tfs = {s: pr.transform_from_cross(cha.load_cross(d / f"cross_sub-{s}.npz"), n, lam) for s in subs}
    return tfs, {"source": "tuned alone on its own Grams", "lam": lam, "objective": float(best["objective"]),
                 "pieces_left_out": len(labs) - len(results)}


def srm_objective(w: dict[str, dict], template_subs: list[str], y: dict[str, np.ndarray]) -> tuple[float, int]:
    """§3.4 objective through the shared space: b carried into a's space by ``W_b W_a'``, per-column r, both ways."""
    a, b = template_subs
    num = 0.0
    cols_n = 0
    for lab, (cols, w_a) in w[a].items():
        cols_b, w_b = w[b][lab]
        if not np.array_equal(cols, cols_b):
            raise ValueError(f"{lab}: template subjects' SRM columns differ")
        ya, yb = y[a][:, cols], y[b][:, cols]
        if not (np.isfinite(ya).all() and np.isfinite(yb).all()):
            continue  # left out, never filled (as combined.tune_piece)
        r_ab = sc.column_r(yb @ w_b @ w_a.T, ya)
        r_ba = sc.column_r(ya @ w_a @ w_b.T, yb)
        ok = np.isfinite(r_ab) & np.isfinite(r_ba)
        num += r_ab[ok].sum() + r_ba[ok].sum()
        cols_n += int(ok.sum())
    return (num / (2 * cols_n) if cols_n else np.nan), cols_n


def srm_models(job: dict, template_subs: list[str], y: dict, n: int) -> tuple[dict, dict]:
    """SRM and its PCA control at the k the §3.4 objective picks (DECIDED #3)."""
    import cha
    import srm_route as srm

    target = job["target"]
    d = srm.out_dir(job["derivatives"], job["scenario"], job["pct"], job["draw"], target)
    side = json.loads((d / "srm.json").read_text())
    subs = [*template_subs, target]

    def load(kind, k):
        return {s: cha.load_cross(d / f"{kind}_sub-{s}_k-{k:03d}.npz") for s in subs}

    objective = {k: srm_objective(load("w", k), template_subs, y)[0] for k in side["k_grid"]}
    k = max(objective, key=lambda kk: (objective[kk], -kk))  # ties go to the smaller k
    models = {kind: {s: pr.PiecewiseTransform(n, dict(v)) for s, v in load(kind, k).items()} for kind in ("w", "pca")}
    info = {"source": "srm route, k tuned by the §3.4 objective", "k": int(k),
            "objective": {str(kk): float(v) for kk, v in objective.items()}}
    return {"srm": models["w"], "pca": models["pca"]}, {"srm": info, "pca": {**info, "source": "PCA control at SRM's k"}}


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def project_all(models: dict, data: dict[str, dict[str, np.ndarray]], template_subs: list[str], target: str
                ) -> dict[str, dict[str, np.ndarray]]:
    """``{model: {film: template mean in the target's space}}``; a ``None`` model is identity, and a callable
    model is identity followed by that function of the template mean (the smoothed-anatomical control)."""
    out = {}
    for name, tfs in models.items():
        out[name] = {}
        for k in data[target]:
            if tfs is None:
                out[name][k] = np.mean([data[s][k] for s in template_subs], axis=0)
            elif callable(tfs):
                out[name][k] = tfs(np.mean([data[s][k] for s in template_subs], axis=0))
            else:
                out[name][k] = sc.project({s: data[s][k] for s in template_subs}, tfs, target).astype(np.float32)
    return out


def shared_valid(target_films: dict[str, np.ndarray], proj: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    """Columns finite in the target's data and in every model's projection over every film (DECIDED #2)."""
    ok = np.logical_and.reduce([np.isfinite(x).all(axis=0) for x in target_films.values()])
    for per_film in proj.values():
        for x in per_film.values():
            ok &= np.isfinite(x).all(axis=0)
    return ok


def score_set(target_films: dict[str, np.ndarray], proj: dict[str, dict[str, np.ndarray]],
              networks: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """M2b (per segment) and M2a (per film) for every model on the shared column set."""
    valid = shared_valid(target_films, proj)
    nets = {net: cols[valid[cols]] for net, cols in sc.network_columns(networks).items()}
    empty = {net: 0 for net, c in nets.items() if not c.size}  # recorded in the column counts, not scored
    nets = {net: c for net, c in nets.items() if c.size}
    m2b, m2a = [], []
    for name, per_film in proj.items():
        m2b.append(sc.m2b(target_films, per_film, nets).assign(model=name))
        for k, x in target_films.items():
            m2a += [{"model": name, "network": net, "film": k, "r": r}
                    for net, r in sc.m2a(x, per_film[k], nets).items()]
    cols = {**{net: int(c.size) for net, c in nets.items()}, **empty}
    return pd.concat(m2b, ignore_index=True), pd.DataFrame(m2a), cols


def score_items(models: dict, items: dict, floor: dict, foils: np.ndarray, networks: np.ndarray,
                template_subs: list[str], target: str) -> tuple[pd.DataFrame, dict]:
    """M3 for every model, with both foil pools, on one shared column set (finite everywhere, above every floor)."""
    data = {s: {"items": x} for s, x in items.items()}
    proj = project_all(models, data, template_subs, target)
    valid = shared_valid({"items": items[target]}, proj) & np.logical_and.reduce(list(floor.values()))
    nets = {net: cols[valid[cols]] for net, cols in sc.network_columns(networks).items()}
    nets = {net: c for net, c in nets.items() if c.size}  # an ROI without TB columns (hippocampus) is not scored
    rows = [sc.m3(items[target], d["items"], nets, f).assign(model=name, foils=pool)
            for name, d in proj.items() for pool, f in (("all", None), ("no_triplet", foils))]
    return pd.concat(rows, ignore_index=True), {net: int(c.size) for net, c in nets.items()}


def score_maps(models: dict, maps: dict, keys: list[tuple], keep: dict, networks: np.ndarray, hemi: np.ndarray,
               template_subs: list[str], target: str, read_by_network: dict | None = None) -> pd.DataFrame:
    """M1 for every model: per contrast and the component median (fLoc, motor); pRF angle; ``read`` marks §9's cells.

    Per map row, a column is scored where the target's map and every model's
    projection are finite; pRF angle additionally needs the target's keep mask.
    """
    data = {s: {"maps": x} for s, x in maps.items()}
    proj = project_all(models, data, template_subs, target)
    tgt = maps[target]
    valid = np.isfinite(tgt)
    for d in proj.values():
        valid &= np.isfinite(d["maps"])
    nets = sc.network_columns(networks)
    ix = {k: i for i, k in enumerate(keys)}
    rows = []
    for name, d in proj.items():
        pm = d["maps"]
        for net, cols in nets.items():
            read = (M1_COMPONENTS if read_by_network is None else read_by_network).get(net, ())
            for comp in ("floc", "motor"):
                rs = []
                for key in (k for k in keys if k[0] == comp):
                    i = ix[key]
                    c = cols[valid[i, cols]]
                    r = sc.m1_map(tgt[i], pm[i], {net: c})[net]
                    rs.append(r)
                    rows.append({"model": name, "network": net, "component": comp, "contrast": key[1], "r": r,
                                 "n_columns": int(c.size), "read": comp in read})
                rows.append({"model": name, "network": net, "component": comp, "contrast": "median",
                             "r": float(np.nanmedian(rs)) if np.isfinite(rs).any() else np.nan,
                             "n_columns": None, "read": comp in read})
            ix_x, ix_y = ix[("prf", "x")], ix[("prf", "y")]
            ok = valid[ix_x] & valid[ix_y] & keep[target]
            c = cols[ok[cols]]
            r = sc.m1_angle((tgt[ix_x], tgt[ix_y]), (pm[ix_x], pm[ix_y]), hemi, ok, {net: c})[net]
            rows.append({"model": name, "network": net, "component": "prf", "contrast": "angle", "r": r,
                         "n_columns": int(c.size), "read": "prf" in read})
    return pd.DataFrame(rows)


def arm_name(name: str, space: str) -> str:
    """A combined-<space> face's name in the psytwill arm: combined faces carry their space."""
    return name.replace("combined", f"combined-{space}", 1) if name.startswith("combined") else name


def build_models(job: dict, template_subs: list[str], n: int, n_jobs: int, cleaned: Path, log=print,
                 arm: str = "ebind") -> tuple[dict, dict]:
    """Every model a partition job scores.

    ebind arm: anatomical, the combined faces, VGG19 tuned alone, SRM/PCA if defined.
    psytwill arm: anatomical, every combined-psytwill face (combined ones renamed
    combined-psytwill*), and combined-EBind's full model and stimulus face as
    ``combined-ebind`` / ``stimulus-ebind``, refitted exactly as the ebind arm does.
    """
    import combined as cb

    models: dict = {"anatomical": None}
    info: dict = {"anatomical": {"source": "fsaverage6 vertex identity"}}
    if arm == "psytwill":
        faces, face_info = face_models(job, template_subs, n, n_jobs, log, space="psytwill",
                                       rename=lambda nm: arm_name(nm, "psytwill"))
        models.update(faces)
        info.update(face_info)
        faces, face_info = face_models(job, template_subs, n, n_jobs, log, space="ebind",
                                       only=("combined", "stimulus-ebind"),
                                       rename=lambda nm: arm_name(nm, "ebind"))
        models.update(faces)
        info.update(face_info)
        return models, info
    faces, face_info = face_models(job, template_subs, n, n_jobs, log)
    models.update(faces)
    info.update(face_info)
    _, y = cb.tuning_rows(job["parts"], job["windows"], job["scenario"], job["pct"], job["draw"], job["target"],
                          template_subs, cleaned)
    models["stimulus-vgg19"], info["stimulus-vgg19"] = tuned_alone(job, "vgg19", template_subs, y, n, n_jobs)
    if "response" in models:  # the target shares films: SRM is defined
        m, i = srm_models(job, template_subs, y, n)
        models.update(m)
        info.update(i)
    del y
    return models, info


def run_score(args: argparse.Namespace, log=print) -> Path:
    import cha
    import combined as cb
    import encoding as enc
    import films as fm
    import grayordinates as go
    import partitions as pt

    t0 = time.time()
    paths = enc.Paths()
    cleaned = cha.Paths().cleaned
    parts = pt.load_partitions(paths.derivatives)
    windows = fm.load_windows(paths.derivatives)
    subs = sorted(parts.loc[(parts["scenario"] == args.scenario) & (parts["pct"] == args.pct)
                            & (parts["draw"] == args.draw) & (parts["target"] == args.target), "subject"].unique())
    if args.target not in subs or len(subs) != 3:
        raise ValueError(f"job resolves to subjects {subs}; expected the target and two template subjects")
    template_subs = [s for s in subs if s != args.target]
    job = {"derivatives": paths.derivatives, "parts": parts, "windows": windows, "scenario": args.scenario,
           "pct": args.pct, "draw": args.draw, "target": args.target}
    table = pd.read_csv(go.grayordinates_path(cleaned), sep="\t")
    n = len(table)
    networks = network_labels(table, cha.Paths().atlases)
    seconds = {}

    t1 = time.time()
    models, info = build_models(job, template_subs, n, args.n_jobs, cleaned, log, arm=args.arm)
    seconds["models"] = round(time.time() - t1, 1)
    log(f"models: {list(models)} ({seconds['models']:.0f} s)")

    frames_b, frames_a, n_cols = [], [], {}
    for role, fset in FILM_SETS.items():
        t1 = time.time()
        data = load_films(film_rows(windows, subs, role), subs, args.target,
                          lambda row, cache: fm.film_series(row, cleaned, cache))
        proj = project_all(models, data, template_subs, args.target)
        b, a, cols = score_set(data[args.target], proj, networks)
        frames_b.append(b.assign(set=fset))
        frames_a.append(a.assign(set=fset))
        n_cols[fset] = cols
        del data, proj
        seconds[f"score_{fset}"] = round(time.time() - t1, 1)
        log(f"{fset}: {b['film'].nunique()} films, columns {cols} ({seconds[f'score_{fset}']:.0f} s)")

    t1 = time.time()
    items, floor, item_ids, foils = load_items(paths.derivatives, subs, n, networks)
    m3, m3_cols = score_items(models, items, floor, foils, networks, template_subs, args.target)
    del items
    seconds["score_m3"] = round(time.time() - t1, 1)
    log(f"M3: {len(item_ids)} items ({int(foils.sum())} triplet-free foils), columns {m3_cols} "
        f"({seconds['score_m3']:.0f} s)")

    t1 = time.time()
    maps, map_keys, keep = load_maps(paths.derivatives, subs, n)
    m1 = score_maps(models, maps, map_keys, keep, networks, table["hemi"].to_numpy(), template_subs, args.target)
    del maps
    seconds["score_m1"] = round(time.time() - t1, 1)
    log(f"M1: {len(map_keys)} maps ({seconds['score_m1']:.0f} s)")

    dest = out_dir(paths.derivatives, args.scenario, args.pct, args.draw, args.target, args.arm)
    dest.mkdir(parents=True, exist_ok=True)
    pd.concat(frames_b, ignore_index=True).to_parquet(dest / "m2b.parquet", index=False)
    pd.concat(frames_a, ignore_index=True).to_csv(dest / "m2a.tsv", sep="\t", index=False, float_format="%.6g")
    m3.to_csv(dest / "m3.tsv", sep="\t", index=False, float_format="%.6g")
    m1.to_csv(dest / "m1.tsv", sep="\t", index=False, float_format="%.6g")
    seconds["total"] = round(time.time() - t0, 1)
    side = {
        "description": "functional-space scores for one partition job, per model and family-A network: M2b per "
                       "segment (m2b.parquet) and M2a per film (m2a.tsv) per film set, M3 per foil pool (m3.tsv), "
                       "M1 per map and component median (m1.tsv); template side = mean of the template subjects "
                       "carried into the target's space",
        "job": {"scenario": args.scenario, "pct": args.pct, "draw": args.draw, "target": args.target},
        **({} if args.arm == "ebind" else {"arm": args.arm}),
        "template_subjects": template_subs, "models": info, "n_columns": n_cols,
        "m3": {"items": len(item_ids), "triplet_free_foils": int(foils.sum()), "n_columns": m3_cols,
               "floor": {"meanvol_frac": MEANVOL_FRAC, "beta_cap": BETA_CAP}},
        "m1": {"maps": [list(k) for k in map_keys], "read": {k: list(v) for k, v in M1_COMPONENTS.items()},
               "prf_r2_floor": PRF_R2_FLOOR},
        "film_sets": {fset: role for role, fset in FILM_SETS.items()}, "reference_grid": "target",
        "code_version": sc._code_version(), "seconds": seconds,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (dest / "score.json").write_text(json.dumps(side, indent=2) + "\n")
    log(f"wrote {dest} in {seconds['total']:.0f} s")
    return dest


#: Routes whose smoothness the smoothed-anatomical control matches (§6; Bazeille 2021), where the job has them.
SMOOTH_MATCH = ("combined", "cha", "srm")


def families_dir(derivatives: Path, scenario: str, pct: int, draw: int, target: str, arm: str = "ebind") -> Path:
    return out_dir(derivatives, scenario, pct, draw, target, arm) / "families"


def run_families(args: argparse.Namespace, log=print) -> Path:
    """§10 secondary families and the §6 smoothed-anatomical control for one partition job.

    The same models as ``score``, plus one smoothed-anatomical model per route
    in SMOOTH_MATCH, scored per family: M2b and M2a on both film sets, M3 on
    the three-exposure and the single-exposure items, M1 on cortical families
    (read only where §9 reads it, i.e. family A).
    """
    import cha
    import encoding as enc
    import families as fam
    import films as fm
    import grayordinates as go
    import partitions as pt

    t0 = time.time()
    paths = enc.Paths()
    cpaths = cha.Paths()
    cleaned = cpaths.cleaned
    parts = pt.load_partitions(paths.derivatives)
    windows = fm.load_windows(paths.derivatives)
    subs = sorted(parts.loc[(parts["scenario"] == args.scenario) & (parts["pct"] == args.pct)
                            & (parts["draw"] == args.draw) & (parts["target"] == args.target), "subject"].unique())
    if args.target not in subs or len(subs) != 3:
        raise ValueError(f"job resolves to subjects {subs}; expected the target and two template subjects")
    template_subs = [s for s in subs if s != args.target]
    job = {"derivatives": paths.derivatives, "parts": parts, "windows": windows, "scenario": args.scenario,
           "pct": args.pct, "draw": args.draw, "target": args.target}
    table = pd.read_csv(go.grayordinates_path(cleaned), sep="\t")
    n = len(table)
    labels = fam.family_labels(table, cpaths.atlases)
    if not np.array_equal(labels["schaefer7n"], network_labels(table, cpaths.atlases)):
        raise ValueError("family A labels differ from the primary scoring's network labels")
    edges = fam.mesh_edges(cpaths.freesurfer)
    smoother = fam.Smoother(edges, n)
    seconds = {}

    t1 = time.time()
    models, info = build_models(job, template_subs, n, args.n_jobs, cleaned, log, arm=args.arm)
    seconds["models"] = round(time.time() - t1, 1)

    frames_b, frames_a, smooth_rows, n_cols, matched = [], [], [], {}, {}
    for role, fset in FILM_SETS.items():
        t1 = time.time()
        data = load_films(film_rows(windows, subs, role), subs, args.target,
                          lambda row, cache: fm.film_series(row, cleaned, cache))
        proj = project_all(models, data, template_subs, args.target)
        if fset == PRIMARY_SET:  # match smoothness on the primary films, then use the same steps for every set
            route_r = {m: fam.neighbour_r(proj[m], edges) for m in SMOOTH_MATCH if m in models}
            grid_r = fam.smoothness_grid(smoother, proj["anatomical"], edges, max(route_r.values()))
            smooth_rows += [{"model": "anatomical", "steps": k, "neighbour_r": r} for k, r in grid_r.items()]
            smooth_rows.append({"model": "target", "steps": 0, "neighbour_r": fam.neighbour_r(data[args.target], edges)})
            for route, r in route_r.items():
                matched[route] = fam.match_steps(r, grid_r)
                smooth_rows.append({"model": route, "steps": matched[route], "neighbour_r": r,
                                    "matched_anatomical_r": grid_r[matched[route]]})
            log(f"smoothness: routes {route_r}, anatomical grid {grid_r}, matched steps {matched}")
        smoothed = {f"anatomical-smooth-{r}": (lambda x, k=k: smoother.run(x, [k])[k]) for r, k in matched.items()}
        proj.update(project_all(smoothed, data, template_subs, args.target))
        for family, lab in labels.items():
            b, a, cols = score_set(data[args.target], proj, lab)
            frames_b.append(b.assign(set=fset, family=family))
            frames_a.append(a.assign(set=fset, family=family))
            n_cols[f"{fset}/{family}"] = cols
        del data, proj
        seconds[f"score_{fset}"] = round(time.time() - t1, 1)
        log(f"{fset}: families scored ({seconds[f'score_{fset}']:.0f} s)")
    all_models = {**models, **{f"anatomical-smooth-{r}": (lambda x, k=k: smoother.run(x, [k])[k])
                               for r, k in matched.items()}}

    t1 = time.time()
    m3_frames, m3_meta = [], {}
    for exposures, item_set in ((3, "three"), (1, "single")):
        pats, base, meanvol, item_ids, foils = item_inputs(paths.derivatives, subs, n, exposures, subcortex=table)
        for family, lab in labels.items():
            floor = {s: item_floor(base[s], meanvol[s], lab) for s in subs}
            m3, cols = score_items(all_models, pats, floor, foils, lab, template_subs, args.target)
            if exposures == 1:
                m3 = m3[m3["foils"] == "all"]  # no single-exposure item is a triplet: one pool
            m3_frames.append(m3.assign(family=family, items=item_set))
            m3_meta[f"{item_set}/{family}"] = cols
        m3_meta[item_set] = {"items": len(item_ids), "triplet_free_foils": int(foils.sum())}
        del pats
    seconds["score_m3"] = round(time.time() - t1, 1)
    log(f"M3: {m3_meta['three']} three-exposure, {m3_meta['single']} single ({seconds['score_m3']:.0f} s)")

    t1 = time.time()
    maps, map_keys, keep = load_maps(paths.derivatives, subs, n)
    m1_frames = [score_maps(all_models, maps, map_keys, keep, labels[f], table["hemi"].to_numpy(), template_subs,
                            args.target, None if f == FAMILY_A else {}).assign(family=f)
                 for f in ("schaefer7n", "familyB", "familyD")]
    del maps
    seconds["score_m1"] = round(time.time() - t1, 1)

    dest = families_dir(paths.derivatives, args.scenario, args.pct, args.draw, args.target, args.arm)
    dest.mkdir(parents=True, exist_ok=True)
    pd.concat(frames_b, ignore_index=True).to_parquet(dest / "m2b.parquet", index=False)
    pd.concat(frames_a, ignore_index=True).to_csv(dest / "m2a.tsv", sep="\t", index=False, float_format="%.6g")
    pd.concat(m3_frames, ignore_index=True).to_csv(dest / "m3.tsv", sep="\t", index=False, float_format="%.6g")
    pd.concat(m1_frames, ignore_index=True).to_csv(dest / "m1.tsv", sep="\t", index=False, float_format="%.6g")
    pd.DataFrame(smooth_rows).to_csv(dest / "smoothing.tsv", sep="\t", index=False, float_format="%.6g")
    seconds["total"] = round(time.time() - t0, 1)
    side = {
        "description": "functional-space secondary scores for one partition job: every model (as `score`) plus a "
                       "smoothed-anatomical control per SMOOTH_MATCH route, per §10 family. M2b/M2a per film set, "
                       "M3 per item set (three-exposure, single-exposure) and foil pool, M1 on cortical families "
                       "(read only in family A); smoothing.tsv = mesh-neighbour temporal r of each projection",
        "job": {"scenario": args.scenario, "pct": args.pct, "draw": args.draw, "target": args.target},
        **({} if args.arm == "ebind" else {"arm": args.arm}),
        "template_subjects": template_subs, "families": list(labels), "models": info,
        "smoothing": {"grid": list(fam.SMOOTH_GRID), "matched_steps": matched,
                      "rule": "fewest steps whose anatomical neighbour r is closest to the route's, held-out films"},
        "n_columns": n_cols, "m3": m3_meta, "code_version": sc._code_version(), "seconds": seconds,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (dest / "families.json").write_text(json.dumps(side, indent=2) + "\n")
    log(f"wrote {dest} in {seconds['total']:.0f} s")
    return dest


M4_LEVEL = 0  # DECIDED 2026-09-30 #4: the stimulus routes at 0% only (primary and secondary)


def m4_residuals(job: dict, subs: list[str], space: str, backend: str, log=print
                 ) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Each subject's held-out residual per film, on its own grid: z-scored window minus its own encoder's prediction.

    The encoder is refitted exactly as the stimulus route fitted it (the job's
    alignment films, per-window z-scored); predictions are centred per window,
    as the route centred them.
    """
    import cha
    import encoding as enc
    import stimulus_route as sr

    cleaned = cha.Paths().cleaned
    cache = enc.Paths().cache
    align = sr.job_films(job["parts"], job["windows"], job["scenario"], job["pct"], job["draw"], job["target"])
    w = job["windows"]
    out, diag = {}, {}
    for s in subs:
        x, y, groups, bands = enc.film_design(align[s], cache, space, cleaned)
        model = enc.Encoder(bands, backend=backend).fit(x, enc._zscore_film_windows(y, groups), groups)
        held = w[(w["sub"] == s) & (w["role"] == "heldout")]
        xh, yh, gh, _ = enc.film_design(held, cache, space, cleaned)
        res = enc._zscore_film_windows(yh, gh) - sr.center_blocks(model.predict(xh), gh)
        out[s] = {sid: res[gh == sid] for sid in held["stimulus_id"]}
        diag[f"sub-{s}"] = model.diagnostics_
        log(f"{space} sub-{s}: encoder {model.diagnostics_['cv_1_plus_score_q50_q95_q99']}")
    return out, diag


def m4_route(job: dict, space: str, subs: list[str], n: int) -> tuple[dict, float]:
    """The stimulus route at its tuned λ: EBind's and psytwill's from their combined table's one-block face,
    VGG19's from its scoring."""
    import cha
    import combined as cb
    import stimulus_route as sr

    target = job["target"]
    if space in (SPACE, "psytwill"):
        side = json.loads((cb.out_dir(job["derivatives"], space, job["scenario"], job["pct"], job["draw"], target)
                           / "combined.json").read_text())
        lam = float(side["selection"]["stimulus"]["lam"])
    else:
        side = json.loads((out_dir(job["derivatives"], job["scenario"], job["pct"], job["draw"], target)
                           / "score.json").read_text())
        lam = float(side["models"][f"stimulus-{space}"]["lam"])
    d = sr.out_dir(job["derivatives"], space, job["scenario"], job["pct"], job["draw"], target)
    return {s: pr.transform_from_cross(cha.load_cross(d / f"cross_sub-{s}.npz"), n, lam) for s in subs}, lam


def run_m4(args: argparse.Namespace, log=print) -> Path:
    import cha
    import encoding as enc
    import films as fm
    import grayordinates as go
    import partitions as pt

    if args.pct != M4_LEVEL:
        raise ValueError(f"M4 is scored at {M4_LEVEL}% only (DECIDED 2026-09-30 #4)")
    t0 = time.time()
    paths = enc.Paths()
    cleaned = cha.Paths().cleaned
    parts = pt.load_partitions(paths.derivatives)
    windows = fm.load_windows(paths.derivatives)
    subs = sorted(parts.loc[(parts["scenario"] == args.scenario) & (parts["pct"] == args.pct)
                            & (parts["draw"] == args.draw) & (parts["target"] == args.target), "subject"].unique())
    if args.target not in subs or len(subs) != 3:
        raise ValueError(f"job resolves to subjects {subs}; expected the target and two template subjects")
    template_subs = [s for s in subs if s != args.target]
    job = {"derivatives": paths.derivatives, "parts": parts, "windows": windows, "scenario": args.scenario,
           "pct": args.pct, "draw": args.draw, "target": args.target}
    table = pd.read_csv(go.grayordinates_path(cleaned), sep="\t")
    n = len(table)
    networks = network_labels(table, cha.Paths().atlases)
    rows = film_rows(windows, subs, "heldout")
    keys = sorted(rows[args.target].index)
    slices = {k: fm.paired_slices({s: rows[s].loc[k] for s in subs}, args.target) for k in keys}
    raw = load_films(rows, subs, args.target, lambda row, cache: fm.film_series(row, cleaned, cache))

    frames, info = [], {}
    for space in M4_SPACES_BY_ARM[args.arm]:
        res, diag = m4_residuals(job, subs, space, args.backend, log)
        resid = {s: {k: sc.zscore_columns(res[s][k][slices[k][s]]).astype(np.float32) for k in keys} for s in subs}
        tfs, lam = m4_route(job, space, subs, n)
        models = {"anatomical": None, f"stimulus-{space}": tfs}
        for kind, data in (("raw", raw), ("residual", resid)):
            proj = project_all(models, data, template_subs, args.target)
            _, a, cols = score_set(data[args.target], proj, networks)
            frames.append(a.assign(space=space, kind=kind))
        info[space] = {"lam": lam, "encoders": diag, "n_columns": cols}
        del res, resid

    dest = out_dir(paths.derivatives, args.scenario, args.pct, args.draw, args.target, args.arm)
    dest.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(dest / "m4.tsv", sep="\t", index=False, float_format="%.6g")
    side = {"description": "M4 raw-residual check (§9; scope DECIDED 2026-09-30 #4): M2a per held-out film for "
                           "anatomical and each stimulus route, on the raw series and on the residual after each "
                           "subject's own refitted encoder prediction is removed from its own series; one shared "
                           "column set per (space, kind)",
            "job": {"scenario": args.scenario, "pct": args.pct, "draw": args.draw, "target": args.target},
            **({} if args.arm == "ebind" else {"arm": args.arm}),
            "template_subjects": template_subs, "spaces": info, "backend": args.backend,
            "code_version": sc._code_version(), "seconds": round(time.time() - t0, 1),
            "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    (dest / "m4.json").write_text(json.dumps(side, indent=2) + "\n")
    log(f"wrote {dest / 'm4.tsv'} in {side['seconds']:.0f} s")
    return dest


def run_mni(args: argparse.Namespace, log=print) -> Path:
    """The MNI voxel-identity baseline for one target: job-independent (no alignment)."""
    import cha
    import encoding as enc
    import films as fm

    t0 = time.time()
    paths = enc.Paths()
    cleaned = cha.Paths().cleaned
    windows = fm.load_windows(paths.derivatives)
    voxels = pd.read_csv(Path(cleaned) / "mni_voxels.tsv", sep="\t")
    networks = mni_network_labels(voxels)
    subs = sorted(windows["sub"].unique())
    if args.target not in subs or len(subs) != 3:
        raise ValueError(f"film windows cover subjects {subs}; expected three including the target")
    template_subs = [s for s in subs if s != args.target]
    frames_b, frames_a, n_cols = [], [], {}
    for role, fset in FILM_SETS.items():
        data = load_films(film_rows(windows, subs, role), subs, args.target, mni_reader(cleaned, len(voxels)))
        proj = project_all({"anatomical-mni": None}, data, template_subs, args.target)
        b, a, cols = score_set(data[args.target], proj, networks)
        frames_b.append(b.assign(set=fset))
        frames_a.append(a.assign(set=fset))
        n_cols[fset] = cols
        log(f"{fset}: {b['film'].nunique()} films, columns {cols}")
    dest = mni_dir(paths.derivatives, args.target)
    dest.mkdir(parents=True, exist_ok=True)
    pd.concat(frames_b, ignore_index=True).to_parquet(dest / "m2b.parquet", index=False)
    pd.concat(frames_a, ignore_index=True).to_csv(dest / "m2a.tsv", sep="\t", index=False, float_format="%.6g")
    side = {"description": "MNI152NLin2009cAsym res-2 voxel-identity baseline (§6): M2b/M2a on the held-out films, "
                           "family-A networks from the MNI Schaefer 7-network atlas; no alignment, so one per target",
            "target": args.target, "template_subjects": template_subs, "n_columns": n_cols,
            "code_version": sc._code_version(), "seconds": round(time.time() - t0, 1),
            "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    (dest / "score.json").write_text(json.dumps(side, indent=2) + "\n")
    log(f"wrote {dest}")
    return dest


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------

def load_level(derivatives: Path, parts: pd.DataFrame, scenario: str, pct: int, fset: str = PRIMARY_SET,
               arm: str = "ebind") -> pd.DataFrame:
    """Per-film M2b (segment means) of every job at one level: target, draw, model, network, film, rank_acc.

    Every job the partition table lists must have been scored; a missing one is an error.
    """
    import partitions as pt

    jobs = pt.job_list(parts)
    jobs = jobs[(jobs["scenario"] == scenario) & (jobs["pct"] == pct)]
    frames, missing = [], []
    for j in jobs.itertuples(index=False):
        path = out_dir(derivatives, scenario, pct, int(j.draw), j.target, arm) / "m2b.parquet"
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_parquet(path)
        df = df[df["set"] == fset]
        frames.append(df.groupby(["model", "network", "film"], as_index=False)["rank_acc"].mean()
                      .assign(target=j.target, draw=int(j.draw)))
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(jobs)} jobs unscored, e.g. {missing[:3]}")
    return pd.concat(frames, ignore_index=True)


def draw_average(level: pd.DataFrame, model: str) -> pd.DataFrame:
    df = level[level["model"] == model]
    if df.empty:
        raise ValueError(f"no scores for model {model!r}")
    return df.groupby(["target", "network", "film"], as_index=False)["rank_acc"].mean()


def mc_error(level: pd.DataFrame, model: str, reference: pd.DataFrame, fs6_model: str = "anatomical") -> dict:
    """Monte Carlo error of each film's draw-averaged gain (SD over draws / sqrt(draws)), summarised (§4)."""
    key = ["target", "network", "film"]
    route = level[level["model"] == model]
    rows = []
    for (t, net), ref in reference.groupby(["target", "network"]):
        base = ref["baseline"].iat[0]
        r = route[(route["target"] == t) & (route["network"] == net)]
        if base == "fsaverage6":
            b = level[(level["model"] == fs6_model) & (level["target"] == t) & (level["network"] == net)]
            g = r.merge(b, on=[*key, "draw"], suffixes=("", "_ref"))
            g = g.assign(gain=g["rank_acc"] - g["rank_acc_ref"])
        else:
            g = r.merge(ref[[*key, "rank_acc"]], on=key, suffixes=("", "_ref"))
            g = g.assign(gain=g["rank_acc"] - g["rank_acc_ref"])
        rows.append(g.groupby("film")["gain"].agg(lambda v: v.std(ddof=1) / np.sqrt(len(v))))
    se = pd.concat(rows)
    return {"median": float(se.median()), "max": float(se.max()), "n_cells": int(se.size)}


def load_level_table(derivatives: Path, parts: pd.DataFrame, scenario: str, pct: int, name: str,
                     arm: str = "ebind") -> pd.DataFrame:
    """One per-job TSV (``m3.tsv``, ``m1.tsv``) over every job at a level, with target and draw; none may be missing."""
    import partitions as pt

    jobs = pt.job_list(parts)
    jobs = jobs[(jobs["scenario"] == scenario) & (jobs["pct"] == pct)]
    frames = []
    for j in jobs.itertuples(index=False):
        path = out_dir(derivatives, scenario, pct, int(j.draw), j.target, arm) / name
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing")
        frames.append(pd.read_csv(path, sep="\t").assign(target=j.target, draw=int(j.draw)))
    return pd.concat(frames, ignore_index=True)


def robustness(result: dict, m3: pd.DataFrame, m1: pd.DataFrame, route: str, reference: str = "anatomical") -> dict:
    """§9: M1 and M3 must agree in sign with M2b. Per rejected (target, network) cell, the sign of the draw-averaged
    M3 gain (all-item foils) and of every read M1 component's gain, route minus fsaverage6 anatomical (the maps and
    TB betas are surface data, so the MNI baseline has none). Read per network, as the go criterion is (DECIDED
    2026-10-04): a network is robust if its rejected-and-agreeing cells alone still reach ``min_targets``;
    ``robust`` = at least one such network."""
    def gain(df, keys):
        piv = df.pivot_table(index=[*keys, "draw"], columns="model", values=df.attrs["value"])
        return (piv[route] - piv[reference]).groupby(level=list(range(len(keys)))).mean()

    m3 = m3[m3["foils"] == "all"]
    m3.attrs["value"] = "rank_acc"
    m1 = m1[m1["read"] & m1["contrast"].isin(["median", "angle"])]
    m1.attrs["value"] = "r"
    g3 = gain(m3, ["target", "network"])
    g1 = gain(m1, ["target", "network", "component"])
    cells, agreeing = [], {}
    for t, per in result["reject"].items():
        for net, rej in per.items():
            if not rej:
                continue
            comps = {c: float(v) for (tt, nn, c), v in g1.items() if tt == t and nn == net}
            m3g = float(g3.get((t, net), np.nan))
            ok = m3g > 0 and all(v > 0 for v in comps.values())
            agreeing[net] = agreeing.get(net, 0) + ok
            cells.append({"target": t, "network": net, "m3_gain": m3g, "m1_gain": comps, "agrees": bool(ok)})
    min_targets = result.get("min_targets", 2)
    robust_networks = sorted(n for n, k in agreeing.items() if k >= min_targets)
    return {"cells": cells, "robust_networks": robust_networks, "robust": bool(robust_networks),
            "all_cells_agree": bool(cells) and all(c["agrees"] for c in cells),
            "rule": f"per network: rejected cells whose M3 (all-item foils) and every read M1 component gain > 0 "
                    f"number >= {min_targets} targets; gains are route minus fsaverage6 anatomical, averaged over draws"}


def run_decide(args: argparse.Namespace, log=print) -> Path:
    import encoding as enc
    import partitions as pt

    paths = enc.Paths()
    parts = pt.load_partitions(paths.derivatives)
    scenario, pct = H1_JOBS
    route = "combined" if args.arm == "ebind" else f"combined-{args.arm}"
    level = load_level(paths.derivatives, parts, scenario, pct, arm=args.arm)
    targets = sorted(level["target"].unique())
    mni = []
    for t in targets:
        df = pd.read_parquet(mni_dir(paths.derivatives, t) / "m2b.parquet")
        df = df[df["set"] == PRIMARY_SET]
        mni.append(df.groupby(["network", "film"], as_index=False)["rank_acc"].mean().assign(target=t))
    baselines = {"fsaverage6": draw_average(level, "anatomical"), "mni": pd.concat(mni, ignore_index=True)}
    reference = sc.reference_scores(baselines)
    combined = draw_average(level, route)
    h1 = sc.decide(sc.film_gains(combined, reference))
    h1["reference_choice"] = (reference.groupby(["target", "network"])["baseline"].first()
                              .unstack().to_dict(orient="index"))
    h1["mc_error"] = mc_error(level, route, reference)
    m3 = load_level_table(paths.derivatives, parts, scenario, pct, "m3.tsv", args.arm)
    m1 = load_level_table(paths.derivatives, parts, scenario, pct, "m1.tsv", args.arm)
    h1["robustness"] = robustness(h1, m3, m1, route)
    out = {"description": "Functional-space H1 (combined vs the stronger anatomical baseline) and, if H1 is a go, "
                          "H2 (combined vs CHA): M2b rank accuracy, film means averaged over draws, exact one-sided "
                          "sign-flip per target x network, Holm across networks within target, go if a network "
                          "survives in >= 2 targets (pre-registration §9)",
           **({} if args.arm == "ebind" else {"arm": args.arm, "route": route}),
           "scenario": scenario, "pct": pct, "film_set": PRIMARY_SET,
           "n_draws": level.groupby("target")["draw"].nunique().to_dict(), "H1": h1}
    cha = draw_average(level, "cha")
    if h1["go"]:
        out["H2"] = sc.decide(sc.film_gains(combined, cha))
        out["H2"]["robustness"] = robustness(out["H2"], m3, m1, route, reference="cha")
    else:
        out["H2"] = "not tested: H1 is a no-go (fixed sequence, §1)"
        out["exploratory_cha_vs_anatomical"] = sc.decide(sc.film_gains(cha, reference))
    out["code_version"] = sc._code_version()
    out["created"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    dest = scores_root(paths.derivatives, args.arm) / "decision.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    log(f"H1 {'GO' if h1['go'] else 'NO-GO'}: networks counted {h1['networks_counted']}; wrote {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("score")
    s.add_argument("--scenario", default="primary", choices=("primary", "secondary"))
    s.add_argument("--pct", type=int, required=True)
    s.add_argument("--draw", type=int, required=True)
    s.add_argument("--target", required=True)
    s.add_argument("--n-jobs", type=int, default=1)
    s.add_argument("--arm", default="ebind", choices=ARMS)
    g = sub.add_parser("m4")
    g.add_argument("--scenario", default="primary", choices=("primary", "secondary"))
    g.add_argument("--pct", type=int, required=True)
    g.add_argument("--draw", type=int, required=True)
    g.add_argument("--target", required=True)
    g.add_argument("--backend", default="torch_cuda")
    g.add_argument("--arm", default="ebind", choices=ARMS)
    m = sub.add_parser("mni")
    m.add_argument("--target", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--arm", default="ebind", choices=ARMS)
    f = sub.add_parser("families")
    f.add_argument("--scenario", default="primary", choices=("primary", "secondary"))
    f.add_argument("--pct", type=int, required=True)
    f.add_argument("--draw", type=int, required=True)
    f.add_argument("--target", required=True)
    f.add_argument("--n-jobs", type=int, default=1)
    f.add_argument("--arm", default="ebind", choices=ARMS)
    args = ap.parse_args()
    {"score": run_score, "m4": run_m4, "mni": run_mni, "decide": run_decide, "families": run_families}[args.verb](args)


if __name__ == "__main__":
    main()
