#!/usr/bin/env python3
"""Fit and score the five map classes for one cell of the neural-rotation
pilot: subject x rung x ROI x phase pair x beta type.

This is the cluster entry point (one SLURM array task per cell) and the
synthetic end-to-end entry point (fake_caches.py writes caches in the same
format). It carries no subject list and no ROI list: the ladder table and
the cache file say what exists.

Per cell, over the leave-one-retrieval-session-out folds of the design
table (design.py):

  1. voxels cleaned by the settled rule (basis.clean_voxels), then, inside
     each training fold, optionally preselected by encoding split-half
     reliability (top --preselect fraction per block; the default 1.0 keeps
     every voxel -- 0.5 dropped the word-cued signal, log 2026-09-23) and
     block-normalised; every identification score is also reported per
     reCon stratum (reCon1 = retrieved in the encoding session, reCon2 =
     later; targets restricted, candidates shared: *_reCon1 / *_reCon2)
     and against the non-triplet foil pool (*_ntf: foils with enCon != 3,
     for every target), and their crossing (*_ntf_reCon1 / *_ntf_reCon2);
  2. a shared PCA basis of the stacked training patterns; every rank of the
     grid is scored on the held-out fold (the rank sweep), and the headline
     rank per class is picked by nested CV inside training (basis fixed);
  3. each class scored forward (E -> R) and reverse (R -> E) by run-matched
     held-out identification, beside the encoding ceiling (exposure 1 vs
     mean of exposures 2+3, identity map) and the run-matched item-identity
     permutation null at the chosen rank;
  4. the rotation metric of the orthogonal map at its chosen rank: plane
     decomposition, variance-weighted mean plane angle, geodesic distance,
     principal angles, translation norm, each beside the encoding
     split-half floor and the item-permuted null; block energy at the
     union rungs; RDM correlation (geometry);
  5. a pooled fit on every non-anchor item for the per-item plane angles,
     the plane spectrum and the age slopes (the planes of a pooled map are
     one basis for all items; held-out fold maps each have their own), the
     content-vs-position split, and the anchor-drift table.

Pair ``enc:ret-word:ret-image`` runs the composition test instead: three
orthogonal maps in one basis, composed E -> R_word -> R_image against the
direct E -> R_image on held-out items.

Outputs, long format, under ``<out_root>/fits/<sub>/`` with the stem
``<sub>_rung-<rung>_roi-<ROI>_pair-<pair>_desc-type<t>``:
  _transformation_class.tsv  _rotation_metrics.tsv  _geometry.tsv
  _item_residuals.tsv  _delay_slopes.tsv  _plane_spectrum.tsv
  _anchor_drift.tsv  _block_energy.tsv  _composition.tsv  _fit.json

Usage:
    python fit_pair.py --subject sub-## --rung i --roi IntracalcarineCortex \\
        --pair enc:ret-word [--beta-type D] [--preselect 1.0] [--n-perm 200]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
SCRIPTS = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import basis  # noqa: E402
import blocks as blocks_mod  # noqa: E402
import maps  # noqa: E402
import rotation  # noqa: E402
import score  # noqa: E402


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tb = _load_module("glmsingle_tb", SCRIPTS / "glmsingle_tb.py")

DESIGN_TREE = "neural_rotation"
CACHE_TREE = "pattern_similarity"
ROI_SET = "ladder"
PHASE_ARMS = {"enc": "enc", "ret-word": "ret-word-tbonly", "ret-image": "ret-image-tbonly"}
RANK_GRID = basis.RANK_GRID
PRINCIPAL_M = (2, 3, 5, 10, 20, 50)
HIGH_RANK = 200          # above this the null draws cost k^3 each: fewer draws (see --n-perm-high)
CHANCE = 0.5
MIN_TEST_ITEMS = 6


# ── loading ──────────────────────────────────────────────────────────────────

def cache_path(cache_root: Path, subject: str, arm: str, beta_type: str) -> Path:
    return (cache_root / "cache" / "glmsingle_tb" / subject / arm
            / f"{subject}_arm-{arm}_set-{ROI_SET}_desc-type{beta_type.lower()}_roipatterns.npz")


def load_design(design_root: Path, subject: str) -> pd.DataFrame:
    p = design_root / subject / f"{subject}_desc-trials.tsv"
    if not p.exists():
        sys.exit(f"ERROR: design table missing: {p} (run design.py --subject {subject})")
    df = pd.read_csv(p, sep="\t", na_values=["n/a"])
    df["key"] = list(zip(df["phase"], df["session"], df["run"].astype(int), df["onset"].round(3)))
    return df


def load_arm(cache_root: Path, subject: str, phase: str, beta_type: str, roi: str,
             design: pd.DataFrame, arm_map: dict) -> dict:
    """Patterns (N_trials, V) of one phase for one ROI, joined to the design.

    Returns dict(P (N, V) float64, trials DataFrame aligned to P's rows,
    meanvol (V,) or None, voxidx (V,), blocks (V,) int or None,
    blocknames list or None).
    """
    arm = arm_map[phase]
    p = cache_path(cache_root, subject, arm, beta_type)
    if not p.exists():
        sys.exit(f"ERROR: cache missing: {p}\n  run extract_roi_betas.py --subject {subject} "
                 f"--arm {arm} --roi-set {ROI_SET}")
    d = np.load(p, allow_pickle=True)
    if f"patterns_{roi}" not in d.files:
        sys.exit(f"ERROR: {p.name} has no ROI {roi!r}; ROIs: "
                 f"{[k[9:] for k in d.files if k.startswith('patterns_')]}")
    P = np.asarray(d[f"patterns_{roi}"], dtype=np.float64).T          # (N, V)
    keys = list(zip([phase] * len(P), d["session"].astype(str),
                    d["run"].astype(int), np.round(d["onset"].astype(float), 3)))
    dsub = design[design["phase"] == phase].set_index("key")
    missing = [k for k in keys if k not in dsub.index]
    if missing:
        sys.exit(f"ERROR: {len(missing)} {phase} trials of {p.name} not in the design table "
                 f"(e.g. {missing[:3]}); (session, run, onset) matching failed")
    trials = dsub.loc[keys].reset_index(drop=True)
    trials["run_key"] = trials["session"] + "/" + phase + "/" + trials["run"].astype(str)
    mv = d[f"meanvol_{roi}"] if f"meanvol_{roi}" in d.files else None
    blk = d[f"blocks_{roi}"].astype(int) if f"blocks_{roi}" in d.files else None
    names = list(d[f"blocknames_{roi}"]) if f"blocknames_{roi}" in d.files else None
    return {"P": P, "trials": trials, "meanvol": mv, "voxidx": d[f"voxidx_{roi}"],
            "blocks": blk, "blocknames": names, "path": str(p)}


def load_features(path: Path, design: pd.DataFrame) -> pd.DataFrame:
    """(items x features) indexed by mmmId, from a stimulus-feature CSV."""
    if not path.exists():
        sys.exit(f"ERROR: feature file missing: {path}")
    feat = pd.read_csv(path)
    feat = feat.set_index("stimulus_id")
    num = feat.select_dtypes(include=[np.number])
    sid = design.drop_duplicates("mmmId").set_index("stimulus_id")["mmmId"]
    num = num.loc[num.index.intersection(sid.index)]
    num.index = sid.loc[num.index].to_numpy()
    return num


# ── items ────────────────────────────────────────────────────────────────────

def item_sets(trials_by_phase: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Non-anchor items present in every phase; their folds; per-item info."""
    common = None
    for t in trials_by_phase.values():
        s = set(t.loc[~t["anchor"].astype(bool), "mmmId"].astype(int))
        common = s if common is None else common & s
    items = np.array(sorted(common))
    # per-item facts (fold, age at retrieval, enCon) come from a retrieval
    # phase: encoding rows carry age 0 by construction
    ret_phase = next((p for p in trials_by_phase if p != "enc"), next(iter(trials_by_phase)))
    info = trials_by_phase[ret_phase].drop_duplicates("mmmId").set_index("mmmId")
    folds = info.loc[items, "fold"].astype(int).to_numpy()
    return items, folds, info


def phase_matrix(arm: dict, items: np.ndarray, phase: str):
    """(X (n_items, V), run_keys (n_items,), n_exposures (n_items,)).
    Encoding = mean over exposures with exposure-1's run as the run key;
    retrieval = the single trial."""
    t = arm["trials"]
    ids = t["mmmId"].astype(int).to_numpy()
    X, n = basis.average_exposures(arm["P"], ids, items)
    first = t[t["exposure"] == 1].drop_duplicates("mmmId").set_index("mmmId")["run_key"]
    runs = first.reindex(items).to_numpy()
    return X, runs, n


def recon_strata(info: pd.DataFrame, items: np.ndarray) -> dict:
    """{"reCon1": mask, "reCon2": mask} over ``items`` from the retrieval
    phase's reCon (1 = retrieved in the encoding session, 2 = a later one).
    Both retrievals of an item share a session, so it is a per-item fact."""
    rc = info.loc[items, "reCon"].to_numpy(float)
    return {"reCon1": rc == 1, "reCon2": rc == 2}


VARIANT_SUFFIXES = ("_reCon1", "_reCon2", "_ntf", "_ntf_reCon1", "_ntf_reCon2")


def score_variants(info: pd.DataFrame, items: np.ndarray) -> dict:
    """{suffix: (target_mask, candidate_mask)} over ``items``: the reCon strata
    (targets restricted, foils shared) and the non-triplet-foil pool ``_ntf``
    (foils restricted to enCon != 3 for EVERY target, so 2AFC stays comparable
    across targets; DECIDED Ben 2026-09-23: a triplet's sequence-mates are
    privileged foils through encoding autocorrelation), and their crossing."""
    rc = info.loc[items, "reCon"].to_numpy(float)
    ntf = info.loc[items, "enCon"].to_numpy(float) != 3
    return {"_reCon1": (rc == 1, None), "_reCon2": (rc == 2, None), "_ntf": (None, ntf),
            "_ntf_reCon1": (rc == 1, ntf), "_ntf_reCon2": (rc == 2, ntf)}


# ── one fold ─────────────────────────────────────────────────────────────────

def fit_classes(k, W, En, Rn, Ev, F_tr, seed, with_features, F_te=None,
                pinned: dict | None = None, only=None):
    """Fit every class (or ``only`` those named) at basis rank k, projecting
    the voxel-space matrices first. Callers that loop (nulls, inner folds)
    project ONCE and use fit_classes_proj: the projection is the whole cost
    of a union cell (1,900 x 72k x 200 per call; 4,000 calls per fold took
    45 min before 2026-09-23).

    ``pinned`` = {class name: fitted object} whose hyperparameters are reused
    (no inner search). Returns {name: (obj, R_hat_test)}.
    """
    Wk = W[:, :k]
    return fit_classes_proj(En @ Wk, Rn @ Wk, Ev @ Wk, F_tr, seed, with_features,
                            F_te=F_te, pinned=pinned, only=only)


def fit_classes_proj(Et, Rt, Ee, F_tr, seed, with_features, F_te=None,
                     pinned: dict | None = None, only=None):
    """fit_classes on already-projected (n, k) matrices."""
    out = {}
    for cls in maps.make_classes(seed=seed, with_features=with_features and F_tr is not None):
        if only is not None and cls.name not in only:
            continue
        if pinned and cls.name in pinned:
            cls = pinned[cls.name].pinned()
        cls.fit(Et, Rt, F=F_tr)
        out[cls.name] = (cls, cls.predict(Ee))
        if cls.name == "procrustes" and cls.det_sign < 0:
            proper = maps.OrthogonalProcrustes(proper=True).fit(Et, Rt)
            out["procrustes_proper"] = (proper, proper.predict(Ee))
        if cls.name == "semantic_warp" and F_te is not None:
            out["semantic_oracle"] = (cls, cls.predict_oracle(F_te))
    return out


def nested_rank(ranks, W, En, Rn, F_tr, seed, with_features, pinned_by_k: dict, n_inner=5):
    """Per class, the rank with the best mean inner-fold 2AFC, the basis
    fixed and each class's hyperparameters pinned from its training fit."""
    rng = np.random.default_rng(seed)
    folds = basis.inner_folds(len(En), n_inner, rng)
    scores = {}
    for k in ranks:
        Wk = W[:, :k]
        Ek, Rk = En @ Wk, Rn @ Wk                     # once per rank, not per inner fold
        for tr, te in folds:
            Ft = F_tr[tr] if F_tr is not None else None
            fitted = fit_classes_proj(Ek[tr], Rk[tr], Ek[te], Ft, seed, with_features,
                                      pinned=pinned_by_k[k])
            for name, (_, R_hat) in fitted.items():
                scores.setdefault(name, {}).setdefault(k, []).append(
                    maps._score_2afc_all(R_hat, Rk[te]))
    return {name: max(d, key=lambda k: np.mean(d[k])) for name, d in scores.items()}


def rotation_rows(Q, Et, Rt, meta: dict, gate_passed: bool) -> list:
    bl = rotation.plane_decomposition(Q, Et)
    geo, reflected = rotation.geodesic_distance(bl)
    rows = [dict(meta, metric="mean_plane_angle_deg", subspace_m="n/a",
                 value=np.degrees(rotation.mean_plane_angle(bl)), gate_passed=gate_passed),
            dict(meta, metric="geodesic_rad", subspace_m="n/a", value=geo,
                 gate_passed=gate_passed, reflected=reflected)]
    for m in PRINCIPAL_M:
        if m <= Et.shape[1]:
            pa = rotation.principal_angles(Et, Rt, m)
            rows.append(dict(meta, metric="principal_mean_cos", subspace_m=m,
                             value=pa["mean_cos"], gate_passed=gate_passed))
            rows.append(dict(meta, metric="enc_var_in_ret", subspace_m=m,
                             value=pa["share_enc_var_in_ret"], gate_passed=gate_passed))
    return rows


def run_fold(f, items, folds, E, R, runs_E, runs_R, enc_arm, F, blocks_vox, args, ctx):
    """Every table row of one outer fold. ctx carries constants + collectors."""
    tr, te = folds != f, folds == f
    if te.sum() < MIN_TEST_ITEMS:
        return
    seed = args.seed + f
    variants = score_variants(ctx["info"], items[te])
    # reliability preselection on training three-exposure items, per block
    t_enc = enc_arm["trials"]
    ids_enc = t_enc["mmmId"].astype(int).to_numpy()
    if args.preselect >= 1.0:
        keep = np.ones(E.shape[1], dtype=bool)          # plain PCA: every voxel, no reliability pass
    else:
        rel = basis.split_half_reliability(enc_arm["P"], ids_enc,
                                           t_enc["exposure"].to_numpy(), items[tr])
        keep = basis.preselect(rel, args.preselect, blocks_vox)
    blk = blocks_vox[keep] if blocks_vox is not None else None
    nm = basis.BlockNormaliser().fit(E[tr][:, keep], R[tr][:, keep], blk)
    En, Rn = nm.transform(E[tr][:, keep], "E"), nm.transform(R[tr][:, keep], "R")
    Ev, Rv = nm.transform(E[te][:, keep], "E"), nm.transform(R[te][:, keep], "R")
    W, explained = basis.shared_basis(En, Rn, k_max=args.k_max)
    ranks = basis.resolve_ranks(RANK_GRID, W.shape[1])
    F_tr = F[tr] if F is not None else None
    F_te = F[te] if F is not None else None

    # encoding halves (exposure 1 vs mean of 2+3) of test and training items
    A_te, B_te, kept_te = basis.exposure_halves(enc_arm["P"], ids_enc, t_enc["exposure"].to_numpy(), items[te])
    A_tr, B_tr, kept_tr = basis.exposure_halves(enc_arm["P"], ids_enc, t_enc["exposure"].to_numpy(), items[tr])
    runs_A = pd.Series(runs_E, index=items).loc[kept_te].to_numpy()
    An_te, Bn_te = nm.transform(A_te[:, keep], "E"), nm.transform(B_te[:, keep], "E")
    An_tr, Bn_tr = nm.transform(A_tr[:, keep], "E"), nm.transform(B_tr[:, keep], "E")

    base = {"subject": args.subject, "rung": args.rung, "roi": args.roi, "pair": args.pair,
            "beta_type": args.beta_type, "preselect_frac": args.preselect, "fold": int(f),
            "n_test_items": int(te.sum()), "n_vox": int(keep.sum())}
    ctx["fold_meta"].append(dict(base, n_train_items=int(tr.sum()), k_max=int(W.shape[1]),
                                 explained_top10=float(explained[:10].sum()),
                                 translation_norm=nm.translation_norm))

    # rank sweep: every class at every rank, forward and reverse
    with_f = args.features is not None
    fitted_by_k, rows_by = {}, {}
    for k in ranks:
        Wk = W[:, :k]
        ceil = score.encoding_ceiling(An_te @ Wk, Bn_te @ Wk, runs_A)["acc_2afc"]
        fwd = fit_classes(k, W, En, Rn, Ev, F_tr, seed, with_f, F_te=F_te)
        rev = fit_classes(k, W, Rn, En, Rv, None, seed, False)
        fitted_by_k[k] = {n: o for n, (o, _) in fwd.items()}
        id_fwd = score.identify_variants(fwd["identity"][1], Rv @ Wk, runs_R[te], variants)
        id_rev = score.identify_variants(rev["identity"][1], Ev @ Wk, runs_E[te], variants)
        for direction, fitted, truth, runs, id_acc in (
                ("forward", fwd, Rv @ Wk, runs_R[te], id_fwd),
                ("reverse", rev, Ev @ Wk, runs_E[te], id_rev)):
            for name, (obj, R_hat) in fitted.items():
                sc = score.identify_variants(R_hat, truth, runs, variants)
                gain = sc["acc_2afc"] - id_acc["acc_2afc"]
                row = dict(base, direction=direction, **{"class": name},
                           det_sign=getattr(obj, "det_sign", "n/a"), rank=k, rank_selected=False,
                           n_params=obj.n_params, acc_2afc=sc["acc_2afc"], acc_rank=sc["acc_rank"],
                           n_pairs=sc["n_pairs"], gain=gain, ceiling=ceil,
                           gain_frac_ceiling=score.gain_fraction(gain, ceil, CHANCE))
                for s in variants:        # reCon strata x foil pools (see score_variants)
                    row[f"acc_2afc{s}"] = sc[f"acc_2afc{s}"]
                    row[f"n_items{s}"] = sc[f"n_items{s}"]
                    row[f"gain{s}"] = sc[f"acc_2afc{s}"] - id_acc[f"acc_2afc{s}"]
                ctx["class_rows"].append(row)
                rows_by[(direction, name, k)] = row
        ctx["geometry_rows"].append(dict(
            base, rank=k, rdm_corr=score.rdm_correlation(Ev @ Wk, Rv @ Wk),
            ceiling_rdm=score.rdm_correlation(An_te @ Wk, Bn_te @ Wk)))

    # headline rank per class by nested CV inside training
    chosen = nested_rank(ranks, W, En, Rn, F_tr, seed, with_f, fitted_by_k)
    for name, k in chosen.items():
        for direction in ("forward", "reverse"):
            if (direction, name, k) in rows_by:
                rows_by[(direction, name, k)]["rank_selected"] = True
    if "procrustes" not in chosen:
        chosen["procrustes"] = ranks[0]

    # permutation null at the chosen rank, per class (training side, within run).
    # Skipped (lossless) when the class's gain over identity is <= 0 in every
    # scoring variant: the gate needs gain > 0, so no null could pass it.
    rng = np.random.default_rng(seed)
    proj = {}                                          # k -> (Et, Rt, Ee, truth) projected once
    for name, k in chosen.items():
        obs = rows_by.get(("forward", name, k))
        if obs is None or name in ("identity", "semantic_oracle", "procrustes_proper"):
            continue
        gains = [obs.get(f"gain{sfx}", np.nan) for sfx in ("",) + VARIANT_SUFFIXES]
        if not any(np.isfinite(g) and g > 0 for g in gains):
            obs["null_n"] = 0
            obs["null_skipped"] = "gain<=0 in every variant"
            continue
        if k not in proj:
            Wk = W[:, :k]
            proj[k] = (En @ Wk, Rn @ Wk, Ev @ Wk, Rv @ Wk)
        Et, Rt, Ee, truth = proj[k]
        nulls = []
        n_draws = args.n_perm if k <= HIGH_RANK else min(args.n_perm, args.n_perm_high)
        for _ in range(n_draws):
            perm = score.permute_within_run(rng, runs_R[tr])
            fitted = fit_classes_proj(Et, Rt[perm], Ee, F_tr, seed, with_f,
                                      pinned=fitted_by_k[k], only={name})
            nulls.append(score.identify_variants(fitted[name][1], truth, runs_R[te], variants))
        all_ = [n["acc_2afc"] for n in nulls]
        obs["null_n"] = len(nulls)
        obs["null_p"] = score.null_p(obs["acc_2afc"], all_)
        obs["null_mean"] = float(np.mean(all_)) if nulls else float("nan")
        obs["null_q95"] = float(np.quantile(all_, 0.95)) if nulls else float("nan")
        for s in variants:
            obs[f"null_p{s}"] = score.null_p(obs[f"acc_2afc{s}"], [n[f"acc_2afc{s}"] for n in nulls])

    # rotation metric of the orthogonal map at its chosen rank
    k = chosen["procrustes"]
    Wk = W[:, :k]
    Et, Rt = En @ Wk, Rn @ Wk
    pro = maps.OrthogonalProcrustes().fit(Et, Rt)
    obs = rows_by[("forward", "procrustes", k)]
    gate = bool(np.isfinite(obs.get("null_p", np.nan)) and obs["null_p"] < 0.05 and obs["gain"] > 0)
    meta = dict(base, rank=k, det_sign=pro.det_sign)
    ctx["rotation_rows"] += [dict(r, arm="fit") for r in rotation_rows(pro.Q, Et, Rt, meta, gate)]
    ctx["rotation_rows"].append(dict(meta, metric="translation_norm", subspace_m="n/a",
                                     value=nm.translation_norm, gate_passed=gate, arm="fit"))
    # encoding split-half floor: Q between exposure halves of training items
    Af, Bf = An_tr @ Wk, Bn_tr @ Wk
    floor = maps.OrthogonalProcrustes().fit(Af, Bf)
    ctx["rotation_rows"] += [dict(r, arm="floor_enc") for r in rotation_rows(floor.Q, Af, Bf, meta, gate)]
    # item-permuted null (fewer draws above HIGH_RANK: each is a k^3 Schur)
    n_rot = args.n_perm_rot if k <= HIGH_RANK else min(args.n_perm_rot, args.n_perm_rot_high)
    for i in range(n_rot):
        perm = score.permute_within_run(rng, runs_R[tr])
        nq = maps.OrthogonalProcrustes().fit(Et, Rt[perm])
        # arm label is "permuted", never "null": pandas reads a literal "null" as NaN
        ctx["rotation_rows"] += [dict(r, arm="permuted", draw=i) for r in rotation_rows(nq.Q, Et, Rt[perm], meta, gate)]
    # block energy at union rungs
    if blk is not None and len(np.unique(blk)) > 1:
        be = blocks_mod.block_energy(pro.Q, Wk, blk)
        names = ctx["blocknames"]
        for a, la in enumerate(be["blocks"]):
            for b, lb in enumerate(be["blocks"]):
                ctx["block_rows"].append(dict(
                    base, rank=k, source_block=names[la - 1] if names else int(la),
                    target_block=names[lb - 1] if names else int(lb),
                    energy_frac=float(be["energy"][a, b]), cross_fraction=be["cross_fraction"]))
    # per-item alignment residual on held-out items (basis-free across folds)
    R_hat = pro.predict(Ev @ Wk)
    Rk = Rv @ Wk
    c = np.array([np.corrcoef(R_hat[i], Rk[i])[0, 1] for i in range(len(Rk))])
    info = ctx["info"]
    for it, r in zip(items[te], c):
        ctx["item_rows"].append(dict(base, rank=k, mmmId=int(it),
                                     age_days=float(info.loc[it, "age_days"]),
                                     enCon=int(info.loc[it, "enCon"]),
                                     alignment_residual=float(1 - r)))


# ── pooled fit: planes, spectrum, age slopes, anchors ────────────────────────

def pooled_fit(items, E, R, runs_R, enc_arm, arms, blocks_vox, k_star, args, ctx):
    t_enc = enc_arm["trials"]
    ids_enc = t_enc["mmmId"].astype(int).to_numpy()
    rel = basis.split_half_reliability(enc_arm["P"], ids_enc, t_enc["exposure"].to_numpy(), items)
    keep = basis.preselect(rel, args.preselect, blocks_vox)
    blk = blocks_vox[keep] if blocks_vox is not None else None
    nm = basis.BlockNormaliser().fit(E[:, keep], R[:, keep], blk)
    En, Rn = nm.transform(E[:, keep], "E"), nm.transform(R[:, keep], "R")
    W, _ = basis.shared_basis(En, Rn, k_max=args.k_max)
    k = min(k_star, W.shape[1])
    Wk = W[:, :k]
    Et, Rt = En @ Wk, Rn @ Wk
    pro = maps.OrthogonalProcrustes().fit(Et, Rt)
    bl = rotation.plane_decomposition(pro.Q, Et)
    planes = [b for b in bl if b["kind"] == "plane"]
    A, B, kept = basis.exposure_halves(enc_arm["P"], ids_enc, t_enc["exposure"].to_numpy(), items)
    An, Bn = nm.transform(A[:, keep], "E") @ Wk, nm.transform(B[:, keep], "E") @ Wk
    info = ctx["info"]
    age = info.loc[items, "age_days"].to_numpy(float)
    encon = info.loc[items, "enCon"].to_numpy(int)
    base = {"subject": args.subject, "rung": args.rung, "roi": args.roi, "pair": args.pair,
            "beta_type": args.beta_type, "rank": k, "basis": "pooled"}
    spectrum = []
    for j, b in enumerate(planes, start=1):
        ang = rotation.unwrap_to_reference(rotation.per_item_plane_angle(Et, Rt, b["P"]), b["theta"])
        floor = rotation.unwrap_to_reference(rotation.per_item_plane_angle(An, Bn, b["P"]), 0.0)
        for group, mask in (("single", encon == 1), ("three", encon == 3)):
            s = rotation.age_slopes(ang, age, mask, seed=args.seed)
            row = dict(base, plane_rank=j, plane_weight=b["weight"],
                       plane_theta_deg=np.degrees(b["theta"]), measure="plane_angle",
                       exposure_group=group, **s,
                       floor_enc_mean_abs_deg=float(np.degrees(np.nanmean(np.abs(floor)))))
            if group == "single" and b["weight"] > args.min_plane_weight:
                spectrum.append(row)
            if j <= 3:
                ctx["delay_rows"].append(row)
    # alignment residual (held-out, pooled across folds) on age
    ir = pd.DataFrame(ctx["item_rows"])
    if len(ir):
        for group, en in (("single", 1), ("three", 3)):
            sub = ir[ir["enCon"] == en]
            s = rotation.age_slopes(np.radians(sub["alignment_residual"].to_numpy()),
                                    sub["age_days"].to_numpy(), np.ones(len(sub), bool), seed=args.seed)
            s["slope_per_day"] = np.degrees(s["slope_per_day"]) if np.isfinite(s["slope_per_day"]) else s["slope_per_day"]
            s["ci_lo"], s["ci_hi"] = (np.degrees(s["ci_lo"]), np.degrees(s["ci_hi"])) if np.isfinite(s["ci_lo"]) else (s["ci_lo"], s["ci_hi"])
            ctx["delay_rows"].append(dict(base, basis="held-out", plane_rank="n/a", plane_weight="n/a",
                                          plane_theta_deg="n/a", measure="alignment_residual",
                                          exposure_group=group, **s, floor_enc_mean_abs_deg="n/a"))
    spectrum.sort(key=lambda r: -r["slope_per_day"] if np.isfinite(r["slope_per_day"]) else np.inf)
    split = rotation.content_position_split([{"weight": r["plane_weight"], "age_dependent": r["age_dependent"]}
                                             for r in spectrum]) if spectrum else {}
    for r in spectrum:
        ctx["spectrum_rows"].append(dict(r, **split))
    # anchors: same-item retrieval similarity across sessions + top-plane angle delta
    lag_path = ctx["design_root"] / args.subject / f"{args.subject}_desc-anchorlags.tsv"
    phases = args.pair.split(":")
    ret_phase = phases[1]
    if lag_path.exists() and ret_phase != "enc":
        lags = pd.read_csv(lag_path, sep="\t", na_values=["n/a"])
        lags = lags[lags["cue"] == ret_phase[4:]]
        t = arms[ret_phase]["trials"]
        Pn = nm.transform(arms[ret_phase]["P"][:, keep], "R")
        top = planes[0]["P"] if planes else None
        for _, row in lags.iterrows():
            a = t.index[(t["mmmId"] == row["mmmId"]) & (t["session"] == row["ses_a"])]
            b = t.index[(t["mmmId"] == row["mmmId"]) & (t["session"] == row["ses_b"])]
            if len(a) != 1 or len(b) != 1:
                continue
            xa, xb = Pn[a[0]], Pn[b[0]]
            sim = float(np.corrcoef(xa, xb)[0, 1])
            delta = float(np.degrees(rotation.per_item_plane_angle((xa @ Wk)[None], (xb @ Wk)[None], top)[0])) if top is not None else np.nan
            ctx["anchor_rows"].append(dict(base, cue=row["cue"], mmmId=int(row["mmmId"]),
                                           ses_a=row["ses_a"], ses_b=row["ses_b"], lag_days=row["lag_days"],
                                           abs_time_days=row["abs_time_days"], similarity=sim,
                                           plane_angle_delta=delta))
        if ctx["anchor_rows"]:
            ar = pd.DataFrame(ctx["anchor_rows"])
            for y in ("similarity", "plane_angle_delta"):
                reg = rotation.anchor_regression(ar[y], ar["lag_days"], ar["abs_time_days"], seed=args.seed)
                ctx["anchor_reg_rows"].append(dict(base, cue=ret_phase[4:], response=y, **reg))


# ── composition ──────────────────────────────────────────────────────────────

def composition(items, folds, X: dict, runs: dict, enc_arm, blocks_vox, args, ctx):
    """E -> R_word -> R_image composed vs direct E -> R_image, per fold."""
    t_enc = enc_arm["trials"]
    ids_enc = t_enc["mmmId"].astype(int).to_numpy()
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if te.sum() < MIN_TEST_ITEMS:
            continue
        variants = score_variants(ctx["info"], items[te])
        rel = basis.split_half_reliability(enc_arm["P"], ids_enc, t_enc["exposure"].to_numpy(), items[tr])
        keep = basis.preselect(rel, args.preselect, blocks_vox)
        # one normaliser over the three phases: centre each, scale jointly
        means = {p: X[p][tr][:, keep].mean(axis=0) for p in X}
        stacked = np.vstack([X[p][tr][:, keep] - means[p] for p in X])
        scale = np.sqrt((stacked ** 2).sum() / keep.sum()) or 1.0
        Z = {p: (X[p][:, keep] - means[p]) / scale for p in X}
        S = np.vstack([Z[p][tr] for p in X])
        _, s, Vt = basis.robust_svd(S - S.mean(axis=0))
        k_max = min(args.k_max or Vt.shape[0], Vt.shape[0])
        W = Vt[:k_max].T
        for k in basis.resolve_ranks(RANK_GRID, k_max):
            Wk = W[:, :k]
            E, Rw, Ri = (Z[p] @ Wk for p in ("enc", "ret-word", "ret-image"))
            q1 = maps.OrthogonalProcrustes().fit(E[tr], Rw[tr])
            q2 = maps.OrthogonalProcrustes().fit(Rw[tr], Ri[tr])
            q3 = maps.OrthogonalProcrustes().fit(E[tr], Ri[tr])
            sd = score.identify_variants(E[te] @ q3.Q, Ri[te], runs["ret-image"][te], variants)
            sc = score.identify_variants(E[te] @ q1.Q @ q2.Q, Ri[te], runs["ret-image"][te], variants)
            si = score.identify_variants(E[te], Ri[te], runs["ret-image"][te], variants)
            consistency = float(np.linalg.norm(q1.Q @ q2.Q - q3.Q) / np.sqrt(k))
            row = dict(
                subject=args.subject, rung=args.rung, roi=args.roi, pair=args.pair,
                beta_type=args.beta_type, fold=int(f), rank=k, n_test_items=int(te.sum()),
                n_vox=int(keep.sum()), acc_identity=si["acc_2afc"], acc_direct=sd["acc_2afc"],
                acc_composed=sc["acc_2afc"], shortfall=sd["acc_2afc"] - sc["acc_2afc"],
                map_inconsistency=consistency)
            for s in variants:
                row[f"acc_identity{s}"], row[f"acc_direct{s}"], row[f"acc_composed{s}"] = (
                    si[f"acc_2afc{s}"], sd[f"acc_2afc{s}"], sc[f"acc_2afc{s}"])
                row[f"shortfall{s}"] = sd[f"acc_2afc{s}"] - sc[f"acc_2afc{s}"]
            ctx["composition_rows"].append(row)


# ── checkpoint (a fold at a time, so a wall-time kill loses one fold) ────────

CKPT_KEYS = ("subject", "rung", "roi", "pair", "beta_type", "preselect", "k_max", "n_perm",
             "n_perm_rot", "n_perm_high", "n_perm_rot_high", "features", "arm_suffix", "seed")
CKPT_ROWS = ("class_rows", "rotation_rows", "geometry_rows", "item_rows", "block_rows", "fold_meta")


def _ckpt_args(args) -> dict:
    return {k: getattr(args, k) for k in CKPT_KEYS}


def save_checkpoint(path: Path, ctx: dict, args, done_folds: set) -> None:
    import pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"args": _ckpt_args(args), "done_folds": sorted(int(x) for x in done_folds),
                     "rows": {k: ctx[k] for k in CKPT_ROWS}}, f)
    tmp.replace(path)


def load_checkpoint(path: Path, ctx: dict, args) -> set:
    """Resume finished folds from a partial file written by an earlier run
    with the same arguments; a mismatch is ignored with a printed line."""
    import pickle
    if not path.exists():
        return set()
    with open(path, "rb") as f:
        d = pickle.load(f)
    if d.get("args") != _ckpt_args(args):
        print(f"  checkpoint {path.name} was written with different arguments; ignored")
        return set()
    for k in CKPT_ROWS:
        ctx[k] = d["rows"][k]
    done = set(d["done_folds"])
    print(f"  resumed {len(done)} finished folds from {path.name}: {sorted(done)}", flush=True)
    return done


# ── main ─────────────────────────────────────────────────────────────────────

def write_tables(out_dir: Path, stem: str, ctx: dict, args, t0: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {"transformation_class": ctx["class_rows"], "rotation_metrics": ctx["rotation_rows"],
              "geometry": ctx["geometry_rows"], "item_residuals": ctx["item_rows"],
              "delay_slopes": ctx["delay_rows"], "plane_spectrum": ctx["spectrum_rows"],
              "anchor_drift": ctx["anchor_rows"], "anchor_regression": ctx["anchor_reg_rows"],
              "block_energy": ctx["block_rows"], "composition": ctx["composition_rows"]}
    written = []
    for name, rows in tables.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if name == "transformation_class":
            for col in ["null_p", "null_mean", "null_q95", "null_n", "null_skipped"] + [f"null_p{s}" for s in VARIANT_SUFFIXES]:
                if col not in df:
                    df[col] = np.nan
            # fold CIs of gain per (direction, class, rank) -> ci_lo / ci_hi on every row,
            # and the same per variant (ci_lo_reCon1, ci_lo_ntf, ...)
            for suffix in ("",) + VARIANT_SUFFIXES:
                gcol = f"gain{suffix}"
                if gcol not in df:
                    continue
                g = df.groupby(["direction", "class", "rank"])[gcol]
                ci = g.apply(lambda v: pd.Series(score.fold_ci(v.to_numpy(), seed=args.seed)[1:],
                                                 index=[f"ci_lo{suffix}", f"ci_hi{suffix}"]))
                ci = ci.unstack() if isinstance(ci, pd.Series) else ci
                df = df.merge(ci.reset_index(), on=["direction", "class", "rank"], how="left")
            df["gate_passed"] = (df["null_p"] < 0.05) & (df["gain"] > 0)
            for s in VARIANT_SUFFIXES:
                if f"gain{s}" in df:
                    df[f"gate_passed{s}"] = (df[f"null_p{s}"] < 0.05) & (df[f"gain{s}"] > 0)
        p = out_dir / f"{stem}_{name}.tsv"
        df.to_csv(p, sep="\t", index=False, na_rep="n/a", float_format="%.6g")
        written.append(p.name)
    with open(out_dir / f"{stem}_fit.json", "w") as f:
        json.dump({"args": vars(args), "folds": ctx["fold_meta"], "tables": written,
                   "seconds": round(time.time() - t0, 1), "caches": ctx["caches"]}, f, indent=2, default=str)
    print(f"wrote {len(written)} tables to {out_dir}/{stem}_* in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--rung", required=True, choices=["i", "ii", "iii", "iv"])
    ap.add_argument("--roi", required=True, help="patterns_<ROI> key in the ladder cache")
    ap.add_argument("--pair", required=True,
                    help="enc:ret-word | enc:ret-image | ret-word:ret-image | enc:ret-word:ret-image")
    ap.add_argument("--beta-type", default="D", choices=["B", "C", "D"])
    ap.add_argument("--preselect", type=float, default=1.0,
                    help="fraction of voxels kept per block by encoding reliability (1.0 = plain PCA, "
                         "the default since 2026-09-23: 0.5 dropped the word-cued signal in mPFC/AG)")
    ap.add_argument("--k-max", type=int, default=None, help="cap on the basis dimension ('full')")
    ap.add_argument("--n-perm", type=int, default=200, help="identity-permutation null draws")
    ap.add_argument("--n-perm-rot", type=int, default=50, help="item-permuted null draws for the rotation metric")
    ap.add_argument("--n-perm-high", type=int, default=200,
                    help=f"identity-null draws when the chosen rank exceeds {HIGH_RANK} (each draw "
                         "costs k^3; 1000 draws at rank ~1,860 took an hour per fold on 2026-09-22)")
    ap.add_argument("--n-perm-rot-high", type=int, default=20,
                    help=f"rotation-null draws when the chosen rank exceeds {HIGH_RANK}")
    ap.add_argument("--min-plane-weight", type=float, default=0.01)
    ap.add_argument("--features", default="clip",
                    help="feature CSV stem under stimuli_features/shared1000 for the semantic warp, "
                         "a path, or 'none'")
    ap.add_argument("--arm-suffix", default="-tbonly",
                    help="retrieval arm suffix ('' for the TB+FIN production arms)")
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--cache-root", default=None, help=f"override <output_dir>/{CACHE_TREE}")
    ap.add_argument("--design-root", default=None, help=f"override <output_dir>/{DESIGN_TREE}")
    ap.add_argument("--out-root", default=None, help=f"override <output_dir>/{DESIGN_TREE}")
    args = ap.parse_args()

    t0 = time.time()
    cfg = tb.load_config()
    output_dir = Path(cfg["output_dir"])
    cache_root = Path(args.cache_root) if args.cache_root else output_dir / CACHE_TREE
    design_root = Path(args.design_root) if args.design_root else output_dir / DESIGN_TREE
    out_root = Path(args.out_root) if args.out_root else output_dir / DESIGN_TREE
    arm_map = {p: (a if p == "enc" else a.replace("-tbonly", args.arm_suffix)) for p, a in PHASE_ARMS.items()}
    phases = args.pair.split(":")
    if len(phases) not in (2, 3) or any(p not in PHASE_ARMS for p in phases):
        sys.exit(f"ERROR: bad --pair {args.pair!r}")
    if args.features == "none":
        args.features = None
    elif not Path(args.features).exists():
        args.features = str(output_dir / "stimuli_features" / "shared1000" / f"{args.features}.csv")

    print(f"=== fit_pair: {args.subject} rung {args.rung} {args.roi} {args.pair} TYPE{args.beta_type} ===")
    design = load_design(design_root, args.subject)
    arms = {p: load_arm(cache_root, args.subject, p, args.beta_type, args.roi, design, arm_map) for p in phases}

    # voxel cleaning across every phase
    keep = np.ones(arms[phases[0]]["P"].shape[1], dtype=bool)
    for a in arms.values():
        keep &= basis.clean_voxels(a["P"].T, a["meanvol"], 0.25, 100.0)
    for a in arms.values():
        a["P"] = a["P"][:, keep]
    blocks_vox = arms[phases[0]]["blocks"]
    blocks_vox = blocks_vox[keep] if blocks_vox is not None else None
    print(f"voxels: {keep.size} in ROI, {int(keep.sum())} after cleaning"
          + (f"; blocks {dict(zip(*np.unique(blocks_vox, return_counts=True)))}" if blocks_vox is not None else ""))

    items, folds, info = item_sets({p: a["trials"] for p, a in arms.items()})
    print(f"{len(items)} non-anchor items in every phase; folds {np.unique(folds).size}")
    X, runs = {}, {}
    for p in phases:
        X[p], runs[p], _ = phase_matrix(arms[p], items, p)
    ctx = {k: [] for k in ("class_rows", "rotation_rows", "geometry_rows", "item_rows", "delay_rows",
                           "spectrum_rows", "anchor_rows", "anchor_reg_rows", "block_rows",
                           "composition_rows", "fold_meta")}
    ctx.update(info=info, design_root=design_root, blocknames=arms[phases[0]]["blocknames"],
               caches={p: a["path"] for p, a in arms.items()})
    stem = f"{args.subject}_rung-{args.rung}_roi-{args.roi}_pair-{args.pair.replace(':', '')}_desc-type{args.beta_type.lower()}"

    enc_arm = arms["enc"] if "enc" in arms else None
    if len(phases) == 3:
        composition(items, folds, X, runs, enc_arm, blocks_vox, args, ctx)
        write_tables(out_root / "fits" / args.subject, stem, ctx, args, t0)
        return

    src, dst = phases
    if enc_arm is None:
        # ret-word <-> ret-image: reliability/halves still come from encoding
        enc_arm = load_arm(cache_root, args.subject, "enc", args.beta_type, args.roi, design, arm_map)
        enc_arm["P"] = enc_arm["P"][:, keep]
    F = None
    if args.features:
        feat = load_features(Path(args.features), design)
        F = feat.reindex(items).to_numpy(float)
        if np.isnan(F).any():
            sys.exit(f"ERROR: {int(np.isnan(F).any(axis=1).sum())} items lack features in {args.features}")
    ckpt = out_root / "fits" / args.subject / f"{stem}_partial.pkl"
    done_folds = load_checkpoint(ckpt, ctx, args)
    for f in np.unique(folds):
        if f in done_folds:
            continue
        run_fold(f, items, folds, X[src], X[dst], runs[src], runs[dst], enc_arm, F, blocks_vox, args, ctx)
        done_folds.add(int(f))
        save_checkpoint(ckpt, ctx, args, done_folds)
        print(f"  fold {f:2d} done ({time.time() - t0:.0f}s)", flush=True)
    ks = [r["rank"] for r in ctx["class_rows"] if r["class"] == "procrustes" and r["rank_selected"]
          and r["direction"] == "forward"]
    k_star = int(np.median(ks)) if ks else 10
    pooled_fit(items, X[src], X[dst], runs[dst], enc_arm, arms, blocks_vox, k_star, args, ctx)
    write_tables(out_root / "fits" / args.subject, stem, ctx, args, t0)
    if ckpt.exists():
        ckpt.unlink()


if __name__ == "__main__":
    main()
