#!/usr/bin/env python3
"""Connectivity hyperalignment (CHA): the zero-overlap comparator route.

Pre-registration §6 (CHA row), §8 (3 densification levels); targets DECIDED
2026-09-29 in mmmdata-agents ``docs/workbench/functional-space/``. One job per
target subject; the template is built from the other two and frozen before the
target enters, and the target's transform comes from its own data only (§2).

Data: every rest run of the subject (INIT/TB/NAT/FIN), cleaned grayordinate
series, each run z-scored per column over its steady-state volumes, then
concatenated. A column is used only if it is valid in every run.

Per level (ico3 -> ico4 -> ico5, see ``pieces.py``):

  1. targets: each target's signal is the mean of its member grayordinates in
     the subject's data, taken **in template space** (the data mapped through
     the subject's transform from the previous level; identity at the first)
  2. profile: correlation of every grayordinate of the subject's own data with
     every target, then z-scored per grayordinate over targets
  3. template subjects: fmralign-style iterative Procrustes averaging of the
     two profiles; the target: piecewise Procrustes onto the frozen template

Densification levels fit with lam = 0. The route's lam (pre-registration §8)
acts on the final level only, and is tuned per draw by the tuning step, so
this job stores what any lam needs: per subject and piece, the cross-product
``profile' template`` at the final level (``procrustes.transform_from_cross``).

Verbs:

  plan   report the inputs and the memory estimate; computes nothing
  fit    one target subject: writes <derivatives>/functional_space/routes/cha/target-<sub>/
         (--save-grams adds the final-level profile Grams the combined model stacks)

Usage:
    python cha.py plan
    python cha.py fit --target 03 [--save-templates] [--save-grams]
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

import grayordinates as go  # noqa: E402
import pieces as pc  # noqa: E402
import procrustes as pr  # noqa: E402
from core.config import load_config  # noqa: E402
from neuroimaging import data_quality as dq  # noqa: E402

REST_TASKS = ("INITresting", "TBresting", "NATresting", "FINresting")
SUBJECTS = ("03", "04", "05")
REGIME = "reference"
TEMPLATE_ITERATIONS = 3  # Procrustes-averaging passes per level


class Paths:
    def __init__(self) -> None:
        cfg = load_config()["paths"]
        self.derivatives = Path(cfg["output_dir"])
        self.cleaned = go.tree_root(self.derivatives)
        self.atlases = self.derivatives / "atlases"
        self.freesurfer = self.derivatives / "fmriprep" / "sourcedata" / "freesurfer"
        self.out = self.derivatives / "functional_space" / "routes" / "cha"


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

class Geometry:
    """Pieces and per-level target assignments over the grayordinate index."""

    def __init__(self, paths: Paths):
        self.table = pd.read_csv(go.grayordinates_path(paths.cleaned), sep="\t")
        parcels, names = pc.load_cortex_parcels(paths.atlases)
        hipp_x = pc.load_hipp_unfold_x(go.hipp_template_path(paths.cleaned))
        self.labels = pc.piece_labels(self.table, parcels, names, hipp_x)
        sphere = pc.load_sphere(paths.freesurfer)
        wall = {h: go.medial_wall(paths.atlases, h) for h in ("L", "R")}
        self.levels: dict[str, tuple[np.ndarray, list[str]]] = {}
        for level in pc.CHA_LEVELS:
            tiles = pc.cortex_tiles(sphere, wall, pc.ICO[level])
            self.levels[level] = pc.target_assignment(self.table, tiles)

    @property
    def n(self) -> int:
        return len(self.table)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def rest_runs(cleaned: Path, sub: str) -> pd.DataFrame:
    m = go.load_manifest(cleaned)
    runs = m[(m["sub"] == sub) & m["task"].isin(REST_TASKS) & (m["regime"] == REGIME)]
    if runs.empty:
        raise FileNotFoundError(f"no {REGIME} rest runs for sub-{sub} in {cleaned}/manifest.tsv")
    return runs.sort_values(["ses", "task", "run"])


def _zscore_columns(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x - mu) / sd


def load_rest(cleaned: Path, sub: str) -> tuple[np.ndarray, dict]:
    """Concatenated, per-run z-scored rest series ``(T, G)`` float32; a column invalid in any run is NaN."""
    runs = rest_runs(cleaned, sub)
    blocks, valid = [], None
    for r in runs.itertuples(index=False):
        x = go.load_run(cleaned / r.path, cleaned).data
        x = x[~np.isnan(x).all(axis=1)]  # non-steady-state rows
        ok = np.isfinite(x).all(axis=0) & (x.std(axis=0) > 0)
        valid = ok if valid is None else valid & ok
        blocks.append(_zscore_columns(x).astype(np.float32))
    data = np.concatenate(blocks, axis=0)
    data[:, ~valid] = np.nan
    info = {"n_runs": len(runs), "n_vol": int(data.shape[0]), "n_valid_columns": int(valid.sum()),
            "runs": [f"ses-{r.ses}_task-{r.task}" + (f"_run-{r.run}" if isinstance(r.run, str) and r.run else "")
                     for r in runs.itertuples(index=False)]}
    return data, info


# ---------------------------------------------------------------------------
# connectivity
# ---------------------------------------------------------------------------

def target_signals(data: np.ndarray, assign: np.ndarray, n_targets: int) -> np.ndarray:
    """Mean over each target's valid members, ``(T, K)``. A target with no valid member is an error."""
    from scipy import sparse

    valid = np.isfinite(data).all(axis=0) & (assign >= 0)
    cols = np.flatnonzero(valid)
    a = sparse.csr_matrix((np.ones(cols.size, dtype=np.float32), (np.arange(cols.size), assign[cols])),
                          shape=(cols.size, n_targets))
    counts = np.asarray(a.sum(axis=0)).ravel()
    if (counts == 0).any():
        raise ValueError(f"{int((counts == 0).sum())} targets have no valid member")
    return (np.asarray(a.T @ data[:, cols].T).T / counts).astype(np.float32)


def connectivity_profiles(data: np.ndarray, signals: np.ndarray) -> np.ndarray:
    """``(K, G)``: correlation of every column with every target, z-scored per column over targets."""
    t = data.shape[0]
    valid = np.isfinite(data).all(axis=0)
    d = _zscore_columns(np.where(valid, data, 0.0).astype(np.float32))
    s = _zscore_columns(signals.astype(np.float32))
    prof = (s.T @ np.nan_to_num(d)) / t
    prof = _zscore_columns(prof)
    prof[:, ~valid] = np.nan
    return prof.astype(np.float32)


def level_profile(data: np.ndarray, transform: pr.PiecewiseTransform | None, assign: np.ndarray,
                  n_targets: int) -> np.ndarray:
    """A subject's profile at one level: targets from its data in template space."""
    in_template = data if transform is None else transform.apply(data)
    return connectivity_profiles(data, target_signals(in_template, assign, n_targets))


def fit_diagnostic(profile: np.ndarray, template: np.ndarray, tf: pr.PiecewiseTransform) -> dict:
    """Fit on the alignment data itself (not a score): mean column correlation with the template."""
    def mean_r(a, b):
        ok = np.isfinite(a).all(0) & np.isfinite(b).all(0)
        a, b = _zscore_columns(a[:, ok]), _zscore_columns(b[:, ok])
        return float(np.nanmean((a * b).mean(axis=0)))
    return {"anatomical": round(mean_r(profile, template), 4), "aligned": round(mean_r(tf.apply(profile), template), 4)}


def fit_target(geo: Geometry, data: dict[str, np.ndarray], target: str, log=print
               ) -> tuple[dict[str, dict], dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Template from the non-target subjects, then the target into it. Returns
    (cross-products per subject at the final level, diagnostics, templates per level,
    final-level profiles per subject)."""
    template_subs = [s for s in data if s != target]
    tfs: dict[str, pr.PiecewiseTransform | None] = {s: None for s in data}
    templates, diag = {}, {}
    profiles: dict[str, np.ndarray] = {}
    for level in pc.CHA_LEVELS:
        t0 = time.time()
        assign, names = geo.levels[level]
        k = len(names)
        profiles = {s: level_profile(data[s], tfs[s], assign, k) for s in template_subs}
        template, fitted = pr.template_average([profiles[s] for s in template_subs], geo.labels, lam=0.0,
                                               n_iter=TEMPLATE_ITERATIONS)
        for s, tf in zip(template_subs, fitted):
            tfs[s] = tf
        templates[level] = template
        # the target enters the frozen template of this level
        profiles[target] = level_profile(data[target], tfs[target], assign, k)
        tfs[target] = pr.fit_piecewise(profiles[target], template, geo.labels, lam=0.0)
        diag[level] = {"n_targets": k,
                       **{f"sub-{s}": fit_diagnostic(profiles[s], template, tfs[s]) for s in data},
                       "elapsed_s": round(time.time() - t0, 1)}
        log(f"{level}: K={k} " + " ".join(f"sub-{s} {diag[level][f'sub-{s}']}" for s in data)
            + f" ({diag[level]['elapsed_s']:.0f} s)")
    final = templates[pc.CHA_LEVELS[-1]]
    cross = {s: pr.cross_products(profiles[s], final, geo.labels) for s in data}
    return cross, diag, templates, profiles


def profile_grams(profiles: dict[str, np.ndarray], labels: np.ndarray, target: str) -> tuple:
    """Per-piece Grams of the final-level profiles, for the combined model's CHA block.

    Same pairs, columns and solver as the stimulus and response routes
    (``stimulus_route.GramAccumulator``): template pairs over the columns valid
    in both template subjects, target pairs over the target's subset of them.
    Returns ``(pcols, template Grams, target Grams)``.
    """
    import stimulus_route as sr

    template_subs = sorted(s for s in profiles if s != target)
    valid = {s: np.isfinite(p).all(axis=0) for s, p in profiles.items()}
    pcols = sr.piece_columns(labels, valid, template_subs, target)
    tpl = sr.GramAccumulator(pcols, [(a, b) for i, a in enumerate(template_subs) for b in template_subs[i:]], target)
    tgt = sr.GramAccumulator(pcols, [(target, s) for s in template_subs], target)
    tpl.add({s: profiles[s] for s in template_subs})
    tgt.add(profiles)
    return pcols, tpl.g, tgt.g


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

def save_cross(cross: dict, path: Path) -> None:
    arrays = {}
    for i, (lab, (cols, m)) in enumerate(cross.items()):
        arrays[f"p{i}_cols"] = cols.astype(np.int32)
        arrays[f"p{i}_M"] = m.astype(np.float64)
    arrays["labels"] = np.array([str(lab) for lab in cross], dtype=object)
    np.savez(path, **arrays)


def load_cross(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {str(lab): (z[f"p{i}_cols"], z[f"p{i}_M"]) for i, lab in enumerate(z["labels"])}


def out_dir(derivatives: Path, target: str) -> Path:
    return Path(derivatives) / "functional_space" / "routes" / "cha" / f"target-{target}"


# ---------------------------------------------------------------------------
# verbs
# ---------------------------------------------------------------------------

def cmd_plan(args: argparse.Namespace) -> None:
    paths = Paths()
    geo = Geometry(paths)
    g = geo.n
    rows = []
    for s in SUBJECTS:
        runs = rest_runs(paths.cleaned, s)
        rows.append({"sub": s, "rest_runs": len(runs), "volumes": int((runs["n_vol"] - runs["n_nss"]).sum())})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    labs = pd.Series(geo.labels)
    print(f"\ngrayordinates {g}; pieces {labs[labs != ''].nunique()} "
          f"(largest {labs[labs != ''].value_counts().iloc[0]} columns)")
    for level in pc.CHA_LEVELS:
        k = len(geo.levels[level][1])
        print(f"{level}: {k} targets; one profile {k * g * 4 / 1e9:.1f} GB")
    t = int(df["volumes"].max())
    k = len(geo.levels[pc.CHA_LEVELS[-1]][1])
    peak = (3 * t * g + 6 * k * g) * 4 / 1e9
    print(f"\nestimated peak ~{peak:.0f} GB (3 subjects' rest series + 6 final-level profile-sized arrays)")


def cmd_fit(args: argparse.Namespace) -> None:
    paths = Paths()
    if args.target not in SUBJECTS:
        sys.exit(f"--target must be one of {SUBJECTS}")
    t0 = time.time()
    geo = Geometry(paths)
    data, info = {}, {}
    for s in SUBJECTS:
        data[s], info[s] = load_rest(paths.cleaned, s)
        print(f"sub-{s}: {info[s]['n_runs']} rest runs, {info[s]['n_vol']} volumes, "
              f"{info[s]['n_valid_columns']}/{geo.n} valid columns")
    cross, diag, templates, profiles = fit_target(geo, data, args.target)
    del data
    dest = out_dir(paths.derivatives, args.target)
    dest.mkdir(parents=True, exist_ok=True)
    for s, c in cross.items():
        save_cross(c, dest / f"cross_sub-{s}.npz")
    if args.save_grams:
        import stimulus_route as sr

        t1 = time.time()
        pcols, g_tpl, g_tgt = profile_grams(profiles, geo.labels, args.target)
        sr.save_grams(g_tpl, dest / "grams_template_profile.npz", pcols)
        sr.save_grams(g_tgt, dest / "grams_target_profile.npz", pcols)
        print(f"grams: {len(pcols.template)} pieces in {time.time() - t1:.0f} s")
    del profiles
    if args.save_templates:
        for level, tpl in templates.items():
            np.save(dest / f"template_{level}.npy", tpl)
    side = {
        "description": "CHA route, one target subject: per-piece cross-products (profile' template) at the "
                       "final level for every subject; procrustes.transform_from_cross(cross, n, lam) "
                       "gives the transform into the template for any lam.",
        "target": args.target, "template_subjects": [s for s in SUBJECTS if s != args.target],
        "levels": list(pc.CHA_LEVELS), "ico_centres_per_hemi": {k: pc.ICO[k] for k in pc.CHA_LEVELS},
        "targets": "cortex Voronoi tiles (mean) + 12 HOSPA structures + 2 hippocampi",
        "pieces": "Schaefer-400 17n (fsaverage6) + HOSPA structures + hippocampal thirds (unfold x)",
        "densification_lam": 0.0, "template_iterations": TEMPLATE_ITERATIONS,
        "grams_saved": bool(args.save_grams),
        "rest": {f"sub-{s}": info[s] for s in SUBJECTS},
        "fit_diagnostics": diag,
        "fit_diagnostics_note": "mean correlation of each subject's rest-connectivity profile with the template, "
                                "before (anatomical) and after alignment; alignment data only, not a score",
        "n_grayordinates": geo.n, "regime": REGIME,
        "series_root": str(paths.cleaned),
        "code_version": dq.code_version(REPO_ROOT),
        "elapsed_s": round(time.time() - t0, 1),
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (dest / "cha.json").write_text(json.dumps(side, indent=2) + "\n")
    print(f"wrote {dest} in {side['elapsed_s']:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    sub.add_parser("plan")
    f = sub.add_parser("fit")
    f.add_argument("--target", required=True, help="target subject label without 'sub-'")
    f.add_argument("--save-templates", action="store_true",
                   help="also save each level's template profile (large; for entering new subjects later)")
    f.add_argument("--save-grams", action="store_true",
                   help="also keep the final-level profile Grams (the combined model's CHA block)")
    args = ap.parse_args()
    {"plan": cmd_plan, "fit": cmd_fit}[args.verb](args)


if __name__ == "__main__":
    main()
