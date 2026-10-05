#!/usr/bin/env python3
"""Secondary ROI families, the smoothed-anatomical control, and the subcortical TB cache.

Pre-registration §10 (families) and §6 (controls), in mmmdata-agents
``docs/workbench/functional-space/``. ``score_route.py families`` uses this
module to score each partition job's models over every family, not only
family A.

Families (one label per grayordinate, '' = not in the family):

  schaefer7n   family A, the 7 Schaefer networks (as ``score_route.network_labels``)
  familyB      the pilot's Harvard-Oxford cortical ROIs (``localizer_ceiling.FAMILY_B``)
               plus the hippocampus (the hippunfold grayordinates, both hemispheres)
  familyD      HCP-MMP1's 22 sections, both hemispheres pooled
  subcortex    thalamus, caudate, putamen, pallidum, accumbens, amygdala; hemispheres pooled
  hippocampus  the hippunfold grayordinates per hemisphere

Smoothed-anatomical control (Bazeille 2021). The anatomical template mean is
smoothed on the fsaverage6 mesh until its spatial smoothness matches a route's
projection. Smoothing is k steps of normalized neighbour diffusion,
``x <- (x + mean of neighbours) / 2``. Missing columns are excluded from the
average and stay missing. Smoothness is the mean, over mesh edges, of the
temporal correlation between the two vertices' held-out film series.
Subcortex and hippocampus are not smoothed.

Subcortical TB cache. M3 for the subcortex reads the MNI res-2 GLMsingle TYPED
fit (``glmsingle_tb/sub-##/enc``, §11.5). That fit is a whole-volume array,
too large to load in every scoring job, so ``cache-tb`` writes each subject's
subcortical grayordinate rows once:
``<derivatives>/functional_space/glmsingle_tb_subcortex/sub-##/enc/``.

Usage:
    python families.py cache-tb --subject 03
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "src" / "python", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FAMILIES = ("schaefer7n", "familyB", "familyD", "subcortex", "hippocampus")
SUBCORTEX = ("THALAMUS", "CAUDATE", "PUTAMEN", "PALLIDUM", "ACCUMBENS", "AMYGDALA")
SECTIONS_STEM = "den-41k_atlas-HCPMMP1_seg-sections_dseg"
#: Smoothing steps tried when matching a route's smoothness.
SMOOTH_GRID = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------

def _cortex_rows(table: pd.DataFrame) -> np.ndarray:
    import grayordinates as go

    cortex = np.flatnonzero(table["piece"].to_numpy() == "cortex")
    if cortex.size != 2 * go.FSAVERAGE6_N or not np.array_equal(cortex, np.arange(cortex.size)):
        raise ValueError("grayordinate cortex rows are not the first 2 x fsaverage6 rows")
    return cortex


def family_labels(table: pd.DataFrame, atlases: Path) -> dict[str, np.ndarray]:
    """``{family: per-grayordinate label}`` for every family in ``FAMILIES``."""
    import localizer_ceiling as lc

    cortex = _cortex_rows(table)
    n = len(table)
    out = {f: np.full(n, "", dtype=object) for f in FAMILIES}
    for (fam, roi), mask in lc.load_rois(Path(atlases)).items():
        if fam in ("schaefer7n", "familyB"):
            out[fam][cortex[mask]] = roi
    anat = Path(atlases) / "tpl-fsaverage" / "anat"
    sec = np.concatenate([lc._label_gii(anat / f"tpl-fsaverage_hemi-{h}_{SECTIONS_STEM}.label.gii") for h in lc.HEMIS])
    names = pd.read_csv(anat / f"tpl-fsaverage_{SECTIONS_STEM}.tsv", sep="\t")
    name_of = dict(zip(names["index"].astype(int), names["name"]))
    if sec.size != cortex.size:
        raise ValueError(f"HCP-MMP1 sections cover {sec.size} vertices, cortex has {cortex.size}")
    out["familyD"][cortex] = [name_of.get(int(v), "") for v in sec]
    piece = table["piece"].to_numpy()
    structure = table["structure"].astype(str).to_numpy()
    sub = piece == "subcortex"
    base = np.array([s.rsplit("_", 1)[0] for s in structure], dtype=object)
    unknown = set(base[sub]) - set(SUBCORTEX)
    if unknown:
        raise ValueError(f"subcortical structures outside the family: {sorted(unknown)}")
    out["subcortex"][sub] = [s.lower() for s in base[sub]]
    hipp = piece == "hippocampus"
    out["hippocampus"][hipp] = ["hippocampus_" + h for h in table.loc[hipp, "hemi"]]
    out["familyB"][hipp] = "Hippocampus"
    return out


# ---------------------------------------------------------------------------
# mesh smoothing
# ---------------------------------------------------------------------------

def mesh_edges(freesurfer_dir: Path) -> np.ndarray:
    """Unique (u, v) edges of the fsaverage6 mesh over the L+R cortex rows (R offset by one hemisphere)."""
    import nibabel.freesurfer as fs
    import grayordinates as go

    edges = []
    for off, h in ((0, "lh"), (go.FSAVERAGE6_N, "rh")):
        coords, faces = fs.read_geometry(str(Path(freesurfer_dir) / "fsaverage6" / "surf" / f"{h}.white"))
        if coords.shape[0] != go.FSAVERAGE6_N:
            raise ValueError(f"{h}.white has {coords.shape[0]} vertices, not {go.FSAVERAGE6_N}")
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        e = np.unique(np.sort(e, axis=1), axis=0)
        edges.append(e + off)
    return np.concatenate(edges)


def diffusion_operator(edges: np.ndarray, n: int):
    """Row-normalized neighbour average over ``n`` columns (sparse; columns without edges are left out)."""
    from scipy import sparse

    u, v = edges[:, 0], edges[:, 1]
    a = sparse.coo_matrix((np.ones(2 * u.size), (np.r_[u, v], np.r_[v, u])), shape=(n, n)).tocsr()
    deg = np.asarray(a.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        inv = np.where(deg > 0, 1.0 / deg, 0.0)
    return sparse.diags(inv) @ a, deg > 0


class Smoother:
    """``k`` steps of normalized neighbour diffusion on the rows of ``x`` (rows = time, items or maps)."""

    def __init__(self, edges: np.ndarray, n: int):
        self.avg, self.on_mesh = diffusion_operator(edges, n)

    def step(self, x: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One step on the weighted pair (x·m, m); the caller divides at the end."""
        xs = x.copy()
        ms = m.copy()
        xs[:, self.on_mesh] = 0.5 * (x[:, self.on_mesh] + (self.avg @ x.T).T[:, self.on_mesh])
        ms[:, self.on_mesh] = 0.5 * (m[:, self.on_mesh] + (self.avg @ m.T).T[:, self.on_mesh])
        return xs, ms

    def run(self, x: np.ndarray, ks) -> dict[int, np.ndarray]:
        """``{k: x smoothed k steps}`` for every k in ``ks``; missing entries of ``x`` stay missing."""
        ks = sorted(set(int(k) for k in ks))
        missing = ~np.isfinite(x)
        m = (~missing).astype(np.float64)
        xw = np.where(missing, 0.0, x).astype(np.float64)
        out, done = {}, 0
        for k in ks:
            while done < k:
                xw, m = self.step(xw, m)
                done += 1
            with np.errstate(invalid="ignore", divide="ignore"):
                y = np.where(m > 0, xw / m, np.nan)
            y[missing] = np.nan
            out[k] = y.astype(np.float32)
        return out


def neighbour_r(films: dict[str, np.ndarray], edges: np.ndarray) -> float:
    """Mean over films of the mean, over mesh edges with both ends finite, of the edge's temporal correlation."""
    import scoring as sc

    vals = []
    for x in films.values():
        z = sc.zscore_columns(np.asarray(x, dtype=np.float64))
        a, b = z[:, edges[:, 0]], z[:, edges[:, 1]]
        ok = np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)
        if ok.any():
            vals.append(float(np.mean((a[:, ok] * b[:, ok]).mean(axis=0))))
    return float(np.mean(vals)) if vals else np.nan


def smoothness_grid(smoother: Smoother, films: dict[str, np.ndarray], edges: np.ndarray, target_r: float
                    ) -> dict[int, float]:
    """Anatomical neighbour r at each SMOOTH_GRID step, stopping at the first step that reaches ``target_r``.

    Every film is carried forward step by step, so each grid point costs only
    the steps since the last one.
    """
    state = {}
    for f, x in films.items():
        missing = ~np.isfinite(x)
        state[f] = [np.where(missing, 0.0, x).astype(np.float64), (~missing).astype(np.float64), missing]
    out, done = {}, 0
    for k in SMOOTH_GRID:
        while done < k:
            for st in state.values():
                st[0], st[1] = smoother.step(st[0], st[1])
            done += 1
        ys = {}
        for f, (xw, m, missing) in state.items():
            with np.errstate(invalid="ignore", divide="ignore"):
                y = np.where(m > 0, xw / m, np.nan)
            y[missing] = np.nan
            ys[f] = y
        out[k] = neighbour_r(ys, edges)
        if out[k] >= target_r:
            break
    return out


def match_steps(route_r: float, grid_r: dict[int, float]) -> int:
    """The smoothing step count whose anatomical smoothness is closest to the route's (ties to fewer steps)."""
    ks = sorted(k for k, v in grid_r.items() if np.isfinite(v))
    if not ks or not np.isfinite(route_r):
        raise ValueError("no finite smoothness to match")
    return min(ks, key=lambda k: (abs(grid_r[k] - route_r), k))


# ---------------------------------------------------------------------------
# subcortical TB cache
# ---------------------------------------------------------------------------

def tb_cache_dir(derivatives: Path, subject: str) -> Path:
    return Path(derivatives) / "functional_space" / "glmsingle_tb_subcortex" / f"sub-{subject}" / "enc"


def load_tb_subcortex(derivatives: Path, subject: str, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows, betas rows x trials, meanvol per row) for the subcortical grayordinates; a missing cache is an error."""
    d = tb_cache_dir(derivatives, subject)
    path = d / "subcortex_betas.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `families.py cache-tb --subject {subject}`")
    z = np.load(path)
    rows = z["rows"]
    want = np.flatnonzero(table["piece"].to_numpy() == "subcortex")
    if not np.array_equal(rows, want):
        raise ValueError(f"{path}: cached rows differ from the grayordinate table's subcortex")
    return rows, z["betas"], z["meanvol"]


def cmd_cache_tb(args: argparse.Namespace) -> None:
    import nibabel as nib
    import cha
    import encoding as enc
    import grayordinates as go

    paths = enc.Paths()
    s = args.subject
    src = Path(paths.derivatives) / "glmsingle_tb" / f"sub-{s}" / "enc"
    surf = Path(paths.derivatives) / "functional_space" / "glmsingle_tb_fsaverage6" / f"sub-{s}" / "enc"
    t_vol = pd.read_csv(src / "trial_info.csv", dtype={"mmmId": str})
    t_surf = pd.read_csv(surf / "trial_info.csv", dtype={"mmmId": str})
    if not t_vol.equals(t_surf):
        raise ValueError(f"sub-{s}: the volume fit's trial table differs from the fsaverage6 fit's")
    table = pd.read_csv(go.grayordinates_path(cha.Paths().cleaned), sep="\t")
    rows = np.flatnonzero(table["piece"].to_numpy() == "subcortex")
    ijk = table.loc[rows, ["i", "j", "k"]].to_numpy().astype(int)
    _, affine, shape = go.subcortical_masks(cha.Paths().atlases)
    ref = nib.load(str(next(src.glob(f"sub-{s}_task-TBencoding_space-MNI152NLin2009cAsym_res-2_*.nii.gz"))))
    if tuple(ref.shape[:3]) != tuple(shape) or not np.allclose(ref.affine, affine, atol=1e-4):
        raise ValueError(f"sub-{s}: the volume fit's grid {ref.shape[:3]} differs from the HOSPA res-2 grid {shape}")
    fit = np.load(src / "glmsingle_outputs" / "TYPED_FITHRF_GLMDENOISE_RR.npy", allow_pickle=True).item()
    b = fit["betasmd"]
    if b.shape[:3] != tuple(shape) or b.shape[3] != len(t_vol):
        raise ValueError(f"sub-{s}: betasmd {b.shape} vs grid {shape} and {len(t_vol)} trials")
    betas = np.asarray(b[ijk[:, 0], ijk[:, 1], ijk[:, 2], :], dtype=np.float32)
    meanvol = np.asarray(fit["meanvol"], dtype=np.float32).reshape(shape)[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    del fit, b
    dest = tb_cache_dir(paths.derivatives, s)
    dest.mkdir(parents=True, exist_ok=True)
    np.savez(dest / "subcortex_betas.npz", rows=rows, betas=betas, meanvol=meanvol)
    t_vol.to_csv(dest / "trial_info.csv", index=False)
    import scoring as sc
    (dest / "subcortex_betas.json").write_text(json.dumps({
        "description": "Subcortical grayordinate rows of the MNI res-2 GLMsingle TYPED fit (betasmd, meanvol), "
                       "one row per subcortex grayordinate in table order; trial columns as trial_info.csv",
        "source": str(src / "glmsingle_outputs" / "TYPED_FITHRF_GLMDENOISE_RR.npy"),
        "n_rows": int(rows.size), "n_trials": int(betas.shape[1]), "code_version": sc._code_version(),
    }, indent=2) + "\n")
    print(f"sub-{s}: {rows.size} subcortical rows x {betas.shape[1]} trials -> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    c = sub.add_parser("cache-tb")
    c.add_argument("--subject", required=True)
    args = ap.parse_args()
    {"cache-tb": cmd_cache_tb}[args.verb](args)


if __name__ == "__main__":
    main()
