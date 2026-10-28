"""NSD (the Natural Scenes Dataset, Allen et al. 2022) as a comparison dataset for MMMData.

MMMData's image set is NSD's shared1000, so NSD's shared trials show the same images that
MMMData's trial-based sessions do. This module is the shared layer for comparisons:

* where NSD is (``paths.nsd_dir`` in the config; NSD stays outside this dataset);
* NSD's shared trials mapped to MMMData images through ``stimuli/shared1000/shared1000.csv``
  (``nsdId`` -> ``mmmId`` and the image filename, which is the ``stimulus_id``);
* single-trial betas (``betas_fithrf_GLMdenoise_RR``) in ``func1pt8mm`` volumes or on fsaverage,
  cut to an icosahedral order so they line up with MMMData's fsaverage6 grayordinates;
* ROIs: NSD's manual MTL labels, streams, nsdgeneral, aseg, an anterior/posterior hippocampal
  split in MNI y, and depth subsets for partial-volume-safe controls;
* pattern-level reliability across repeats that never compares two trials from one session.

Conventions:

* **Units.** A volume unit is a voxel of the subject's ``func1pt8mm/brainmask.nii.gz``, in C
  order. A surface unit is a vertex: lh then rh, each cut to the first ``ICO_VERTICES[ico]``
  vertices of fsaverage. FreeSurfer's icosahedra nest, so fsaverage6's vertices are exactly the
  first 40,962 of fsaverage's; the cut resamples nothing.
* **Betas.** Volume files are int16 in units of 1/300 % signal; surface files are float32 %
  signal. Both are z-scored per unit within each session over all of that session's trials,
  NSD's recommendation (:func:`neuroimaging.data_quality_glmsingle.zscore_by_session`). A unit
  with no variance in a session is NaN there and stays NaN.
* **Betas may live apart from the release tree.** They are large (tens of GB per subject), so
  every loader takes ``NSDPaths.betas``, the directory that holds ``nsddata_betas/``; by default
  it is ``nsd_dir`` itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from neuroimaging import data_quality_glmsingle as dq

N_SHARED = 1000
TRIALS_PER_SESSION = 750
MAX_SESSIONS = 40
SPACES = ("func1pt8mm", "fsaverage")
BETAS = "betas_fithrf_GLMdenoise_RR"
VOLUME_SCALE = 300.0  # int16 volume betas are % signal x 300
#: Vertices per hemisphere of FreeSurfer's icosahedral surfaces (fsaverage5/6/7).
ICO_VERTICES = {5: 10242, 6: 40962, 7: 163842}

# Manual MTL labels (``templates/MTL.ctab``). Hippocampus = SUB, CA1, CA2, CA3, DG, HT.
MTL_HIPPOCAMPUS = (4, 5, 6, 7, 8, 10)
MTL_OTHER = {"erc": (1,), "prc": (2, 3), "phc": (9,)}
STREAMS = {1: "early", 2: "midventral", 3: "midlateral", 4: "midparietal", 5: "ventral",
           6: "lateral", 7: "parietal"}
ASEG = {"white_matter": (2, 41), "cortex": (3, 42), "hippocampus": (17, 53), "amygdala": (18, 54)}
#: MNI y (mm) of the uncal apex, splitting anterior from posterior hippocampus (Poppenk et al. 2013).
UNCAL_APEX_Y = -21.0


@dataclass(frozen=True)
class NSDPaths:
    """``root`` holds ``nsddata/``; ``betas`` holds ``nsddata_betas/`` (defaults to ``root``)."""

    root: Path
    betas: Path

    @classmethod
    def from_config(cls, betas: str | Path | None = None) -> "NSDPaths":
        from core.config import load_config

        paths = load_config()["paths"]
        if "nsd_dir" not in paths:
            raise KeyError("no paths.nsd_dir in the mmmdata config; set it in config/local.toml")
        root = Path(paths["nsd_dir"])
        return cls(root, Path(betas) if betas else root)

    @property
    def nsddata(self) -> Path:
        return self.root / "nsddata"

    def ppdata(self, subj: int) -> Path:
        return self.nsddata / "ppdata" / f"subj{subj:02d}"

    def func(self, subj: int) -> Path:
        return self.ppdata(subj) / "func1pt8mm"

    def beta_file(self, subj: int, space: str, session: int, hemi: str | None = None) -> Path:
        d = self.betas / "nsddata_betas" / "ppdata" / f"subj{subj:02d}" / space / BETAS
        if space == "func1pt8mm":
            return d / f"betas_session{session:02d}.nii.gz"
        if space == "fsaverage":
            if hemi not in ("lh", "rh"):
                raise ValueError("fsaverage betas need hemi 'lh' or 'rh'")
            return d / f"{hemi}.betas_session{session:02d}.mgh"
        raise ValueError(f"space must be one of {SPACES}, not {space!r}")


def check_space(space: str) -> None:
    if space not in SPACES:
        raise ValueError(f"space must be one of {SPACES}, not {space!r}")


# ---------------------------------------------------------------------------
# trials -> images
# ---------------------------------------------------------------------------

def load_expdesign(paths: NSDPaths) -> tuple[np.ndarray, np.ndarray]:
    """``(masterordering, subjectim)`` from ``nsd_expdesign.mat``; both hold 1-based indices."""
    import scipy.io as sio

    e = sio.loadmat(paths.nsddata / "experiments" / "nsd" / "nsd_expdesign.mat")
    return e["masterordering"].ravel().astype(int), e["subjectim"].astype(int)


def shared_trials(masterordering: np.ndarray, subjectim_row: np.ndarray, shared1000: pd.DataFrame,
                  n_sessions: int) -> pd.DataFrame:
    """One row per shared trial in the first ``n_sessions`` sessions, in acquisition order.

    ``masterordering`` gives each trial an index into the subject's ``subjectim`` row, whose
    first ``N_SHARED`` entries are the shared images; ``shared1000`` is MMMData's
    ``shared1000.csv`` (``fileNames``, ``mmmId``, ``nsdId``). Columns: ``session``, ``trial``
    (1-based within session), ``shared_idx``, ``nsd_id``, ``mmm_id``, ``stimulus_id`` (the
    filename stem) and ``rep`` (1 for an image's first shared trial, then 2, 3, ...).
    """
    if not 1 <= n_sessions <= MAX_SESSIONS:
        raise ValueError(f"n_sessions must be 1..{MAX_SESSIONS}, got {n_sessions}")
    order = np.asarray(masterordering)[: n_sessions * TRIALS_PER_SESSION]
    keep = np.flatnonzero(order <= N_SHARED)
    shared_idx = order[keep]
    nsd_id = np.asarray(subjectim_row)[shared_idx - 1]
    lookup = shared1000.set_index("nsdId")
    missing = sorted(set(nsd_id.tolist()) - set(lookup.index))
    if missing:
        raise ValueError(f"{len(missing)} shared NSD images are not in shared1000.csv, e.g. nsdId {missing[0]}")
    t = pd.DataFrame({"session": keep // TRIALS_PER_SESSION + 1, "trial": keep % TRIALS_PER_SESSION + 1,
                      "shared_idx": shared_idx, "nsd_id": nsd_id})
    t["mmm_id"] = lookup.loc[nsd_id, "mmmId"].to_numpy()
    t["stimulus_id"] = [Path(f).stem for f in lookup.loc[nsd_id, "fileNames"]]
    t["rep"] = t.groupby("stimulus_id").cumcount() + 1
    return t


def sessions_present(paths: NSDPaths, subj: int, space: str) -> int:
    """How many sessions, counting from 1, have their beta file(s) on disk."""
    check_space(space)
    n = 0
    for s in range(1, MAX_SESSIONS + 1):
        files = ([paths.beta_file(subj, space, s)] if space == "func1pt8mm"
                 else [paths.beta_file(subj, space, s, h) for h in ("lh", "rh")])
        if not all(f.exists() for f in files):
            break
        n = s
    return n


# ---------------------------------------------------------------------------
# betas
# ---------------------------------------------------------------------------

def brainmask(paths: NSDPaths, subj: int):
    """``(mask, image)``: the subject's ``func1pt8mm`` brainmask as a 3-D bool array."""
    import nibabel as nib

    img = nib.load(paths.func(subj) / "brainmask.nii.gz")
    return np.asanyarray(img.dataobj) > 0, img


def cut_hemispheres(lh: np.ndarray, rh: np.ndarray, ico: int) -> np.ndarray:
    """lh then rh, each cut to ``ico``'s vertex count along axis 0."""
    n = ICO_VERTICES[ico]
    if lh.shape[0] < n or rh.shape[0] < n:
        raise ValueError(f"hemispheres have {lh.shape[0]}/{rh.shape[0]} vertices; ico{ico} needs {n}")
    return np.concatenate([lh[:n], rh[:n]], axis=0)


def session_betas(paths: NSDPaths, subj: int, space: str, session: int, *, mask: np.ndarray | None = None,
                  ico: int = 6) -> np.ndarray:
    """``(units, trials)`` % signal change for one session, float32."""
    import nibabel as nib

    check_space(space)
    if space == "func1pt8mm":
        if mask is None:
            mask, _ = brainmask(paths, subj)
        b = np.asanyarray(nib.load(paths.beta_file(subj, space, session)).dataobj)[mask]
        return b.astype(np.float32) / VOLUME_SCALE
    hemis = [np.asanyarray(nib.load(paths.beta_file(subj, space, session, h)).dataobj).reshape(-1, TRIALS_PER_SESSION)
             for h in ("lh", "rh")]
    return cut_hemispheres(*hemis, ico).astype(np.float32)


def extract_shared(paths: NSDPaths, subj: int, space: str, out: Path, shared1000: pd.DataFrame,
                   *, ico: int = 6, n_sessions: int | None = None, log=print) -> pd.DataFrame:
    """Session-z-scored shared-trial betas, image means and the noise ceiling, written to ``out``.

    Writes ``trials.tsv``; ``shared_betas_z.npy`` ``(trials, units)``; ``images.tsv`` and
    ``image_mean_z.npy`` ``(N_SHARED, units)`` in ``shared_idx`` order (an image with no trial
    is NaN); ``ncsnr.npy`` and ``nc.npy`` (NSD noise ceiling, %, for the mean of the modal
    repeat count, over the images that have it). Returns the trials table.
    """
    from numpy.lib.format import open_memmap

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n_sessions = n_sessions or sessions_present(paths, subj, space)
    if n_sessions < 1:
        raise FileNotFoundError(f"no {space} betas for subj{subj:02d} under {paths.betas}")
    mo, si = load_expdesign(paths)
    trials = shared_trials(mo, si[subj - 1], shared1000, n_sessions)
    mask = brainmask(paths, subj)[0] if space == "func1pt8mm" else None
    z = None
    for s in range(1, n_sessions + 1):
        b = session_betas(paths, subj, space, s, mask=mask, ico=ico)
        zs = dq.zscore_by_session(b, np.zeros(b.shape[1], int))
        rows = trials.index[trials.session == s].to_numpy()
        if z is None:
            z = open_memmap(out / "shared_betas_z.npy", mode="w+", dtype=np.float32, shape=(len(trials), b.shape[0]))
        z[rows] = zs[:, trials.loc[rows, "trial"].to_numpy() - 1].T
        log(f"subj{subj:02d} {space} session {s:02d}: {len(rows)} shared trials")
    z.flush()
    trials.to_csv(out / "trials.tsv", sep="\t", index=False)

    images = pd.DataFrame({"shared_idx": np.arange(1, N_SHARED + 1)})
    first = trials.drop_duplicates("shared_idx").set_index("shared_idx")
    images["stimulus_id"] = images.shared_idx.map(first.stimulus_id)
    images["n_reps"] = images.shared_idx.map(trials.shared_idx.value_counts()).fillna(0).astype(int)
    images.to_csv(out / "images.tsv", sep="\t", index=False)
    means = open_memmap(out / "image_mean_z.npy", mode="w+", dtype=np.float32, shape=(N_SHARED, z.shape[1]))
    by_image = trials.groupby("shared_idx").indices
    for i in range(N_SHARED):
        r = by_image.get(i + 1)
        means[i] = np.asarray(z[r]).mean(axis=0) if r is not None else np.nan
    means.flush()

    n = int(images.n_reps[images.n_reps > 0].mode().iloc[0])
    full = images.shared_idx[images.n_reps == n].to_numpy()
    groups = np.stack([by_image[i][:n] for i in full])  # (images, n) rows of z
    snr = np.empty(z.shape[1], np.float32)
    for c in range(0, z.shape[1], 20_000):
        snr[c:c + 20_000] = dq.ncsnr(np.asarray(z[:, c:c + 20_000]).T, groups)[0]
    np.save(out / "ncsnr.npy", snr)
    np.save(out / "nc.npy", dq.noise_ceiling(snr, n))
    log(f"subj{subj:02d} {space}: {len(trials)} trials over {n_sessions} sessions, {z.shape[1]} units; "
        f"noise ceiling for the mean of {n} over {len(full)} images")
    return trials


# ---------------------------------------------------------------------------
# ROIs
# ---------------------------------------------------------------------------

def depth_subsets(white_matter: np.ndarray, hippocampus: np.ndarray, neighbours: np.ndarray) -> dict[str, np.ndarray]:
    """Depth-defined subsets on a 3-D grid, for controls that partial volume cannot reach.

    ``white_matter_edge``: WM voxels touching non-WM (6-connected). ``white_matter_deep``: WM
    after a 2-step 6-connected erosion, so >= 3 voxels from non-WM. For the hippocampus, by
    26-connected steps from ``neighbours`` (grey matter outside it that can leak into it):
    ``hippocampus_gray_adjacent`` touches it, ``hippocampus_gray_gap1`` does not,
    ``hippocampus_gray_gap2`` is >= 2 steps clear; ``hippocampus_eroded`` is a 6-connected
    erosion of the hippocampus itself.
    """
    from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

    s26 = generate_binary_structure(3, 3)
    gray = neighbours & ~hippocampus
    near1 = binary_dilation(gray, s26)
    near2 = binary_dilation(gray, s26, iterations=2)
    return {
        "white_matter_edge": white_matter & ~binary_erosion(white_matter),
        "white_matter_deep": binary_erosion(white_matter, iterations=2),
        "hippocampus_gray_adjacent": hippocampus & near1,
        "hippocampus_gray_gap1": hippocampus & ~near1,
        "hippocampus_gray_gap2": hippocampus & ~near2,
        "hippocampus_eroded": binary_erosion(hippocampus),
    }


def volume_rois(paths: NSDPaths, subj: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """``(rois, mni_y)`` over the subject's brainmask voxels (bool vectors; MNI y in mm)."""
    import nibabel as nib

    mask, _ = brainmask(paths, subj)
    f = paths.func(subj)
    vol = lambda p: np.asanyarray(nib.load(p).dataobj)  # noqa: E731
    mtl3, aseg3 = vol(f / "roi" / "MTL.nii.gz"), vol(f / "aseg.nii.gz")
    lh = vol(f / "roi" / "lh.MTL.nii.gz")[mask] > 0
    rh = vol(f / "roi" / "rh.MTL.nii.gz")[mask] > 0
    # MNI-to-func1pt8 holds, per func voxel, its 1-based MNI152 1 mm voxel coordinates.
    t = vol(paths.ppdata(subj) / "transforms" / "MNI-to-func1pt8.nii.gz")[mask]
    mni_aff = nib.load(paths.nsddata / "templates" / "MNI152_T1_1mm.nii.gz").affine
    mni_y = (np.column_stack([t - 1, np.ones(len(t))]) @ mni_aff.T)[:, 1].astype(np.float32)

    mtl, aseg = mtl3[mask], aseg3[mask]
    hippo = np.isin(mtl, MTL_HIPPOCAMPUS)
    r = {"hippocampus": hippo,
         "hippocampus_anterior": hippo & (mni_y > UNCAL_APEX_Y),
         "hippocampus_posterior": hippo & (mni_y <= UNCAL_APEX_Y),
         "lh.hippocampus": hippo & lh, "rh.hippocampus": hippo & rh}
    for k, v in MTL_OTHER.items():
        r[k] = np.isin(mtl, v)
    r["hippocampus_aseg"] = np.isin(aseg, ASEG["hippocampus"])
    r["amygdala_aseg"] = np.isin(aseg, ASEG["amygdala"])
    r["white_matter"] = np.isin(aseg, ASEG["white_matter"])
    r["nsdgeneral"] = vol(f / "roi" / "nsdgeneral.nii.gz")[mask] == 1
    streams = vol(f / "roi" / "streams.nii.gz")[mask]
    for k, v in STREAMS.items():
        r[f"streams_{v}"] = streams == k

    hippo3 = np.isin(mtl3, MTL_HIPPOCAMPUS) & mask
    neighbours3 = np.isin(aseg3, ASEG["cortex"]) | np.isin(aseg3, ASEG["amygdala"])
    neighbours3 |= np.isin(mtl3, [c for v in MTL_OTHER.values() for c in v])
    for k, v in depth_subsets(np.isin(aseg3, ASEG["white_matter"]), hippo3, neighbours3).items():
        r[k] = v[mask]
    r["brain"] = np.ones(int(mask.sum()), bool)
    return r, mni_y


def surface_rois(paths: NSDPaths, ico: int = 6) -> dict[str, np.ndarray]:
    """nsdgeneral and the streams on fsaverage, cut to ``ico`` (lh then rh)."""
    import nibabel as nib

    lab = paths.nsddata / "freesurfer" / "fsaverage" / "label"
    load = lambda name: cut_hemispheres(*[np.asanyarray(nib.load(lab / f"{h}.{name}.mgz").dataobj).ravel()  # noqa: E731
                                          for h in ("lh", "rh")], ico)
    r = {"nsdgeneral": load("nsdgeneral") == 1}
    streams = load("streams")
    for k, v in STREAMS.items():
        r[f"streams_{v}"] = streams == k
    r["cortex_all"] = np.ones(2 * ICO_VERTICES[ico], bool)
    return r


def sample_units(roi: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """A bool vector with ``min(n, roi.sum())`` of ``roi``'s units drawn without replacement."""
    idx = np.flatnonzero(roi)
    out = np.zeros(roi.size, bool)
    out[rng.choice(idx, min(n, idx.size), replace=False)] = True
    return out


# ---------------------------------------------------------------------------
# pattern reliability across repeats
# ---------------------------------------------------------------------------

def corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of every row of ``a`` with every row of ``b`` (patterns over units)."""
    a = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
    b = (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)
    return a @ b.T / a.shape[1]


def pattern_reliability(x: np.ndarray, trials: pd.DataFrame, n_perm: int, rng: np.random.Generator,
                        n_reps: int = 3) -> dict[str, float]:
    """Repeat reliability of one ROI's patterns, never comparing two trials from one session.

    ``x`` is ``(trials, units)`` aligned with ``trials`` (``stimulus_id``, ``rep``,
    ``session``); units with any NaN are dropped. Images with ``n_reps`` repeats enter; for
    each pair of repeats ``(a, b)`` only images whose two trials fall in different sessions
    count. Trials near in time share pattern structure whatever the image, so baselines and
    nulls are session-matched:

    * ``roi_mean_r``: r across images of the ROI-mean response, rep a vs rep b (and its
      Spearman-Brown step-up to the mean of ``n_reps``, ``roi_mean_r_sb``);
    * ``pattern_discrim``: same-image corr(rep a, rep b) minus the mean corr of the rep-a
      pattern with other images' rep-b patterns from the same session as the same-image rep b;
      ``pattern_discrim_p`` permutes images within those sessions;
    * ``ident_top1``: share of images whose own rep b wins among those candidates, beside
      ``ident_chance``;
    * ``rdm_rho``: Spearman of the rep-a and rep-b correlation-distance RDMs over image pairs
      that differ in session in both repeats.
    """
    from scipy.stats import spearmanr

    x = x[:, ~np.isnan(x).any(axis=0)]
    full = trials.groupby("stimulus_id").rep.max()
    sids = sorted(full.index[full >= n_reps])
    t = trials.reset_index(drop=True).set_index(["stimulus_id", "rep"])
    pos = {k: i for i, k in enumerate(t.index)}
    reps = [np.array([pos[(s, r)] for s in sids]) for r in range(1, n_reps + 1)]
    sess = [t.loc[[(s, r) for s in sids], "session"].to_numpy() for r in range(1, n_reps + 1)]
    pats = [x[r] for r in reps]
    r_mean, d_all, null_all, hit, chance, rdm = [], [], [], [], [], []
    for a, b in combinations(range(n_reps), 2):
        sa, sb = sess[a], sess[b]
        cross = np.flatnonzero(sa != sb)
        r_mean.append(np.corrcoef(pats[a].mean(1)[cross], pats[b].mean(1)[cross])[0, 1])
        c = corr_rows(pats[a], pats[b])
        groups = {s: np.flatnonzero(sb == s) for s in np.unique(sb)}
        base = np.empty(len(cross))
        for k, i in enumerate(cross):
            g = groups[sb[i]]
            base[k] = (c[i, g].sum() - c[i, i]) / (len(g) - 1) if len(g) > 1 else np.nan
            hit.append(c[i, g].argmax() == np.flatnonzero(g == i)[0])
            chance.append(1 / len(g))
        ok = ~np.isnan(base)
        d_all.append((c[cross, cross] - base)[ok])
        perm = np.empty((n_perm, int(ok.sum())))
        for q in range(n_perm):
            pi = np.arange(len(sb))
            for g in groups.values():
                pi[g] = rng.permutation(g)
            perm[q] = (c[cross, pi[cross]] - base)[ok]
        null_all.append(perm)
        da, db = 1 - corr_rows(pats[a], pats[a]), 1 - corr_rows(pats[b], pats[b])
        iu = np.triu_indices(len(sids), 1)
        keep = (sa[iu[0]] != sa[iu[1]]) & (sb[iu[0]] != sb[iu[1]])
        rdm.append(spearmanr(da[iu][keep], db[iu][keep])[0])
    d = float(np.mean(np.concatenate(d_all)))
    null = np.concatenate(null_all, axis=1).mean(1)
    rm = float(np.mean(r_mean))
    return {"n_units": int(x.shape[1]), "n_images": len(sids), "roi_mean_r": rm,
            "roi_mean_r_sb": n_reps * rm / (1 + (n_reps - 1) * rm),
            "pattern_discrim": d, "pattern_discrim_p": float((np.sum(null >= d) + 1) / (n_perm + 1)),
            "ident_top1": float(np.mean(hit)), "ident_chance": float(np.mean(chance)),
            "rdm_rho": float(np.mean(rdm))}
