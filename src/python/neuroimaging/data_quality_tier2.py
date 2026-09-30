"""Tier 2 of the data-quality collection: naturalistic measures pooled from tier-1 caches.

Tier 1 (``data_quality.py``) cleans every run under every confound regime and
caches parcel time series. Tier 2 reads only those caches, the BIDS events,
the stimulus registry and the Contract B feature store -- never voxels -- so
the whole of it rebuilds from tier 1 in minutes (Settles-when 4 of the design
record, mmmdata-agents ``docs/archive/workbench/data-quality/``).

This module holds the tier-2 measures on the film task:

* **LOO-ISFC** (registry T2.2): for each film and regime, each subject's parcel
  series against the mean of every other subject's, as a parcel x parcel
  matrix. It is deliberately asymmetric: ``[p, q]`` is the subject's own parcel
  ``p`` against the others' parcel ``q`` (Simony et al. 2016). The diagonal is
  the leave-one-out ISC.
* **Audio-envelope lag scan** (T2.4): Pearson r between a parcel series and the
  film's loudness envelope at lags ``-MAX_LAG .. +MAX_LAG`` TRs. A **negative
  lag means the audio leads the BOLD** (BOLD volume ``i + |lag|`` is paired with
  envelope volume ``i``), the physiologically expected direction.
* **Repeat reliability** (T2.1, WSC): for every film a subject saw in more than
  one session (the two that recur in every film session), the Pearson r of each
  parcel between every pair of that subject's showings, all windows cut to the
  subject's shortest showing of the film; pairs averaged in Fisher-z. Port of
  ``scripts/isc_confounds/isc_confound_comparison.py``'s WSC.
* **Discriminability** (T2.3): per session and parcel, the mean cross-subject r
  for the same film (within) minus that for different films (between); each pair
  is cut to its shorter window from the start. Plain means of r, as
  ``scripts/nordic_benchmark/nat_eval.py`` defines it (its NORDIC arm dropped).

Definitions:

* **Which showing.** LOO-ISFC and the lag scan read each film from each
  subject's **first** showing only, so every number is an exposure-matched first
  viewing. Repeat reliability reads every showing of a recurring film.
  Discriminability reads every showing of its session: within a session all
  subjects are at the same exposure count for every film.
* **Window.** A showing's window is the volumes that lie wholly inside the
  film, ``ceil(onset / TR) .. floor((onset + duration) / TR)``, with onset and
  duration from the events file. For ISFC every subject is cut to the shortest
  window of the film. Volume ``i`` is taken to span ``[i*TR, (i+1)*TR)``.
* **Envelope.** The feature store's ``loudness_rms`` (``linear``) and
  ``loudness_db`` (``log``) on the 0.5 s grid, averaged over the frames whose
  centre falls in each window volume's span, measured in film time. So the
  envelope is binned onto the run's own volumes, and nothing is shifted to match
  the BOLD.
* **Parcel names.** Tier-1 Schaefer columns carry the names of the MNI
  ``dseg.tsv``. Until 2026-09-28 that table was TemplateFlow's, which predates
  CBIG's label renaming (252 of 400 names agreed; what CBIG calls
  ``SomMotB_Ins``/``S2``/``Cent`` was ``SomMotB_Aud`` there). Since the restage
  (``scripts/stage_mni_atlases.py``) it is CBIG's own, so the rename by index to
  the fsaverage table is an identity -- kept as a cross-check that the volume and
  surface tables still agree index for index, and so caches built on the old
  table fail loudly instead of being read under stale names. See
  :func:`schaefer_current_names`.
* **Missing values.** A parcel with any non-finite value in the window is n/a
  for that window. Nothing is imputed. The auditory ROI is the mean of the
  ``_Aud_`` parcels that are finite in the window, and the count used is
  recorded.

Outputs, under ``<tree>/tier2/naturalistic/``::

    film_viewings.tsv                    every showing found, first-viewing flag, window
    loo_isc.tsv                          subject x film x regime x parcel: the ISFC diagonal
    loo_isfc_summary.tsv                 subject x film x regime: ISC median, off-diagonal mean/sd
    isfc/stim-<id>_seg-<atlas>_desc-<regime>_isfc.npy   group-mean ISFC (float32, P x P)
    envelope_lag.tsv                     auditory-ROI lag curve per showing x regime x envelope
    envelope_parcel.tsv                  per parcel: best r inside PLAUSIBLE_LAGS, and its lag
    wsc.tsv                              subject x recurring film x regime x parcel: repeat reliability
    wsc_pairs.tsv                        subject x recurring film x regime x session pair: median r over parcels
    discriminability.tsv                 session x regime x parcel: within, between, their difference
    provenance.json                      inputs, parameters, code version

The ``.npy`` holds the mean over subjects of the per-subject matrices, i.e.
what the colleague's "group" panel shows. The per-subject matrices are not
kept (660 of them would be ~2 GB); they rebuild in seconds.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

TIER2_DIR = "tier2"
SCHEMA_VERSION = "1.0"
TASK = "NATencoding"
SEG = "Schaefer17n400"

#: Lag scan half-width in TRs (+-30 s at TR 1.5).
MAX_LAG = 20
#: Lags at which a peak is hemodynamically believable: audio leads BOLD by 3-7.5 s.
PLAUSIBLE_LAGS = (-5, -2)
#: Fewest paired volumes a lag correlation is computed from.
MIN_PAIRS = 10

#: Every parcel whose CBIG-current name contains ``_Aud_``: 7 at scale 400
#: (4 LH, 3 RH), the colleague's "'Aud' in name" ROI.
AUD_ROI = "SomMotB_Aud"
AUD_SUBSTRING = "_Aud_"

#: Tier-1 names come from the MNI table; CBIG's fsaverage table is the cross-check.
SCHAEFER_TIER1_TABLE = "tpl-MNI152NLin2009cAsym/anat/tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_seg-17n_scale-400_res-2_dseg.tsv"
SCHAEFER_CURRENT_TABLE = "tpl-fsaverage/anat/tpl-fsaverage_den-41k_atlas-Schaefer2018_seg-17n_scale-400_dseg.tsv"

FEATURE_GROUP = "movies_audio_frames"
#: Envelope kind -> feature-store column.
ENVELOPES = {"linear": "loudness_rms", "log": "loudness_db"}

FLOAT_FORMAT = "%.6f"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def file_sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def movie_name_index(registry_tsv: Path) -> dict[str, str]:
    """Casefolded film name (and every spelling in ``movie_name_variants``) -> ``stimulus_id``."""
    reg = pd.read_csv(registry_tsv, sep="\t", dtype=str, keep_default_na=False)
    index: dict[str, str] = {}
    for row in reg.itertuples(index=False):
        names = [row.movie_name, *[v for v in row.movie_name_variants.split("|") if v]]
        for name in names:
            key = name.strip().casefold()
            if index.get(key, row.stimulus_id) != row.stimulus_id:
                raise ValueError(f"{registry_tsv}: {name!r} maps to both {index[key]} and {row.stimulus_id}")
            index[key] = row.stimulus_id
    return index


def schaefer_current_names(atlases_dir: Path) -> dict[str, str]:
    """Tier-1 (MNI table) Schaefer name -> CBIG's fsaverage-table name, matched by index.

    An identity since the MNI table was restaged from CBIG (2026-09-28); tier-1
    caches built on the TemplateFlow table then carry names absent from the map,
    and :func:`load_series` raises on them.

    Refuses unless both tables have the same indices and every colour agrees:
    the colour is the only evidence that an index means the same parcel in both.
    """
    old = pd.read_csv(Path(atlases_dir) / SCHAEFER_TIER1_TABLE, sep="\t").set_index("index")
    new = pd.read_csv(Path(atlases_dir) / SCHAEFER_CURRENT_TABLE, sep="\t").set_index("index")
    if not old.index.equals(new.index):
        raise ValueError("Schaefer tables index different parcels; cannot map names by index")
    bad = old["color"].str.lower() != new["color"].str.lower()
    if bad.any():
        raise ValueError(f"Schaefer tables disagree on colour at indices {list(old.index[bad])[:10]}; "
                         "an index may not mean the same parcel in both")
    return dict(zip(old["name"].astype(str), new["name"].astype(str)))


def load_tier1_runs(tree_root: Path) -> pd.DataFrame:
    path = Path(tree_root) / "tier1_runs.tsv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; build tier 1 first (`tier1.py run` then `tier1.py collect`)")
    runs = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    runs["absent"] = runs["absent"].str.lower() == "true"
    return runs


def window(onset: float, duration: float, tr: float, n_vol: int) -> tuple[int, int]:
    """``(start, n)``: the volumes lying wholly inside ``[onset, onset + duration)``."""
    eps = 1e-6
    start = math.ceil(onset / tr - eps)
    stop = min(math.floor((onset + duration) / tr + eps), n_vol)
    return start, max(stop - start, 0)


def film_viewings(
    tier1_runs: pd.DataFrame,
    events_path: Callable[[str, str, str], Optional[Path]],
    name_index: dict[str, str],
    task: str = TASK,
) -> pd.DataFrame:
    """Every film showing in the task's runs, with its window and a first-viewing flag.

    ``events_path(sub, ses, run)`` locates a run's events file; a run without
    one is a loud error, because a silently dropped run changes which showing
    counts as first.
    """
    runs = (tier1_runs[tier1_runs["task"] == task]
            .drop_duplicates(["sub", "ses", "run"])
            .sort_values(["sub", "ses", "run"]))
    if runs.empty:
        raise ValueError(f"tier1_runs.tsv has no {task} runs")
    rows = []
    for r in runs.itertuples(index=False):
        path = events_path(r.sub, r.ses, r.run)
        if path is None or not Path(path).exists():
            raise FileNotFoundError(f"No events file for sub-{r.sub} ses-{r.ses} task-{task} run-{r.run}")
        ev = pd.read_csv(path, sep="\t")
        movies = ev[ev["trial_type"] == "movie"]
        tr, n_vol = float(r.repetition_time), int(float(r.n_vol))
        for m in movies.itertuples(index=False):
            key = str(m.movie_name).strip().casefold()
            if key not in name_index:
                raise KeyError(f"{path}: film {m.movie_name!r} is not in the stimulus registry")
            start, n = window(float(m.onset), float(m.duration), tr, n_vol)
            rows.append({
                "sub": r.sub, "ses": r.ses, "run": r.run, "space": r.space,
                "stimulus_id": name_index[key], "movie_name": str(m.movie_name).strip(),
                "onset": float(m.onset), "duration": float(m.duration),
                "repetition_time": tr, "n_vol": n_vol, "start": start, "n": n,
            })
    df = pd.DataFrame(rows).sort_values(["sub", "stimulus_id", "ses", "run", "onset"], ignore_index=True)
    df["showing"] = df.groupby(["sub", "stimulus_id"]).cumcount() + 1
    df["first_viewing"] = df["showing"] == 1
    return df


def series_path(tree_root: Path, v, regime: str, seg: str = SEG, task: str = TASK) -> Path:
    """Tier-1 parcel-series path for one showing's run (naming as ``data_quality.output_stem``)."""
    name = (f"sub-{v.sub}_ses-{v.ses}_task-{task}_run-{v.run}_space-{v.space}"
            f"_seg-{seg}_desc-{regime}_timeseries.tsv")
    return Path(tree_root) / f"sub-{v.sub}" / f"ses-{v.ses}" / "func" / name


def load_series(path: Path, n_vol: int, rename: Optional[dict[str, str]] = None) -> pd.DataFrame:
    """One tier-1 parcel-series table; ``rename`` maps its column names (see :func:`schaefer_current_names`)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; tier 1 is incomplete for this cell (`tier1.py plan`)")
    ts = pd.read_csv(path, sep="\t", na_values=["n/a"])
    if len(ts) != n_vol:
        raise ValueError(f"{path}: {len(ts)} rows, tier1_runs.tsv says n_vol={n_vol}")
    if rename is not None:
        unknown = [c for c in ts.columns if c not in rename]
        if unknown:
            raise KeyError(f"{path}: columns {unknown[:5]} are not in the Schaefer name map")
        ts = ts.rename(columns=rename)
    return ts


def load_envelopes(feature_store: Path, stimulus_ids: Iterable[str]) -> dict[str, pd.DataFrame]:
    """``stimulus_id -> frame table (time, loudness_rms, loudness_db)`` from the Contract B store."""
    from stimuli.plotting import load_group  # duckdb pushdown; the group is ~48 M rows

    wide = load_group(str(feature_store), FEATURE_GROUP, models=["loudness"])
    out = {}
    for sid in sorted(set(stimulus_ids)):
        f = wide[wide["stimulus_id"] == sid]
        if f.empty:
            raise KeyError(f"No '{FEATURE_GROUP}' loudness rows for {sid!r} in {feature_store}")
        out[sid] = f[["time", *ENVELOPES.values()]].sort_values("time").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------

def envelope_on_volumes(frames: pd.DataFrame, column: str, onset: float, start: int, n: int,
                        tr: float) -> np.ndarray:
    """Mean of the frames whose centre falls in each window volume's span, in film time."""
    t = frames["time"].to_numpy(float)
    x = frames[column].to_numpy(float)
    idx = np.floor((t - (start * tr - onset)) / tr).astype(int)
    keep = (idx >= 0) & (idx < n) & np.isfinite(x)
    total = np.bincount(idx[keep], weights=x[keep], minlength=n)
    count = np.bincount(idx[keep], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def covered_prefix(env: np.ndarray) -> int:
    """Length of the envelope's finite leading stretch.

    A stimulus file shorter than its showing (one film's file is truncated on
    disk) leaves the window's tail without frames; the lag scan then runs on
    the covered stretch only. A gap anywhere but the tail is not a truncation
    and is refused.
    """
    finite = np.isfinite(env)
    n = int(np.argmin(finite)) if not finite.all() else len(env)
    if finite[n:].any():
        raise ValueError("envelope has a gap inside the window that is not a trailing truncation")
    return n


def _zscore_columns(x: np.ndarray) -> np.ndarray:
    """Column z-scores (ddof 0); a column with any non-finite value or no variance is all NaN."""
    x = np.asarray(x, dtype=float)
    bad = ~np.isfinite(x).all(axis=0)
    mu = np.where(bad, 0.0, np.nan_to_num(x).mean(axis=0))
    sd = np.where(bad, 0.0, np.nan_to_num(x).std(axis=0))
    bad |= sd == 0
    z = (x - mu) / np.where(sd == 0, 1.0, sd)
    z[:, bad] = np.nan
    return z


def cross_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(N, P) x (N, Q) -> (P, Q)``: Pearson r of every column of ``a`` with every column of ``b``."""
    za, zb = _zscore_columns(a), _zscore_columns(b)
    return (za.T @ zb) / za.shape[0]


def loo_isfc(segments: np.ndarray) -> np.ndarray:
    """``(S, N, P) -> (S, P, P)``; ``[s, p, q]`` = r(subject s parcel p, mean of the others' parcel q)."""
    segments = np.asarray(segments, dtype=float)
    n_sub = segments.shape[0]
    if n_sub < 3:
        raise ValueError(f"LOO-ISFC needs at least 3 subjects, got {n_sub}")
    out = np.empty((n_sub, segments.shape[2], segments.shape[2]))
    for s in range(n_sub):
        others = segments[np.arange(n_sub) != s].mean(axis=0)  # NaN propagates: no partial means
        out[s] = cross_corr(segments[s], others)
    return out


def lag_curves(bold: np.ndarray, env: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    """``(N, P), (N,) -> (2*max_lag + 1, P)``: r per lag, ``-max_lag .. +max_lag``.

    Lag ``L < 0`` pairs ``bold[i - L]`` with ``env[i]`` (audio leads BOLD);
    ``L > 0`` pairs ``bold[i]`` with ``env[i + L]``. The paired stretch shrinks
    with ``|L|``; below ``MIN_PAIRS`` volumes the lag is NaN.
    """
    bold = np.asarray(bold, dtype=float)
    env = np.asarray(env, dtype=float)
    if bold.ndim == 1:
        bold = bold[:, None]
    if not np.isfinite(env).all():
        raise ValueError("envelope has non-finite values inside the window")
    n = min(len(env), bold.shape[0])
    out = np.full((2 * max_lag + 1, bold.shape[1]), np.nan)
    for k, lag in enumerate(range(-max_lag, max_lag + 1)):
        if lag < 0:
            b, e = bold[-lag:n], env[:n + lag]
        else:
            b, e = bold[:n - lag], env[lag:n]
        if len(e) >= MIN_PAIRS:
            out[k] = cross_corr(b, e[:, None])[:, 0]
    return out


def column_r(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(N, P), (N, P) -> (P,)``: Pearson r of each column of ``a`` with the same column of ``b``."""
    za, zb = _zscore_columns(a), _zscore_columns(b)
    return (za * zb).sum(axis=0) / za.shape[0]


def fisher_mean(r: np.ndarray, axis: int = 0) -> np.ndarray:
    """Mean in Fisher-z space; any non-finite r makes that mean n/a."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.arctanh(np.asarray(r, dtype=float)).mean(axis=axis)


def repeat_reliability(segments: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]], int]:
    """Showings of one film by one subject -> ``(r (pairs, P), pairs, n_vol)``, cut to the shortest showing."""
    if len(segments) < 2:
        raise ValueError(f"repeat reliability needs at least 2 showings, got {len(segments)}")
    n = min(s.shape[0] for s in segments)
    pairs = [(i, j) for i in range(len(segments)) for j in range(i + 1, len(segments))]
    return np.stack([column_r(segments[i][:n], segments[j][:n]) for i, j in pairs]), pairs, n


def discriminability(entities: list[tuple[str, str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, int, int]:
    """``(subject, film, segment)`` of one session -> per-parcel ``(within, between, n_within, n_between)``.

    Only pairs of different subjects count; each pair is cut to its shorter segment from the start.
    """
    within, between = [], []
    for a in range(len(entities)):
        for b in range(a + 1, len(entities)):
            (sa, fa, xa), (sb, fb, xb) = entities[a], entities[b]
            if sa == sb:
                continue
            n = min(len(xa), len(xb))
            (within if fa == fb else between).append(column_r(xa[:n], xb[:n]))
    p = entities[0][2].shape[1]
    w = np.mean(within, axis=0) if within else np.full(p, np.nan)
    b = np.mean(between, axis=0) if between else np.full(p, np.nan)
    return w, b, len(within), len(between)


def aud_columns(columns: Iterable[str]) -> list[str]:
    return [c for c in columns if AUD_SUBSTRING in c]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Tier2Result:
    viewings: pd.DataFrame
    loo_isc: pd.DataFrame
    isfc_summary: pd.DataFrame
    envelope_lag: pd.DataFrame
    envelope_parcel: pd.DataFrame
    isfc_group: dict[tuple[str, str], np.ndarray]
    skipped: list[dict]
    wsc: pd.DataFrame
    wsc_pairs: pd.DataFrame
    discriminability: pd.DataFrame


def compute(
    tree_root: Path,
    viewings: pd.DataFrame,
    envelopes: dict[str, pd.DataFrame],
    regimes: list[str],
    tier1_runs: pd.DataFrame,
    rename: Optional[dict[str, str]] = None,
) -> Tier2Result:
    """Every measure for every regime. Reads each run's series once per regime."""
    first = viewings[viewings["first_viewing"]]
    absent = set(tier1_runs.loc[tier1_runs["absent"], ["sub", "ses", "task", "run", "regime"]]
                 .itertuples(index=False, name=None))
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    in_window = (lags >= PLAUSIBLE_LAGS[0]) & (lags <= PLAUSIBLE_LAGS[1])
    isc_rows, summ_rows, lag_rows, parcel_rows, skipped = [], [], [], [], []
    wsc_rows, pair_rows, disc_rows = [], [], []
    shown = viewings.groupby(["sub", "stimulus_id"])["ses"].nunique()
    recurring = viewings.set_index(["sub", "stimulus_id"]).index.isin(shown[shown > 1].index)
    group: dict[tuple[str, str], np.ndarray] = {}

    for regime in regimes:
        cache: dict[tuple[str, str, str], pd.DataFrame] = {}

        def series(v) -> Optional[pd.DataFrame]:
            key = (v.sub, v.ses, v.run)
            if (v.sub, v.ses, TASK, v.run, regime) in absent:
                return None
            if key not in cache:
                cache[key] = load_series(series_path(tree_root, v, regime), v.n_vol, rename)
            return cache[key]

        for sid, film in first.groupby("stimulus_id", sort=True):
            film = film.sort_values("sub")
            segs, subs = [], []
            for v in film.itertuples(index=False):
                ts = series(v)
                if ts is None:
                    skipped.append({"stimulus_id": sid, "regime": regime, "sub": v.sub,
                                    "reason": "tier-1 cell declared absent"})
                    continue
                if subs and list(ts.columns) != parcels:
                    raise ValueError(f"{sid}/{regime}: sub-{v.sub} parcel columns differ from sub-{subs[0]}'s")
                parcels = list(ts.columns)
                seg = ts.to_numpy(float)[v.start:v.start + v.n]
                segs.append(seg)
                subs.append(v.sub)

                aud = aud_columns(parcels)
                aud_seg = seg[:, [parcels.index(c) for c in aud]]
                finite = np.isfinite(aud_seg).all(axis=0)
                roi = aud_seg[:, finite].mean(axis=1) if finite.any() else np.full(len(seg), np.nan)
                for kind, column in ENVELOPES.items():
                    env = envelope_on_volumes(envelopes[sid], column, v.onset, v.start, v.n, v.repetition_time)
                    n_env = covered_prefix(env)
                    env = env[:n_env]
                    roi_curve = lag_curves(roi[:n_env], env)[:, 0]
                    for lag, r in zip(lags, roi_curve):
                        lag_rows.append({"stimulus_id": sid, "regime": regime, "sub": v.sub, "ses": v.ses,
                                         "run": v.run, "envelope": kind, "roi": AUD_ROI,
                                         "n_parcels": int(finite.sum()), "n_vol": n_env, "n_window": int(v.n),
                                         "lag": int(lag), "r": r})
                    curves = lag_curves(seg[:n_env], env)[in_window]
                    has = np.isfinite(curves).any(axis=0)
                    best = np.full(curves.shape[1], -1)
                    best[has] = np.nanargmax(np.where(np.isfinite(curves[:, has]), curves[:, has], -np.inf), axis=0)
                    for j, p in enumerate(parcels):
                        parcel_rows.append({
                            "stimulus_id": sid, "regime": regime, "sub": v.sub, "envelope": kind, "parcel": p,
                            "n_vol": n_env,
                            "r_best": curves[best[j], j] if has[j] else np.nan,
                            "lag_best": int(lags[in_window][best[j]]) if has[j] else pd.NA,
                        })

            if len(segs) < 3:
                skipped.append({"stimulus_id": sid, "regime": regime, "sub": "",
                                "reason": f"ISFC needs >= 3 subjects, {len(segs)} available"})
                continue
            n = min(s.shape[0] for s in segs)
            mats = loo_isfc(np.stack([s[:n] for s in segs]))
            group[(sid, regime)] = mats.mean(axis=0).astype(np.float32)
            off = ~np.eye(mats.shape[1], dtype=bool)
            for s, sub in enumerate(subs):
                diag = np.diag(mats[s])
                for p, r in zip(parcels, diag):
                    isc_rows.append({"stimulus_id": sid, "regime": regime, "sub": sub, "parcel": p, "isc": r})
                offd = mats[s][off]
                summ_rows.append({
                    "stimulus_id": sid, "regime": regime, "sub": sub, "n_subjects": len(subs), "n_vol": n,
                    "isc_median": np.nanmedian(diag), "isfc_offdiag_mean": np.nanmean(offd),
                    "isfc_offdiag_sd": np.nanstd(offd),
                })

        def segment(v) -> Optional[tuple[list[str], np.ndarray]]:
            ts = series(v)
            if ts is None:
                return None
            return list(ts.columns), ts.to_numpy(float)[v.start:v.start + v.n]

        # T2.1: every showing of a recurring film, per subject
        for (sub, sid), g in viewings[recurring].groupby(["sub", "stimulus_id"], sort=True):
            g = g.sort_values(["ses", "run", "onset"])
            got = [(v, segment(v)) for v in g.itertuples(index=False)]
            missing = [v for v, seg in got if seg is None]
            if missing:
                skipped.append({"stimulus_id": sid, "regime": regime, "sub": sub,
                                "reason": f"repeat reliability: {len(missing)} showing(s) declared absent"})
            got = [(v, seg) for v, seg in got if seg is not None]
            if len(got) < 2:
                continue
            names = got[0][1][0]
            r, pairs, n = repeat_reliability([seg[1] for _, seg in got])
            z = fisher_mean(r)
            for p, zz in zip(names, z):
                wsc_rows.append({"stimulus_id": sid, "regime": regime, "sub": sub, "parcel": p,
                                 "n_showings": len(got), "n_pairs": len(pairs), "n_vol": n,
                                 "z_mean": zz, "r": np.tanh(zz)})
            for (i, j), rr in zip(pairs, r):
                pair_rows.append({"stimulus_id": sid, "regime": regime, "sub": sub,
                                  "ses_a": got[i][0].ses, "ses_b": got[j][0].ses, "n_vol": n,
                                  "r_median": np.nanmedian(rr) if np.isfinite(rr).any() else np.nan})

        # T2.3: every showing in a session, across subjects
        for ses, g in viewings.groupby("ses", sort=True):
            entities = []
            for v in g.sort_values(["sub", "stimulus_id"]).itertuples(index=False):
                seg = segment(v)
                if seg is None:
                    skipped.append({"stimulus_id": v.stimulus_id, "regime": regime, "sub": v.sub,
                                    "reason": f"discriminability ses-{ses}: tier-1 cell declared absent"})
                    continue
                entities.append((v.sub, v.stimulus_id, seg[1]))
                names = seg[0]
            if len({e[0] for e in entities}) < 2:
                continue
            w, b, n_w, n_b = discriminability(entities)
            for p, ww, bb in zip(names, w, b):
                disc_rows.append({"ses": ses, "regime": regime, "parcel": p, "within_r": ww, "between_r": bb,
                                  "discriminability": ww - bb, "n_within": n_w, "n_between": n_b})

    return Tier2Result(
        viewings=viewings,
        loo_isc=pd.DataFrame(isc_rows),
        isfc_summary=pd.DataFrame(summ_rows),
        envelope_lag=pd.DataFrame(lag_rows),
        envelope_parcel=pd.DataFrame(parcel_rows),
        isfc_group=group,
        skipped=skipped,
        wsc=pd.DataFrame(wsc_rows),
        wsc_pairs=pd.DataFrame(pair_rows),
        discriminability=pd.DataFrame(disc_rows),
    )


def out_dir(tree_root: Path) -> Path:
    return Path(tree_root) / TIER2_DIR / "naturalistic"


TABLES = ("film_viewings", "loo_isc", "loo_isfc_summary", "envelope_lag", "envelope_parcel",
          "wsc", "wsc_pairs", "discriminability")


def write(result: Tier2Result, dest: Path, provenance: dict) -> list[Path]:
    """Write every table (fixed float format, sorted rows) so a rebuild is byte-identical."""
    dest = Path(dest)
    (dest / "isfc").mkdir(parents=True, exist_ok=True)
    frames = {
        "film_viewings": result.viewings,
        "loo_isc": result.loo_isc,
        "loo_isfc_summary": result.isfc_summary,
        "envelope_lag": result.envelope_lag,
        "envelope_parcel": result.envelope_parcel,
        "wsc": result.wsc,
        "wsc_pairs": result.wsc_pairs,
        "discriminability": result.discriminability,
    }
    written = []
    for name, df in frames.items():
        path = dest / f"{name}.tsv"
        df.to_csv(path, sep="\t", index=False, na_rep="n/a", float_format=FLOAT_FORMAT)
        written.append(path)
    for (sid, regime), mat in sorted(result.isfc_group.items()):
        path = dest / "isfc" / f"stim-{sid}_seg-{SEG}_desc-{regime}_isfc.npy"
        np.save(path, mat)
        written.append(path)
    prov = dict(provenance, schema_version=SCHEMA_VERSION, skipped=result.skipped,
                parameters={"task": TASK, "seg": SEG, "max_lag": MAX_LAG, "plausible_lags": list(PLAUSIBLE_LAGS),
                            "min_pairs": MIN_PAIRS, "aud_roi": AUD_ROI, "envelopes": ENVELOPES,
                            "feature_group": FEATURE_GROUP})
    path = dest / "provenance.json"
    path.write_text(json.dumps(prov, indent=2, default=str) + "\n")
    written.append(path)
    return written


def diff(a: Path, b: Path) -> list[str]:
    """Differences between two tier-2 trees (every table and matrix); empty means identical."""
    a, b = Path(a), Path(b)
    problems = []
    for name in TABLES:
        pa, pb = a / f"{name}.tsv", b / f"{name}.tsv"
        if not (pa.exists() and pb.exists()):
            problems.append(f"{name}.tsv missing in {'a' if not pa.exists() else 'b'}")
        elif file_sha256(pa) != file_sha256(pb):
            problems.append(f"{name}.tsv differs")
    ma = sorted(p.name for p in (a / "isfc").glob("*.npy"))
    mb = sorted(p.name for p in (b / "isfc").glob("*.npy"))
    if ma != mb:
        problems.append(f"isfc/ file sets differ ({len(ma)} vs {len(mb)})")
    for name in set(ma) & set(mb):
        if not np.array_equal(np.load(a / "isfc" / name), np.load(b / "isfc" / name), equal_nan=True):
            problems.append(f"isfc/{name} differs")
    return problems
