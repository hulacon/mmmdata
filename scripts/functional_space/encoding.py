#!/usr/bin/env python3
"""Banded-ridge FIR encoders for the functional-space stimulus route.

Pre-registration §6 (stimulus rows), §8 (encoding ridge); FIR delays and the
virtual probe set DECIDED 2026-09-29 in mmmdata-agents
``docs/workbench/functional-space/``. Per draw, each subject's encoder is fit
on that subject's own alignment films and predicts responses to any film or
probe clip; the stimulus route pairs measured with predicted responses.

Feature spaces (frames on the extractors' 0.5 s grid):

  ebind  one band: the 1,024-d EBind visual embedding
  vgg19  four bands: VGG19 post-ReLU ``conv1_2``, ``conv2_2``, ``conv3_3``,
         ``conv4_3`` at 112 px, each channel averaged over space (Wasserman
         2026's blocks; DECIDED 2026-09-28)

Design: for a volume spanning ``[a, a + TR)`` in film time and a delay of
``d`` TRs, the feature is the mean of the frames whose timestamps fall in
``[a - d*TR, a - d*TR + TR)``. Delays are 3-7 TRs (4.5-10.5 s): with the film
window of ``films.py`` every delay of every window volume lands on real
frames, and a span that does not is an error, never a filled value. Probe
clips get a virtual volume grid from clip time 0, keeping only the volumes
whose every delay lands on the clip.

Ridge (himalaya): features z-scored on the training rows, one linear kernel
per band, per-vertex hyperparameters by grouped CV over films (5 folds). One
band: ``KernelRidgeCV`` over ``ALPHAS``; several: ``MultipleKernelRidgeCV``
random search over band weights x ``ALPHAS``.

Verbs:

  cache  convert the selected feature columns of every film and probe clip to
         <derivatives>/functional_space/features/<space>/<stimulus_id>.npz
  plan   report cached inputs and sizes
  fit    sizing run: fit one subject's encoder for one partition job on its
         own alignment films and time a full probe-set prediction; writes
         only timings and fit diagnostics (alignment data; no score)

Usage:
    python encoding.py cache --space ebind
    python encoding.py cache --space vgg19
    python encoding.py plan
    python encoding.py fit --space ebind --sub 04 --pct 0 --draw 0 --target 03
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "src" / "python", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.config import load_config  # noqa: E402

FRAME_S = 0.5
DELAYS_TR = (3, 4, 5, 6, 7)
N_FOLDS = 5
ALPHAS = np.logspace(-2, 10, 25)
N_ITER_BANDS = 20  # random-search samples of band weights (several bands only)
PROBE_CORPORA = ("movie10", "friends")
_VGG_BLOCKS = ("conv1_2", "conv2_2", "conv3_3", "conv4_3")
FEATURE_SPACES = {
    "ebind": {"file": "ebind.csv", "bands": {"ebind": r"^ebind_\d{4}$"}},
    "vgg19": {"file": "vgg19.csv",
              "bands": {b: rf"^vgg19_{b}_relu112_\d{{4}}$" for b in _VGG_BLOCKS}},
}


class Paths:
    def __init__(self) -> None:
        cfg = load_config()["paths"]
        self.derivatives = Path(cfg["output_dir"])
        self.films = self.derivatives / "stimuli_features" / "movies"
        self.fit_corpora = Path(cfg["fit_corpora_dir"])
        self.cache = self.derivatives / "functional_space" / "features"


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

@dataclass
class Features:
    """Frames of one stimulus in one feature space."""

    stimulus_id: str
    times: np.ndarray  # (n_frames,) seconds from stimulus start
    x: np.ndarray  # (n_frames, n_features) float32
    bands: dict[str, slice] = field(default_factory=dict)


def select_columns(columns: list[str], space: str) -> tuple[list[str], dict[str, slice]]:
    """The space's feature columns in band order, and each band's slice."""
    chosen, bands = [], {}
    for band, pattern in FEATURE_SPACES[space]["bands"].items():
        cols = [c for c in columns if re.match(pattern, c)]
        if not cols:
            raise KeyError(f"no column matches band {band!r} ({pattern})")
        bands[band] = slice(len(chosen), len(chosen) + len(cols))
        chosen += cols
    return chosen, bands


def read_feature_csv(path: Path, space: str) -> Features:
    import pyarrow.csv as pacsv

    header = path.open().readline().rstrip("\n").split(",")
    cols, bands = select_columns(header, space)
    table = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(include_columns=["stimulus_id", "time"] + cols))
    sid = table.column("stimulus_id")[0].as_py()
    times = np.asarray(table.column("time"), dtype=np.float64)
    x = np.column_stack([np.asarray(table.column(c), dtype=np.float32) for c in cols])
    return Features(sid, times, x, bands)


def save_features(f: Features, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, stimulus_id=f.stimulus_id, times=f.times, x=f.x,
             band_names=np.array(list(f.bands)),
             band_bounds=np.array([[s.start, s.stop] for s in f.bands.values()]))


def load_features(path: Path) -> Features:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} is missing; run `encoding.py cache`")
    z = np.load(path)
    bands = {str(n): slice(int(a), int(b)) for n, (a, b) in zip(z["band_names"], z["band_bounds"])}
    return Features(str(z["stimulus_id"]), z["times"], z["x"], bands)


def cache_path(cache_root: Path, space: str, stimulus_id: str) -> Path:
    return Path(cache_root) / space / f"{stimulus_id}.npz"


def probe_sources(fit_corpora: Path, space: str) -> list[Path]:
    fname = FEATURE_SPACES[space]["file"]
    out = []
    for corpus in PROBE_CORPORA:
        clips = sorted((Path(fit_corpora) / corpus / "features").glob(f"*/{fname}"))
        if not clips:
            raise FileNotFoundError(f"no {fname} under {fit_corpora}/{corpus}/features")
        out += clips
    return out


# ---------------------------------------------------------------------------
# design
# ---------------------------------------------------------------------------

def binned(f: Features, span_starts: np.ndarray, tr: float, end: float | None = None) -> np.ndarray:
    """Mean of the frames whose timestamps fall in each ``[a, a + tr)``.

    Every span must lie inside ``[0, end]`` (``end`` defaults to the last
    frame's end; for a film it is the played length) and hold a frame; a span
    that does not is an error.
    """
    t = f.times
    end = t[-1] + FRAME_S if end is None else end
    if np.any(np.abs(np.diff(t) - FRAME_S) > 1e-3) or abs(t[0]) > 1e-3:
        raise ValueError(f"{f.stimulus_id}: frames are not a regular {FRAME_S} s grid from 0")
    lo = np.searchsorted(t, span_starts - 1e-6, side="left")
    hi = np.searchsorted(t, span_starts + tr - 1e-6, side="left")
    if np.any(hi <= lo) or np.any(span_starts < -1e-6) or np.any(span_starts + tr > end + 1e-6):
        raise ValueError(f"{f.stimulus_id}: a volume/delay span has no frames inside the stimulus")
    csum = np.vstack([np.zeros((1, f.x.shape[1]), np.float64), np.cumsum(f.x, axis=0, dtype=np.float64)])
    return ((csum[hi] - csum[lo]) / (hi - lo)[:, None]).astype(np.float32)


def delayed(f: Features, volume_starts: np.ndarray, tr: float, delays=DELAYS_TR, end: float | None = None
            ) -> tuple[np.ndarray, dict[str, slice]]:
    """FIR design ``(n_vol, n_features * n_delays)``, columns grouped by band then delay."""
    per_delay = [binned(f, volume_starts - d * tr, tr, end) for d in delays]
    blocks, bands, at = [], {}, 0
    for band, sl in f.bands.items():
        cols = [x[:, sl] for x in per_delay]
        blocks += cols
        width = sum(c.shape[1] for c in cols)
        bands[band] = slice(at, at + width)
        at += width
    return np.concatenate(blocks, axis=1), bands


def film_volume_starts(onset: float, start: int, n: int, tr: float) -> np.ndarray:
    """Film-time start of each window volume (run volume ``i`` spans ``[i*tr, (i+1)*tr)``)."""
    return np.arange(start, start + n) * tr - onset


def probe_volume_starts(f: Features, tr: float, delays=DELAYS_TR) -> np.ndarray:
    """A virtual volume grid from clip time 0, keeping volumes whose every delay lands on the clip."""
    end = f.times[-1] + FRAME_S
    j = np.arange(int(np.floor(end / tr)) + 1)
    a = j * tr
    ok = (a - max(delays) * tr >= -1e-6) & (a + tr - min(delays) * tr <= end + 1e-6)
    return a[ok]


# ---------------------------------------------------------------------------
# ridge
# ---------------------------------------------------------------------------

def grouped_splits(groups: np.ndarray, n_folds: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds that keep every film whole: films are dealt round-robin in order of first appearance."""
    films = list(dict.fromkeys(groups.tolist()))
    if len(films) < n_folds:
        raise ValueError(f"{len(films)} films cannot fill {n_folds} folds")
    fold_of = {g: i % n_folds for i, g in enumerate(films)}
    fold = np.array([fold_of[g] for g in groups.tolist()])
    return [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(n_folds)]


class Encoder:
    """Banded kernel ridge from a delayed design to the valid grayordinates."""

    def __init__(self, bands: dict[str, slice], alphas=ALPHAS, n_iter: int = N_ITER_BANDS,
                 backend: str = "torch", random_state: int = 0):
        self.bands = bands
        self.alphas = np.asarray(alphas)
        self.n_iter = n_iter
        self.backend = backend
        self.random_state = random_state

    def _pipeline(self, cv):
        from himalaya.backend import set_backend
        from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, KernelRidgeCV, MultipleKernelRidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        set_backend(self.backend, on_error="warn")
        if len(self.bands) == 1:
            model = KernelRidgeCV(alphas=self.alphas, kernel="linear", cv=cv)
            return make_pipeline(StandardScaler(), model)
        kernelizer = ColumnKernelizer([(b, Kernelizer(kernel="linear"), sl) for b, sl in self.bands.items()])
        model = MultipleKernelRidgeCV(
            kernels="precomputed", solver="random_search", cv=cv, random_state=self.random_state,
            solver_params={"n_iter": self.n_iter, "alphas": self.alphas, "progress_bar": False},
        )
        return make_pipeline(StandardScaler(), kernelizer, model)

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "Encoder":
        from himalaya.backend import get_backend

        self.valid_ = np.isfinite(y).all(axis=0)
        self.n_targets_ = y.shape[1]
        self.pipeline_ = self._pipeline(grouped_splits(np.asarray(groups)))
        self.pipeline_.fit(x.astype(np.float32), y[:, self.valid_].astype(np.float32))
        model = self.pipeline_[-1]
        be = get_backend()
        alphas = np.asarray(be.to_numpy(model.best_alphas_))
        low, high = np.isclose(alphas, self.alphas[0]), np.isclose(alphas, self.alphas[-1])
        cv = np.asarray(be.to_numpy(model.cv_scores_))  # (targets,) one band; (search samples, targets) several
        cv = cv.max(axis=0) if cv.ndim > 1 else cv
        self.diagnostics_ = {
            "n_samples": int(x.shape[0]), "n_features": int(x.shape[1]), "n_targets": int(self.valid_.sum()),
            "alpha_at_grid_edge": float(np.mean(low | high)),
            "alpha_at_low_edge": float(np.mean(low)), "alpha_at_high_edge": float(np.mean(high)),
            "alpha_median": float(np.median(alphas)),
            # inner CV over the alignment films (a fit diagnostic, not a score). himalaya's default
            # score is the negative MSE, so on per-window z-scored responses 1 + score ~ CV R^2.
            "cv_neg_mse_median": float(np.median(cv)),
            "cv_1_plus_score_q50_q95_q99": [round(float(v), 4) for v in np.quantile(1.0 + cv, [0.5, 0.95, 0.99])],
        }
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """``(n, n_targets)``; columns not fitted are NaN."""
        from himalaya.backend import get_backend

        pred = np.asarray(get_backend().to_numpy(self.pipeline_.predict(x.astype(np.float32))), dtype=np.float32)
        out = np.full((x.shape[0], self.n_targets_), np.nan, dtype=np.float32)
        out[:, self.valid_] = pred
        return out


# ---------------------------------------------------------------------------
# verbs
# ---------------------------------------------------------------------------

def film_design(windows_rows, cache_root: Path, space: str, cleaned_root: Path, delays=DELAYS_TR
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, slice]]:
    """Stack the FIR design, the measured window series and film groups over showings."""
    import films as fm

    xs, ys, groups, bands, runs = [], [], [], None, {}
    for r in windows_rows.itertuples(index=False):
        f = load_features(cache_path(cache_root, space, r.stimulus_id))
        x, bands = delayed(f, film_volume_starts(r.onset, r.start, r.n, r.repetition_time), r.repetition_time,
                           delays, end=r.play_s)
        xs.append(x)
        ys.append(fm.film_series(r, cleaned_root, runs))
        groups += [r.stimulus_id] * r.n
    return np.concatenate(xs), np.concatenate(ys), np.array(groups), bands


def probe_design(cache_root: Path, space: str, tr: float, stimulus_ids: list[str], delays=DELAYS_TR
                 ) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """FIR design over the probe clips' virtual volumes, and (clip, n_volumes) per clip."""
    xs, sizes = [], []
    for sid in stimulus_ids:
        f = load_features(cache_path(cache_root, space, sid))
        x, _ = delayed(f, probe_volume_starts(f, tr, delays), tr, delays)
        xs.append(x)
        sizes.append((sid, x.shape[0]))
    return np.concatenate(xs), sizes


def probe_ids(cache_root: Path, space: str) -> list[str]:
    ids = sorted(p.stem for p in (Path(cache_root) / space).glob("ext-*.npz"))
    if not ids:
        raise FileNotFoundError(f"no cached probe clips (ext-*) under {cache_root}/{space}; run `encoding.py cache`")
    return ids


def _zscore_film_windows(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Z-score each film window per column (the cleaned series are raw-unit residuals)."""
    out = np.empty_like(y)
    for g in dict.fromkeys(groups.tolist()):
        sel = groups == g
        blk = y[sel]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[sel] = (blk - blk.mean(0)) / blk.std(0)
    return out


def cmd_fit(args: argparse.Namespace) -> None:
    import time

    import films as fm
    import grayordinates as go
    import partitions as pt

    paths = Paths()
    t0 = time.time()
    windows = fm.load_windows(paths.derivatives)
    parts = pt.load_partitions(paths.derivatives)
    job = parts[(parts["scenario"] == args.scenario) & (parts["pct"] == args.pct) & (parts["draw"] == args.draw)
                & (parts["target"] == args.target) & (parts["subject"] == args.sub) & (parts["use"] != "tuning")]
    if len(job) != pt.N_FILMS:
        sys.exit(f"partition job resolves to {len(job)} alignment films, expected {pt.N_FILMS}")
    rows = windows[(windows["sub"] == args.sub) & (windows["role"] == "alignment")
                   & windows["stimulus_id"].isin(job["stimulus_id"])]
    x, y, groups, bands = film_design(rows, paths.cache, args.space, go.tree_root(paths.derivatives))
    y = _zscore_film_windows(y, groups)
    t_load = time.time() - t0
    e = Encoder(bands, backend=args.backend).fit(x, y, groups)
    t_fit = time.time() - t0 - t_load
    tr = float(rows["repetition_time"].iat[0])
    xp, sizes = probe_design(paths.cache, args.space, tr, probe_ids(paths.cache, args.space))
    t1 = time.time()
    n_pred = 0
    for lo in range(0, xp.shape[0], args.chunk):
        n_pred += e.predict(xp[lo: lo + args.chunk]).shape[0]
    t_pred = time.time() - t1
    rec = {
        "space": args.space, "sub": args.sub, "job": {"scenario": args.scenario, "pct": args.pct, "draw": args.draw,
                                                        "target": args.target},
        "train": {"films": int(len(rows)), "volumes": int(x.shape[0]), "features": int(x.shape[1]),
                  "bands": {b: s.stop - s.start for b, s in bands.items()}},
        "probe": {"clips": len(sizes), "volumes": int(xp.shape[0])},
        "diagnostics": e.diagnostics_,
        "seconds": {"load": round(t_load, 1), "fit": round(t_fit, 1), "predict_probe": round(t_pred, 1)},
        "backend": args.backend,
    }
    dest = paths.derivatives / "functional_space" / "dryrun" / f"encoder_{args.space}_sub-{args.sub}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))


def cmd_cache(args: argparse.Namespace) -> None:
    import films as fm

    paths = Paths()
    windows = fm.load_windows(paths.derivatives)
    fname = FEATURE_SPACES[args.space]["file"]
    sources = [paths.films / sid / fname for sid in sorted(windows["stimulus_id"].unique())]
    sources += probe_sources(paths.fit_corpora, args.space)
    missing = [s for s in sources if not s.exists()]
    if missing:
        sys.exit(f"{len(missing)} feature files missing, e.g. {missing[0]}")
    done = 0
    for src in sources:
        f = read_feature_csv(src, args.space)
        dest = cache_path(paths.cache, args.space, f.stimulus_id)
        if dest.exists() and not args.force:
            continue
        if not np.isfinite(f.x).all():
            sys.exit(f"{src}: non-finite features")
        save_features(f, dest)
        done += 1
    side = {"space": args.space, "bands": list(FEATURE_SPACES[args.space]["bands"]),
            "n_stimuli": len(sources), "frame_s": FRAME_S,
            "sources": {"films": str(paths.films), "probe": [str(paths.fit_corpora / c) for c in PROBE_CORPORA]},
            "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    (paths.cache / args.space / "cache.json").write_text(json.dumps(side, indent=2) + "\n")
    print(f"{args.space}: {done} written, {len(sources) - done} already cached, {len(sources)} total")


def cmd_plan(args: argparse.Namespace) -> None:
    paths = Paths()
    for space in FEATURE_SPACES:
        d = paths.cache / space
        n = len(list(d.glob("*.npz"))) if d.exists() else 0
        print(f"{space}: {n} cached stimuli in {d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--space", choices=list(FEATURE_SPACES), required=True)
    c.add_argument("--force", action="store_true")
    sub.add_parser("plan")
    f = sub.add_parser("fit")
    f.add_argument("--space", choices=list(FEATURE_SPACES), required=True)
    f.add_argument("--sub", required=True)
    f.add_argument("--scenario", default="primary")
    f.add_argument("--pct", type=int, default=0)
    f.add_argument("--draw", type=int, default=0)
    f.add_argument("--target", required=True)
    f.add_argument("--backend", default="torch")
    f.add_argument("--chunk", type=int, default=4000)
    args = ap.parse_args()
    {"cache": cmd_cache, "plan": cmd_plan, "fit": cmd_fit}[args.verb](args)


if __name__ == "__main__":
    main()
