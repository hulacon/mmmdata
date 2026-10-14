#!/usr/bin/env python3
"""MMMData registers for braintwill's encoding harness (``braintwill.encoding``).

braintwill must not know this dataset exists, so everything that resolves a path, a film
window or a session lives here, and braintwill is imported as a library. The harness scores
any feature set against a register's fixed responses by nested grouped CV and caches the
result by (feature-set content, register spec, subject); see braintwill's
``braintwill/encoding/`` for the method.

Registers:

  films   each subject's 46-film alignment pool from ``functional_space/films.py``: the
          cleaned grayordinate series (89,820 columns) over each film's window, z-scored per
          window; FIR delays, α grid, fold rule and band-weight search are functional-space's
          (``functional_space/encoding.py``), so the film path is a refactor of those
          encoders, not a redesign. Groups are films. **ses-23 and ses-26 never enter**: only
          ``role == "alignment"`` rows are read, and a row from a held-out session is an
          error even if a later table change gave it that role.
  tb-enc  each subject's TBencoding trials (ses-04–18) as GLMsingle TYPED betas on the same
          89,820 grayordinates: fsaverage6 cortex from the surface refit, subcortex from the
          MNI fit's cache; the hippocampus has no TB betas, so its columns are invalid here.
          Betas are z-scored per column within each session; the six super-repeat images
          (shown in every session) are averaged within each session and the session's rows
          re-centred, so each contributes one
          row per session rather than the weight of ~14 ordinary items. Groups: an image, or
          a whole triplet for triplet mates (enCon 3), because TYPED leaks ~28 % of each
          neighbour into a beta and a leaked neighbour must not sit across a fold. Features
          are per image (one-frame stimuli): shared1000 EBind, or psytwill-space V + VL.
  tb-enc-ts  each subject's cleaned TBencoding time series (ses-04–18) on the films' grid,
          FIR delays and α grid, so TB and NAT are fitted the same way. Each run loses its
          first 7 volumes (their delay spans start before the run) and is z-scored per column;
          groups are sessions (repeats and triplet mates cross runs, not sessions). Features
          come from the run's composed tables under ``stimuli_features/tb/`` (run time,
          0.5 s grid): an ``onoff`` band (visual-on, audio-on, from the tables' NaN pattern)
          in every fit, then each band as a modulator — the feature minus its mean over the
          run's on-bins, times on, so off-bins are 0 by design.

Feature sets: for films, the per-stimulus caches ``functional_space/encoding.py cache``
writes (``<derivatives>/functional_space/features/<space>/``), only the register's films;
for tb-enc, the shared1000 image tables under ``<derivatives>/stimuli_features/``. Either
way their content hash is the cache key's feature half.

Scores go to ``<derivatives>/encoding_harness/register-<name>/spec-<hash>/features-<hash>/
sub-##/`` (``scores.npz`` + ``encoding.json``). A cached score is read back, never refit.

Verbs:

  plan         build one subject's register and report sizes, key and cache state
  score        nested-CV scores for one subject (cached)
  equivalence  refit one functional-space stimulus-route job's encoders through braintwill on
               that job's films and compare the fit diagnostics with the job's sidecar; also
               checks the design and responses equal ``encoding.film_design``'s exactly

Usage:
    python harness.py plan --space ebind --sub ##
    python harness.py score --space psytwill --sub ## --backend torch_cuda
    python harness.py score --register tb-enc --space psytwill --sub ##
    python harness.py plan --register tb-enc-ts --space ebind --sub ##
    python harness.py equivalence --space ebind --scenario primary --pct 0 --draw 0 --target ##
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_FS = _REPO / "scripts" / "functional_space"
for _p in (_REPO / "src" / "python", _FS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _import_braintwill():
    """braintwill from the env, the sibling checkout, or ``$BRAINTWILL_SRC``; loud otherwise."""
    try:
        import braintwill.encoding  # noqa: F401
        return
    except ImportError:
        pass
    src = Path(os.environ.get("BRAINTWILL_SRC", _REPO.parent / "braintwill"))
    if not (src / "braintwill" / "encoding" / "__init__.py").exists():
        raise ImportError(f"braintwill.encoding not importable and no checkout at {src}; "
                          "set BRAINTWILL_SRC to the braintwill repository root")
    sys.path.insert(0, str(src))


_import_braintwill()

import encoding as fs_enc  # noqa: E402  functional_space/encoding.py: feature cache, settings
import films as fm  # noqa: E402
import grayordinates as go  # noqa: E402
from braintwill.encoding import (BandedRidge, EncodingSpec, FrameFeatures, RidgeSpec,  # noqa: E402
                                 ScoreCache, cache_key, feature_set_hash, fir_design, item_design,
                                 score)

TREE = "encoding_harness"
#: Outer folds of the nested CV (DECIDED 2026-10-12, encoding-harness settle 1).
OUTER_FOLDS = 5


def film_spec(solver: str = "random_search", n_iter: int | None = None,
              solver_params: dict | None = None) -> EncodingSpec:
    """functional-space's encoder settings, wrapped in the outer CV.

    ``random_search`` with functional-space's ``N_ITER_BANDS`` is the default and the only
    setting the equivalence covers; ``hyper_gradient`` is opt-in for fits with many bands.
    ``solver_params`` go to himalaya verbatim (``TB_TS_SOLVER_PARAMS`` for tb-enc-ts).
    """
    ridge = RidgeSpec(alphas=tuple(fs_enc.ALPHAS), inner_folds=fs_enc.N_FOLDS, solver=solver,
                      n_iter=fs_enc.N_ITER_BANDS if n_iter is None else n_iter, random_state=0,
                      solver_params=dict(solver_params or {}))
    return EncodingSpec(ridge=ridge, outer_folds=OUTER_FOLDS)


class Paths:
    def __init__(self) -> None:
        p = fs_enc.Paths()
        self.derivatives = p.derivatives
        self.features = p.cache
        self.cleaned = go.tree_root(p.derivatives)
        self.stimfeat = p.derivatives / "stimuli_features"
        self.out = p.derivatives / TREE


# ---------------------------------------------------------------------------
# the films register
# ---------------------------------------------------------------------------

def alignment_rows(windows: pd.DataFrame, sub: str, films: list[str] | None = None) -> pd.DataFrame:
    """One subject's alignment-pool window rows (all of them, or the named films).

    The held-out sessions are refused here, not just filtered: a held-out row that reached
    this point with an alignment role would mean the role table changed under the harness.
    """
    rows = windows[(windows["sub"] == sub) & (windows["role"] == "alignment")]
    leaked = rows[rows["ses"].isin(fm.HELDOUT_SESSIONS)]
    if len(leaked):
        raise RuntimeError(f"sub-{sub}: {len(leaked)} alignment rows come from held-out sessions "
                           f"{sorted(set(leaked['ses']))}; the films register must never read them")
    if rows["stimulus_id"].duplicated().any():
        raise RuntimeError(f"sub-{sub}: an alignment film appears twice")
    if films is None:
        if rows["stimulus_id"].nunique() != fm.N_POOL:
            raise RuntimeError(f"sub-{sub}: {rows['stimulus_id'].nunique()} alignment films, "
                               f"expected the pool of {fm.N_POOL}")
        return rows
    missing = sorted(set(films) - set(rows["stimulus_id"]))
    if missing:
        raise RuntimeError(f"sub-{sub}: films not in the alignment pool: {missing}")
    return rows[rows["stimulus_id"].isin(films)]


def frame_features(features_root: Path, space: str, ids) -> list[FrameFeatures]:
    out = []
    for sid in ids:
        f = fs_enc.load_features(fs_enc.cache_path(features_root, space, sid))
        out.append(FrameFeatures(f.stimulus_id, f.times, f.x, fs_enc.FRAME_S, f.bands))
    return out


def film_design(rows: pd.DataFrame, feats: dict[str, FrameFeatures], cleaned: Path):
    """FIR design, per-window z-scored responses and film groups, in row order."""
    xs, ys, groups, bands, runs = [], [], [], None, {}
    for r in rows.itertuples(index=False):
        tr = float(r.repetition_time)
        starts = fs_enc.film_volume_starts(r.onset, r.start, r.n, tr)
        x, bands = fir_design(feats[r.stimulus_id], starts, tr, fs_enc.DELAYS_TR, end=r.play_s)
        xs.append(x)
        ys.append(fm.film_series(r, cleaned, runs))
        groups += [r.stimulus_id] * int(r.n)
    groups = np.array(groups)
    y = fs_enc._zscore_film_windows(np.concatenate(ys), groups)
    return np.concatenate(xs), y, groups, bands


def response_provenance(windows: pd.DataFrame, cleaned: Path) -> dict:
    """What fixes the films register's responses, hashed.

    Every subject's alignment window rows (so one spec serves all subjects) and the cleaned
    series manifest rows they read (regime version, code version, creation time).
    """
    rows = windows[windows["role"] == "alignment"].sort_values(["sub", "stimulus_id"])
    cols = ["sub", "ses", "run", "stimulus_id", "onset", "play_s", "repetition_time", "start", "n",
            "series"]
    manifest = pd.read_csv(Path(cleaned) / "manifest.tsv", sep="\t", dtype=str)
    used = manifest[manifest["path"].isin(set(rows["series"]))]
    if len(used) != rows["series"].nunique():
        raise RuntimeError(f"{len(used)} manifest rows for {rows['series'].nunique()} series")
    used = used.sort_values("path")[["path", "regime", "regime_version", "code_version", "created"]]
    h = hashlib.sha256(rows[cols].to_csv(index=False).encode() + used.to_csv(index=False).encode())
    return {"windows_sha256": h.hexdigest(), "n_series": int(len(used)),
            "regime_versions": sorted(set(used["regime_version"])),
            "code_versions": sorted(set(used["code_version"]))}


def film_register(windows: pd.DataFrame, cleaned: Path, films: list[str]) -> dict:
    trs = sorted(set(windows.loc[windows["role"] == "alignment", "repetition_time"].astype(float)))
    if len(trs) != 1:
        raise RuntimeError(f"alignment windows have several TRs {trs}")
    n_columns = len(pd.read_csv(go.grayordinates_path(cleaned), sep="\t"))
    return {
        "name": "films", "design": "fir", "tr": trs[0], "frame_s": fs_enc.FRAME_S,
        "delays_tr": list(fs_enc.DELAYS_TR), "groups": "film",
        "responses": {"tree": str(Path(cleaned).name), "columns": "grayordinates",
                      "n_columns": n_columns, "standardize": "z-score per film window",
                      **response_provenance(windows, cleaned)},
        "heldout_sessions_excluded": list(fm.HELDOUT_SESSIONS),
        "films": sorted(films),
    }


def _git(repo: Path, paths: tuple[str, ...]) -> dict:
    """HEAD, and whether the paths this code runs from have uncommitted changes.

    Scoped to those paths: the checkouts are shared with other sessions, whose unrelated
    uncommitted files would otherwise mark every run dirty and make the flag meaningless.
    """
    def run(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()
    return {"head": run("rev-parse", "--short", "HEAD"), "dirty_paths": list(paths),
            "dirty": bool(run("status", "--porcelain", "--", *paths))}


def provenance(backend: str, seconds: dict) -> dict:
    import braintwill

    bt = Path(braintwill.__file__).resolve().parents[1]
    mmm = ("scripts/braintwill_encoding", "scripts/functional_space", "src/python/core")
    return {"backend": backend, "seconds": seconds, "mmmdata": _git(_REPO, mmm),
            "braintwill": _git(bt, ("braintwill/encoding", "braintwill/ledger.py", "pyproject.toml")),
            "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}


def _ensure_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    dd = root / "dataset_description.json"
    if not dd.exists():
        dd.write_text(json.dumps({
            "Name": "encoding harness scores", "BIDSVersion": "1.10.0", "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "braintwill.encoding",
                             "Description": "nested grouped-CV banded ridge, per-column held-out R²; "
                                            "driver mmmdata scripts/braintwill_encoding/harness.py"}],
        }, indent=2) + "\n")


# ---------------------------------------------------------------------------
# the tb-enc register
# ---------------------------------------------------------------------------

TB_TREE = Path("functional_space") / "glmsingle_tb_fsaverage6"
TB_SESSIONS = tuple(f"{s:02d}" for s in range(4, 19))  # trial-based sessions (study design)
#: Image feature sets for tb-enc: Contract B tables for the shared1000 set, by band.
#: Image-keyed only, on purpose: the tb-enc mask (reliability.py tb, pair repeats) does not
#: cover auditory cortex, and which mask a word- or audio-keyed TB feature set should be read
#: through is an open decision (mmmdata-agents project-todos, "TB register: a mask for
#: auditory / word-keyed encoding questions"). Settle it before adding such a set here.
TB_SPACES = {
    "ebind": {"csv": "shared1000/ebind.csv", "bands": {"ebind": r"^ebind_\d{4}$"}},
    "psytwill": {"parquet": "psytwill/shared1000_psytwill_space_features.parquet",
                 "bands": {"V": "pspace_v", "VL": "pspace_vl"}},
}


def tb_trials(derivatives: Path, bids_root: Path, sub: str) -> pd.DataFrame:
    """The subject's TBencoding trials in beta order, with each trial's shared1000 ``stimulus_id``."""
    d = Path(derivatives) / TB_TREE / f"sub-{sub}" / "enc"
    t = pd.read_csv(d / "trial_info.csv", dtype={"mmmId": str})
    ses = t["session"].str.removeprefix("ses-")
    if not ses.isin(TB_SESSIONS).all() or ses.isin(fm.HELDOUT_SESSIONS).any():
        raise RuntimeError(f"sub-{sub}: trials outside the trial-based sessions: {sorted(set(ses) - set(TB_SESSIONS))}")
    if not t.groupby(["session", "run"], sort=False)["onset"].apply(lambda o: o.is_monotonic_increasing).all():
        raise RuntimeError(f"sub-{sub}: trial_info is not in time order within runs")
    reg = pd.read_csv(Path(bids_root) / "stimuli" / "stimulus_registry" / "shared1000.tsv", sep="\t",
                      dtype={"mmmId": str})
    t = t.merge(reg[["mmmId", "stimulus_id"]], on="mmmId", how="left", validate="many_to_one")
    if t["stimulus_id"].isna().any():
        raise RuntimeError(f"sub-{sub}: {t['stimulus_id'].isna().sum()} trials have no shared1000 image")
    return t


def tb_betas(derivatives: Path, sub: str, trials: pd.DataFrame, table: pd.DataFrame) -> np.ndarray:
    """``(n_trials, n_grayordinates)`` TYPED betas; columns with no TB fit (hippocampus) are NaN."""
    import families as fam

    d = Path(derivatives) / TB_TREE / f"sub-{sub}" / "enc"
    fit = np.load(d / "glmsingle_outputs" / "TYPED_FITHRF_GLMDENOISE_RR.npy", allow_pickle=True).item()
    b = fit["betasmd"].reshape(fit["betasmd"].shape[0], -1)
    del fit
    if b.shape[1] != len(trials):
        raise RuntimeError(f"sub-{sub}: {b.shape[1]} betas for {len(trials)} trials")
    vi = pd.read_csv(d / "vertex_index.tsv", sep="\t")
    if not np.array_equal(vi["row"].to_numpy(), np.arange(len(vi))) or len(vi) != b.shape[0]:
        raise RuntimeError(f"sub-{sub}: vertex_index does not match the beta rows")
    out = np.full((len(trials), len(table)), np.nan, dtype=np.float32)
    out[:, np.where(vi["hemi"].to_numpy() == "R", go.FSAVERAGE6_N, 0) + vi["vertex"].to_numpy()] = b.T
    sub_trials = pd.read_csv(fam.tb_cache_dir(derivatives, sub) / "trial_info.csv", dtype={"mmmId": str})
    if not trials.drop(columns="stimulus_id").equals(sub_trials):
        raise RuntimeError(f"sub-{sub}: the subcortical cache's trials differ from the fsaverage6 fit's")
    rows, sb, _ = fam.load_tb_subcortex(derivatives, sub, table)
    out[:, rows] = sb.T
    return out


def zscore_sessions(trials: pd.DataFrame, betas: np.ndarray) -> np.ndarray:
    """Each column z-scored over each session's trials (all of them, super repeats included)."""
    y = np.empty_like(betas)
    for _, idx in trials.groupby("session", sort=False).indices.items():
        blk = betas[idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            y[idx] = (blk - blk.mean(0)) / blk.std(0)
    return y


def super_repeats(trials: pd.DataFrame) -> set:
    """The images shown in more than one session (the super repeats)."""
    n_ses = trials.groupby("stimulus_id")["session"].nunique()
    return set(n_ses[n_ses > 1].index)


def tb_rows(trials: pd.DataFrame, betas: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows, responses and groups for tb-enc (z-score per session, super repeats averaged per session)."""
    y = zscore_sessions(trials, betas)
    super_ids = super_repeats(trials)
    if trials.loc[~trials["stimulus_id"].isin(super_ids)].groupby("stimulus_id")["session"].nunique().max() != 1:
        raise RuntimeError("an ordinary item spans sessions")
    con = trials.groupby("stimulus_id")["enCon"].nunique()
    if (con > 1).any():
        raise RuntimeError(f"items with mixed enCon: {sorted(con[con > 1].index)[:3]}")
    group = trials["stimulus_id"].to_numpy(dtype=object).copy()
    trip = (trials["enCon"] == 3).to_numpy()
    sup = trials["stimulus_id"].isin(super_ids).to_numpy()
    group[trip & ~sup] = ("triplet:" + trials["session"] + ":"
                          + ((trials["pairId"] - 1) // 3).astype(int).astype(str))[trip & ~sup]
    group[trip & sup] = "triplet:super"
    sizes = pd.Series(trials.loc[trip & ~sup, "stimulus_id"].to_numpy()).groupby(group[trip & ~sup]).nunique()
    if not (sizes == 3).all():
        raise RuntimeError(f"triplet groups without exactly 3 images: {sizes[sizes != 3].to_dict()}")
    keep, ys, seen = [], [], {}
    for i, r in enumerate(trials.itertuples(index=False)):
        if r.stimulus_id not in super_ids:
            keep.append({"stimulus_id": r.stimulus_id, "session": r.session, "group": group[i], "n_trials": 1})
            ys.append(y[i])
        elif (r.stimulus_id, r.session) not in seen:
            sel = np.flatnonzero((trials["stimulus_id"] == r.stimulus_id).to_numpy()
                                 & (trials["session"] == r.session).to_numpy())
            seen[(r.stimulus_id, r.session)] = len(keep)
            keep.append({"stimulus_id": r.stimulus_id, "session": r.session, "group": group[i],
                         "n_trials": int(sel.size)})
            ys.append(y[sel].mean(axis=0))
    rows, out = pd.DataFrame(keep), np.stack(ys).astype(np.float64)
    # Averaging a session's super-repeat trials into one row each leaves that session's rows
    # off zero mean (by ~0.05–0.08 SD); re-centre them, so responses are zero-mean as the
    # no-intercept ridge and the band split assume. The per-session scale is kept.
    for _, idx in rows.groupby("session", sort=False).indices.items():
        out[idx] -= out[idx].mean(axis=0)
    return rows, out.astype(np.float32)


def image_features(stimfeat: Path, space: str) -> dict[str, FrameFeatures]:
    """One-frame features per shared1000 image, bands in ``TB_SPACES`` order."""
    import re

    spec = TB_SPACES[space]
    if "csv" in spec:
        df = pd.read_csv(Path(stimfeat) / spec["csv"])
        cols, bands = [], {}
        for band, pattern in spec["bands"].items():
            got = [c for c in df.columns if re.match(pattern, c)]
            if not got:
                raise KeyError(f"no column matches band {band!r}")
            bands[band] = slice(len(cols), len(cols) + len(got))
            cols += got
        wide = df.set_index("stimulus_id")[cols]
    else:
        long = pd.read_parquet(Path(stimfeat) / spec["parquet"], columns=["stimulus_id", "model", "feature", "value"])
        parts, bands, at = [], {}, 0
        for band, model in spec["bands"].items():
            w = long[long["model"] == model].pivot(index="stimulus_id", columns="feature", values="value")
            w = w[sorted(w.columns)]
            if w.empty:
                raise KeyError(f"no rows for model {model!r}")
            parts.append(w)
            bands[band] = slice(at, at + w.shape[1])
            at += w.shape[1]
        wide = pd.concat(parts, axis=1, join="outer")
    if wide.index.duplicated().any():
        raise RuntimeError(f"{space}: an image appears twice")
    x = wide.to_numpy(np.float32)
    return {sid: FrameFeatures(sid, [0.0], x[i:i + 1], 1.0, bands) for i, sid in enumerate(wide.index)}


def tb_register(derivatives: Path, subs=fm.SUBJECTS) -> dict:
    h = hashlib.sha256()
    for s in subs:
        d = Path(derivatives) / TB_TREE / f"sub-{s}" / "enc"
        for f in (d / "trial_info.csv", d / "run_metadata.json",
                  Path(derivatives) / "functional_space" / "glmsingle_tb_subcortex" / f"sub-{s}" / "enc"
                  / "subcortex_betas.json"):
            h.update(f.read_bytes())
    return {
        "name": "tb-enc", "design": "item", "rows": "TBencoding trials; super repeats averaged per session",
        "groups": "image; triplet mates (enCon 3) as one group",
        "standardize": "z-score per column per session over trials; rows re-centred per session after super averaging",
        "responses": {"betas": "GLMsingle TYPED betasmd", "columns": "grayordinates",
                      "cortex": str(TB_TREE), "subcortex": "functional_space/glmsingle_tb_subcortex",
                      "hippocampus": "none (invalid)", "metadata_sha256": h.hexdigest()},
        "sessions": list(TB_SESSIONS),
    }


# ---------------------------------------------------------------------------
# the tb-enc-ts register
# ---------------------------------------------------------------------------

TB_TS_TASK = "TBencoding"
#: Leading volumes dropped from every TB run: the longest delay's span of volume ``i`` starts
#: at ``(i - 7) * TR``, so the first 7 volumes have spans before the run started.
TB_TS_DROP = max(fs_enc.DELAYS_TR)
#: The on/off band's streams: each column is "this stream has a value in the bin", read from
#: the run's composed generic table (any model defined), so every feature space gets the same
#: on/off band.
TB_TS_ON = {"visual": "movies_frames", "audio": "movies_audio_frames"}
#: Feature spaces for tb-enc-ts: band -> (composed table, model). A band's stream (and so the
#: on column it is modulated by) is the table's: ``audio`` in the name means audio.
#: himalaya's kernel decomposition for tb-enc-ts fits. float32 ``eigh`` on the GPU failed to
#: converge on the first tb-enc-ts fits (2026-10-14). Likely cause, not verified: the ``onoff``
#: band's kernel has rank <= 10 over thousands of rows, so a band-weight sample favouring it
#: leaves thousands of near-equal eigenvalues. SVD of the same PSD kernel is the same
#: decomposition, computed more stably.
TB_TS_SOLVER_PARAMS = {"diagonalize_method": "svd"}
TB_TS_SPACES = {
    "ebind": {"ebind": ("movies_frames", "ebind")},
    "psytwill": {"V": ("movies_psytwill_space_frames", "pspace_v"),
                 "A": ("movies_psytwill_space_audio_frames", "pspace_a"),
                 "VL": ("movies_psytwill_space_frames", "pspace_vl")},
}


def table_stream(table: str) -> str:
    return "audio" if "audio" in table else "visual"


def tb_ts_runs(manifest: pd.DataFrame, sub: str) -> pd.DataFrame:
    """The subject's cleaned TBencoding runs in (session, run) order; held-out sessions refused."""
    m = manifest[(manifest["sub"] == sub) & (manifest["task"] == TB_TS_TASK)].copy()
    if m.empty:
        raise RuntimeError(f"sub-{sub}: no cleaned {TB_TS_TASK} runs in the manifest")
    if m["ses"].isin(fm.HELDOUT_SESSIONS).any() or not m["ses"].isin(TB_SESSIONS).all():
        raise RuntimeError(f"sub-{sub}: {TB_TS_TASK} runs outside the trial-based sessions: "
                           f"{sorted(set(m['ses']) - set(TB_SESSIONS))}")
    if m.duplicated(["ses", "run"]).any():
        raise RuntimeError(f"sub-{sub}: a (session, run) appears twice in the manifest")
    bad = m[m["n_nss"].astype(int) >= TB_TS_DROP]
    if len(bad):
        raise RuntimeError(f"sub-{sub}: runs with >= {TB_TS_DROP} non-steady-state volumes: {list(bad['path'])}")
    m["stem"] = m["path"].map(lambda p: Path(p).name.split("_space-")[0])
    return m.sort_values(["ses", "run"]).reset_index(drop=True)


def grid_wide(long: pd.DataFrame, model: str, stamp: str, n_bins: int) -> pd.DataFrame:
    """One model's ``(n_bins, n_features)`` table on bin starts ``0, FRAME_S, …``; undefined = NaN."""
    d = long[long["model"] == model]
    if d.empty:
        raise KeyError(f"no rows for model {model!r}")
    start = d["time"].to_numpy(dtype=float) - (fs_enc.FRAME_S / 2 if stamp == "center" else 0.0)
    b = np.rint(start / fs_enc.FRAME_S).astype(int)
    if np.abs(b * fs_enc.FRAME_S - start).max() > 1e-6 or b.min() < 0 or b.max() >= n_bins:
        raise RuntimeError(f"{model}: times are not on the {fs_enc.FRAME_S} s grid of {n_bins} bins")
    w = d.assign(bin=b).pivot(index="bin", columns="feature", values="value")
    return w.reindex(index=range(n_bins), columns=sorted(w.columns))


def stream_on(long: pd.DataFrame, stamp: str, n_bins: int) -> np.ndarray:
    """Bins where any model of the table has a value."""
    start = long["time"].to_numpy(dtype=float) - (fs_enc.FRAME_S / 2 if stamp == "center" else 0.0)
    b = np.rint(start / fs_enc.FRAME_S).astype(int)
    on = np.zeros(n_bins, bool)
    on[b[long["value"].notna().to_numpy()]] = True
    return on


def modulators(f: np.ndarray, on: np.ndarray, name: str = "") -> np.ndarray:
    """``(f - mean of f over on-bins) * on``: centred within the run, 0 where the stream is off.

    The 0 is the design, not a fill (tb-nat-encoding DECIDED 2026-10-13): off-bins are outside
    the feature space and would be dropped but for the delays. A value present while the
    stream is off is an error; an undefined value at an on-bin stays undefined.
    """
    f = np.asarray(f, dtype=np.float64)
    if np.isfinite(f[~on]).any():
        raise RuntimeError(f"{name}: defined values in bins where its stream is off")
    if not on.any():
        raise RuntimeError(f"{name}: the stream is never on")
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(f[on], axis=0)
    if np.isnan(mean).any():
        raise RuntimeError(f"{name}: {int(np.isnan(mean).sum())} features undefined at every on-bin")
    out = np.zeros_like(f)
    out[on] = f[on] - mean
    return out


def run_features(stem: str, tables: dict[str, tuple[pd.DataFrame, str]], space: str,
                 n_bins: int) -> FrameFeatures:
    """One TB run as one frame stimulus: band ``onoff`` (visual-on, audio-on), then the space's
    bands as centred modulators of their stream's on column.

    ``tables`` maps a composed table name to ``(long rows, stamp)``.
    """
    on = {s: stream_on(*tables[t], n_bins) for s, t in TB_TS_ON.items()}
    blocks = [np.stack([on[s] for s in TB_TS_ON], axis=1).astype(np.float64)]
    bands, at = {"onoff": slice(0, len(TB_TS_ON))}, len(TB_TS_ON)
    for band, (table, model) in TB_TS_SPACES[space].items():
        long, stamp = tables[table]
        w = grid_wide(long, model, stamp, n_bins).to_numpy()
        blocks.append(modulators(w, on[table_stream(table)], f"{stem} {band}"))
        bands[band] = slice(at, at + w.shape[1])
        at += w.shape[1]
    return FrameFeatures(stem, np.arange(n_bins) * fs_enc.FRAME_S, np.concatenate(blocks, axis=1),
                         fs_enc.FRAME_S, bands)


def tb_run_dir(stimfeat: Path, sub: str, ses: str, stem: str) -> Path:
    return Path(stimfeat) / "tb" / f"sub-{sub}" / f"ses-{ses}" / stem / "features"


def load_run_features(stimfeat: Path, run, space: str, n_bins: int) -> FrameFeatures:
    """Read a run's composed tables (only the models the space and the on/off band need)."""
    d = tb_run_dir(stimfeat, run.sub, run.ses, run.stem)
    need = {t: None for t in TB_TS_ON.values()}
    for table, model in TB_TS_SPACES[space].values():
        if table not in TB_TS_ON.values():
            need[table] = need.get(table) or set()
            need[table].add(model)
    tables = {}
    for table, models in need.items():
        meta = json.loads((d / f"{table}_features.meta.json").read_text())
        if abs(float(meta["grid"]["window"]) - fs_enc.FRAME_S) > 1e-9:
            raise RuntimeError(f"{d / table}: grid window {meta['grid']['window']} is not {fs_enc.FRAME_S}")
        cols = ["time", "model", "feature", "value"]
        long = pd.read_parquet(d / f"{table}_features.parquet", columns=cols,
                               filters=None if models is None else [("model", "in", sorted(models))])
        tables[table] = (long, meta["grid"]["stamp"])
    return run_features(run.stem, tables, space, n_bins)


def tb_ts_design(runs: pd.DataFrame, feats: dict[str, FrameFeatures], cleaned: Path):
    """FIR design over each run minus its first ``TB_TS_DROP`` volumes, responses z-scored per
    run, groups = session; rows in run order."""
    xs, ys, groups, bands = [], [], [], None
    for r in runs.itertuples(index=False):
        tr = float(r.RepetitionTime)
        n_vol = int(r.n_vol)
        starts = np.arange(TB_TS_DROP, n_vol) * tr
        x, bands = fir_design(feats[r.stem], starts, tr, fs_enc.DELAYS_TR, end=n_vol * tr)
        y = go.load_run(Path(cleaned) / r.path, cleaned).data[TB_TS_DROP:]
        if np.isnan(y).all(axis=1).any():
            raise RuntimeError(f"{r.stem}: a non-steady-state row survives the drop")
        with np.errstate(invalid="ignore", divide="ignore"):
            y = (y - y.mean(0)) / y.std(0)
        xs.append(x)
        ys.append(y.astype(np.float32))
        groups += [f"ses-{r.ses}"] * len(starts)
    return np.concatenate(xs), np.concatenate(ys), np.array(groups), bands


def tb_ts_register(manifest: pd.DataFrame, cleaned: Path) -> dict:
    """What fixes tb-enc-ts's responses, hashed over every subject's TBencoding manifest rows."""
    m = manifest[manifest["task"] == TB_TS_TASK].sort_values("path")
    trs = sorted(set(m["RepetitionTime"].astype(float)))
    if len(trs) != 1:
        raise RuntimeError(f"{TB_TS_TASK} runs have several TRs {trs}")
    cols = ["path", "n_vol", "n_nss", "regime", "regime_version", "code_version", "created"]
    n_columns = len(pd.read_csv(go.grayordinates_path(cleaned), sep="\t"))
    return {
        "name": "tb-enc-ts", "design": "fir", "tr": trs[0], "frame_s": fs_enc.FRAME_S,
        "delays_tr": list(fs_enc.DELAYS_TR), "groups": "session", "drop_leading_volumes": TB_TS_DROP,
        "onoff": {"band": "onoff", "columns": list(TB_TS_ON), "source": "composed table NaN pattern"},
        "modulators": "feature minus its mean over the run's on-bins of its stream, times on",
        "responses": {"tree": str(Path(cleaned).name), "columns": "grayordinates", "n_columns": n_columns,
                      "standardize": "z-score per run after the drop",
                      "manifest_sha256": hashlib.sha256(m[cols].to_csv(index=False).encode()).hexdigest(),
                      "n_series": int(len(m)), "regime_versions": sorted(set(m["regime_version"])),
                      "code_versions": sorted(set(m["code_version"]))},
        "sessions": list(TB_SESSIONS), "heldout_sessions_excluded": list(fm.HELDOUT_SESSIONS),
    }


# ---------------------------------------------------------------------------
# requests: one register, one feature set, one subject
# ---------------------------------------------------------------------------

class Request:
    """A resolved request: the key, and how to build the design and responses when it misses."""

    def __init__(self, args, paths: Paths):
        from core.config import load_config

        self.paths = paths
        self.spec = film_spec(args.solver, args.n_iter,  # tb registers share the film grid and folds
                              TB_TS_SOLVER_PARAMS if args.register == "tb-enc-ts" else None)
        self.register_name = args.register
        if args.register == "films":
            windows = fm.load_windows(paths.derivatives)
            films = args.films.split(",") if getattr(args, "films", None) else None
            self.rows = alignment_rows(windows, args.sub, films)
            feats = frame_features(paths.features, args.space, self.rows["stimulus_id"])
            self.feats = {f.stimulus_id: f for f in feats}
            self.register = film_register(windows, paths.cleaned, list(self.rows["stimulus_id"]))
            self.label = _label(paths, args.space)
        elif args.register == "tb-enc-ts":
            if getattr(args, "films", None):
                raise SystemExit("--films applies to the films register only")
            if args.space not in TB_TS_SPACES:
                raise SystemExit(f"tb-enc-ts spaces: {sorted(TB_TS_SPACES)}")
            manifest = pd.read_csv(Path(paths.cleaned) / "manifest.tsv", sep="\t", dtype=str)
            self.runs = tb_ts_runs(manifest, args.sub)
            n_bins = {int(round(int(r.n_vol) * float(r.RepetitionTime) / fs_enc.FRAME_S))
                      for r in self.runs.itertuples(index=False)}
            if len(n_bins) != 1:
                raise RuntimeError(f"sub-{args.sub}: runs of different lengths {sorted(n_bins)}")
            (n,) = n_bins
            self.feats = {r.stem: load_run_features(paths.stimfeat, r, args.space, n)
                          for r in self.runs.itertuples(index=False)}
            self.register = tb_ts_register(manifest, paths.cleaned)
            self.label = {"space": args.space, "source": str(paths.stimfeat / "tb"),
                          "bands": {b: list(v) for b, v in TB_TS_SPACES[args.space].items()}}
        else:
            if getattr(args, "films", None):
                raise SystemExit("--films applies to the films register only")
            self.bids_root = Path(load_config()["paths"]["bids_project_dir"])
            self.trials = tb_trials(paths.derivatives, self.bids_root, args.sub)
            images = image_features(paths.stimfeat, args.space)
            used = sorted(set(self.trials["stimulus_id"]))
            missing = sorted(set(used) - set(images))
            if missing:
                raise RuntimeError(f"{args.space}: {len(missing)} TB images have no features, e.g. {missing[0]}")
            self.feats = {sid: images[sid] for sid in used}
            self.register = tb_register(paths.derivatives)
            self.label = {"space": args.space, "source": str(paths.stimfeat / next(
                v for k, v in TB_SPACES[args.space].items() if k in ("csv", "parquet")))}
        self.sub = args.sub
        self.key = cache_key(feature_set_hash(self.feats.values()), self.register, self.spec, f"sub-{args.sub}")

    def design(self):
        if self.register_name == "films":
            return film_design(self.rows, self.feats, self.paths.cleaned)
        if self.register_name == "tb-enc-ts":
            return tb_ts_design(self.runs, self.feats, self.paths.cleaned)
        table = pd.read_csv(go.grayordinates_path(self.paths.cleaned), sep="\t")
        rows, y = tb_rows(self.trials, tb_betas(self.paths.derivatives, self.sub, self.trials, table))
        x, bands = item_design(self.feats, rows["stimulus_id"])
        return x, y, rows["group"].to_numpy(), bands

    def summary(self) -> dict:
        first = next(iter(self.feats.values()))
        width = 1 if self.register_name == "tb-enc" else len(fs_enc.DELAYS_TR)
        out = {"register": self.register_name, "sub": self.sub,
               "bands": {b: (sl.stop - sl.start) * width for b, sl in first.bands.items()}}
        if self.register_name == "films":
            out.update(films=len(self.rows), volumes=int(self.rows["n"].sum()))
        elif self.register_name == "tb-enc-ts":
            out.update(runs=len(self.runs), sessions=int(self.runs["ses"].nunique()),
                       volumes=int((self.runs["n_vol"].astype(int) - TB_TS_DROP).sum()))
        else:
            out.update(trials=len(self.trials), images=len(self.feats))
        return out


def _label(paths: Paths, space: str) -> dict:
    side = paths.features / space / "cache.json"
    return {"space": space, "cache": str(paths.features / space),
            "cache_json": json.loads(side.read_text()) if side.exists() else None}


def cmd_plan(args) -> None:
    paths = Paths()
    req = Request(args, paths)
    cache = ScoreCache(paths.out)
    print(json.dumps({**req.summary(), "space": args.space, "spec_hash": req.key["spec_hash"],
                      "feature_hash": req.key["feature_hash"], "path": str(cache.path(req.key)),
                      "cached": cache.get(req.key) is not None}, indent=2))


def cmd_score(args) -> None:
    paths = Paths()
    req = Request(args, paths)
    cache = ScoreCache(paths.out)
    hit = cache.get(req.key)
    if hit is not None:
        print(f"cached: {cache.path(req.key)}\n{json.dumps(hit.summary)}")
        return
    t0 = time.time()
    x, y, groups, bands = req.design()
    t_load = time.time() - t0
    result = score(x, y, groups, bands, req.spec, backend=args.backend)
    seconds = {"load": round(t_load, 1), "score": round(time.time() - t0 - t_load, 1)}
    _ensure_root(paths.out)
    dest = cache.put(req.key, result, provenance(args.backend, seconds), req.label)
    print(f"wrote {dest}\n{json.dumps(result.summary)}\nfull fit {json.dumps(result.full_fit)}")


# ---------------------------------------------------------------------------
# equivalence with functional-space's encoders (settle 2)
# ---------------------------------------------------------------------------

#: How close a braintwill full-data fit must come to a functional-space job's recorded
#: encoder diagnostics. Counts are exact. The sidecar rounds CV quantiles to 4 dp, so they
#: may differ by one unit there. Fractions of columns at a grid edge may differ by GPU
#: float noise flipping a column's chosen α between neighbours: ≤ 0.001 (~80 columns).
TOLERANCE = {"exact": ("n_samples", "n_features", "n_targets", "alpha_median"),
             "cv_quantile_abs": 1e-4, "edge_fraction_abs": 1e-3, "cv_neg_mse_median_abs": 1e-5}


def compare(ours: dict, ref: dict) -> dict:
    out, ok = {}, True
    for k in TOLERANCE["exact"]:
        out[k] = {"ours": ours[k], "ref": ref[k], "pass": ours[k] == ref[k]}
    for k in ("alpha_at_grid_edge", "alpha_at_low_edge", "alpha_at_high_edge"):
        d = abs(ours[k] - ref[k])
        out[k] = {"ours": ours[k], "ref": ref[k], "abs_diff": d, "pass": d <= TOLERANCE["edge_fraction_abs"]}
    d = abs(ours["cv_neg_mse_median"] - ref["cv_neg_mse_median"])
    out["cv_neg_mse_median"] = {"ours": ours["cv_neg_mse_median"], "ref": ref["cv_neg_mse_median"],
                                "abs_diff": d, "pass": d <= TOLERANCE["cv_neg_mse_median_abs"]}
    q = np.abs(np.subtract(ours["cv_1_plus_score_q50_q95_q99"], ref["cv_1_plus_score_q50_q95_q99"]))
    out["cv_1_plus_score_q50_q95_q99"] = {"ours": ours["cv_1_plus_score_q50_q95_q99"],
                                          "ref": ref["cv_1_plus_score_q50_q95_q99"],
                                          "abs_diff": q.tolist(),
                                          "pass": bool(np.all(q <= TOLERANCE["cv_quantile_abs"] + 1e-9))}
    ok = all(v["pass"] for v in out.values())
    return {"pass": bool(ok), "fields": out}


def cmd_equivalence(args) -> None:
    import partitions as pt
    import stimulus_route as sr

    paths = Paths()
    job_dir = sr.out_dir(paths.derivatives, args.space, args.scenario, args.pct, args.draw, args.target)
    side = json.loads((job_dir / "stimulus.json").read_text())
    windows = fm.load_windows(paths.derivatives)
    rows = sr.job_films(pt.load_partitions(paths.derivatives), windows, args.scenario, args.pct, args.draw,
                        args.target)
    spec = film_spec()
    record = {"space": args.space, "job": side["job"], "job_code_version": side["code_version"],
              "job_backend": side["backend"], "tolerance": TOLERANCE, "subjects": {}}
    for s in sorted(rows):
        r = alignment_rows(windows, s, list(rows[s]["stimulus_id"]))  # the held-out guard, same row order
        feats = {f.stimulus_id: f for f in frame_features(paths.features, args.space, r["stimulus_id"])}
        x, y, groups, bands = film_design(r, feats, paths.cleaned)
        # the design and responses are functional-space's, bit for bit
        x0, y0, g0, b0 = fs_enc.film_design(r, paths.features, args.space, paths.cleaned)
        y0 = fs_enc._zscore_film_windows(y0, g0)
        same = {"x": bool(np.array_equal(x, x0)), "y": bool(np.array_equal(y, y0, equal_nan=True)),
                "groups": bool(np.array_equal(groups, g0)), "bands": bands == b0}
        t0 = time.time()
        fit = BandedRidge(bands, spec.ridge, args.backend).fit(x, y, groups)
        ours = fit.diagnostics(x.shape[0], x.shape[1])
        cmp = compare(ours, side["encoders"][f"sub-{s}"])
        cmp["design_identical"] = same
        cmp["pass"] = cmp["pass"] and all(same.values())
        cmp["seconds"] = round(time.time() - t0, 1)
        record["subjects"][f"sub-{s}"] = cmp
        print(f"sub-{s}: pass={cmp['pass']} design={same} "
              f"cv q={ours['cv_1_plus_score_q50_q95_q99']} vs {side['encoders'][f'sub-{s}']['cv_1_plus_score_q50_q95_q99']}",
              flush=True)
    record["pass"] = all(v["pass"] for v in record["subjects"].values())
    record["provenance"] = provenance(args.backend, {})
    _ensure_root(paths.out)
    dest = paths.out / "equivalence" / (f"{args.space}_{args.scenario}_pct-{args.pct:03d}_draw-{args.draw:02d}"
                                        f"_target-{args.target}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{'PASS' if record['pass'] else 'FAIL'} -> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    for name in ("plan", "score"):
        p = sub.add_parser(name)
        p.add_argument("--register", choices=("films", "tb-enc", "tb-enc-ts"), default="films")
        p.add_argument("--space", choices=sorted(set(fs_enc.FEATURE_SPACES) | set(TB_SPACES) | set(TB_TS_SPACES)),
                       required=True)
        p.add_argument("--sub", required=True)
        p.add_argument("--films", help="comma-separated subset of the alignment pool (default: all of it)")
        p.add_argument("--solver", choices=("random_search", "hyper_gradient"), default="random_search")
        p.add_argument("--n-iter", type=int, default=None)
        p.add_argument("--backend", default="torch_cuda")
    e = sub.add_parser("equivalence")
    e.add_argument("--space", choices=list(fs_enc.FEATURE_SPACES), required=True)
    e.add_argument("--scenario", default="primary")
    e.add_argument("--pct", type=int, default=0)
    e.add_argument("--draw", type=int, default=0)
    e.add_argument("--target", required=True)
    e.add_argument("--backend", default="torch_cuda")
    args = ap.parse_args()
    {"plan": cmd_plan, "score": cmd_score, "equivalence": cmd_equivalence}[args.verb](args)


if __name__ == "__main__":
    main()
