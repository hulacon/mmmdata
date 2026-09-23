#!/usr/bin/env python3
"""Trial design table for the neural-rotation pilot: one tidy row per trial.

Built from the TB events files and the catalog's session dates only, so it
can be rebuilt anywhere the BIDS tree and ``inventory/catalog.duckdb`` are
staged and matched to a GLMsingle ``trial_info.csv`` on (session, run, onset).
Every later step of the pilot (folds, item age, anchors, exposure counts)
joins on this table rather than re-deriving it from events.

Outputs, under ``<output_dir>/neural_rotation/sub-##/``:

  sub-##_desc-trials.tsv      one row per trial of interest (see COLUMNS)
  sub-##_desc-anchorlags.tsv  every (session, session') pair per anchor x cue
                              with the lag in days -- the anchor-drift skeleton;
                              similarity columns are filled by the fit step

Rows: encoding = the ``trial_type == image`` row of each TBencoding trial (the
word row shares its onset); retrieval = every non-rest TBretrieval row, with
``modality`` (visual -> image-cued, auditory -> word-cued) cross-checked
against ``cueId``. FINretrieval (ses-30) is out of scope for the pilot.

Folds are leave-one-retrieval-session-out: ``fold`` is the 1-based index of
the retrieval session among the subject's TB retrieval sessions. An encoding
row carries the fold of the session its item is retrieved in, so the test
items of fold f are exactly the rows with ``fold == f`` in either phase.
Anchors (``sharedId == 1``) are retrieved every session and get ``fold = 0``;
they never enter a fit.

The self-check block at the end is the verification the design record asks
for; the script exits 1 if any line fails. Its numbers are findings and go
in the workbench log, not here.

Usage:
    python design.py --subject sub-## [--out-root PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tb = _load_module("glmsingle_tb", SCRIPTS / "glmsingle_tb.py")

TREE = "neural_rotation"
PHASES = ("enc", "ret-word", "ret-image")
MODALITY_TO_PHASE = {"visual": "ret-image", "auditory": "ret-word"}
CUE_TO_PHASE = {1: "ret-image", 2: "ret-word"}
COLUMNS = ["subject", "phase", "session", "run", "onset", "duration", "mmmId",
           "nsdId", "word", "pairId", "sharedId", "anchor", "enCon", "reCon",
           "resp", "resp_RT", "exposure", "ses_date", "enc_date", "age_days",
           "fold", "stimulus_id"]


# ── inputs ───────────────────────────────────────────────────────────────────

def session_dates(inventory_dir: Path, subject: str) -> dict:
    """{'ses-##': date} from the catalog's sessions table (bare labels)."""
    import duckdb

    db = inventory_dir / "catalog.duckdb"
    if not db.exists():
        sys.exit(f"ERROR: catalog missing: {db} (rebuild it, or point --inventory-dir at one)")
    con = duckdb.connect(str(db), read_only=True)
    rows = con.execute("select ses, acq_time from sessions where sub = ?",
                       [subject.replace("sub-", "")]).fetchall()
    con.close()
    out = {}
    for ses, acq in rows:
        if acq in (None, "", "n/a"):
            continue
        out[f"ses-{ses}"] = pd.Timestamp(acq).normalize()
    if not out:
        sys.exit(f"ERROR: no dated sessions for {subject} in {db}")
    return out


def registry(bids_root: Path) -> pd.DataFrame:
    p = bids_root / "stimuli" / "stimulus_registry" / "shared1000.tsv"
    if not p.exists():
        sys.exit(f"ERROR: stimulus registry missing: {p}")
    reg = pd.read_csv(p, sep="\t", dtype={"stimulus_id": str})
    reg["mmmId"] = reg["mmmId"].astype(int)
    reg["nsdId"] = reg["nsdId"].astype(int)
    return reg[["stimulus_id", "mmmId", "nsdId"]]


def _runs(bids_root: Path, subject: str, session: str, task: str) -> list:
    d = bids_root / subject / session / "func"
    return sorted(p.name.split("_run-")[1].split("_")[0]
                  for p in d.glob(f"{subject}_{session}_task-{task}_run-*_events.tsv"))


def _read_events(bids_root, subject, session, task, run) -> pd.DataFrame:
    df = pd.read_csv(tb.events_path(bids_root, subject, session, task, f"run-{run}"),
                     sep="\t", na_values=["n/a"])
    df["session"], df["run"] = session, int(run)
    return df


def load_encoding(bids_root: Path, subject: str) -> pd.DataFrame:
    frames = []
    for ses in tb.TASK_SESSIONS["TBencoding"]:
        for run in _runs(bids_root, subject, ses, "TBencoding"):
            df = _read_events(bids_root, subject, ses, "TBencoding", run)
            frames.append(df[df["trial_type"] == "image"])
    if not frames:
        sys.exit(f"ERROR: no TBencoding events for {subject} under {bids_root}")
    enc = pd.concat(frames, ignore_index=True)
    enc["phase"] = "enc"
    return enc


def load_retrieval(bids_root: Path, subject: str) -> pd.DataFrame:
    frames = []
    for ses in tb.TASK_SESSIONS["TBretrieval"]:
        for run in _runs(bids_root, subject, ses, "TBretrieval"):
            df = _read_events(bids_root, subject, ses, "TBretrieval", run)
            df = df[df["trial_type"] != "rest"].copy()
            by_mod = df["modality"].map(MODALITY_TO_PHASE)
            by_cue = df["cueId"].astype(float).map(CUE_TO_PHASE)
            if by_mod.isna().any() or (by_mod != by_cue).any():
                sys.exit(f"ERROR: {subject} {ses} run-{run}: modality and cueId disagree "
                         f"(modality={sorted(df['modality'].unique())}, "
                         f"cueId={sorted(df['cueId'].unique())})")
            df["phase"] = by_mod
            frames.append(df)
    if not frames:
        sys.exit(f"ERROR: no TBretrieval events for {subject} under {bids_root}")
    return pd.concat(frames, ignore_index=True)


# ── the table ────────────────────────────────────────────────────────────────

def build_trials(enc: pd.DataFrame, ret: pd.DataFrame, dates: dict,
                 reg: pd.DataFrame, subject: str) -> pd.DataFrame:
    df = pd.concat([enc, ret], ignore_index=True)
    df["subject"] = subject
    df["mmmId"] = df["mmmId"].map(tb.norm_mmm).astype(int)
    df["nsdId"] = df["nsdId"].astype(float).astype(int)
    df["sharedId"] = df["sharedId"].astype(float).astype(int)
    df["anchor"] = df["sharedId"] == 1
    for c in ("enCon", "reCon", "resp", "pairId"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    missing = sorted(set(df["session"]) - set(dates))
    if missing:
        sys.exit(f"ERROR: {subject}: sessions without a catalog date: {missing}")
    df["ses_date"] = df["session"].map(dates)

    # exposure index within the encoding phase, in presentation order
    df = df.sort_values(["phase", "session", "run", "onset"], kind="stable")
    is_enc = df["phase"] == "enc"
    df["exposure"] = 1
    df.loc[is_enc, "exposure"] = (df[is_enc].groupby("mmmId").cumcount() + 1).to_numpy()

    # encoding date per item: one session per non-anchor item (asserted)
    enc_ses = df[is_enc & ~df["anchor"]].groupby("mmmId")["session"].nunique()
    multi = enc_ses[enc_ses > 1]
    if len(multi):
        sys.exit(f"ERROR: {subject}: {len(multi)} non-anchor items encoded in more "
                 f"than one session (e.g. mmmId {multi.index[:5].tolist()})")
    enc_date = df[is_enc & ~df["anchor"]].groupby("mmmId")["ses_date"].first()
    df["enc_date"] = df["mmmId"].map(enc_date)           # NaT for anchors
    df["age_days"] = (df["ses_date"] - df["enc_date"]).dt.days
    df.loc[is_enc, "age_days"] = 0

    # folds: retrieval session index; encoding rows inherit their item's fold
    ret_sessions = sorted(df.loc[~is_enc, "session"].unique())
    fold_of_ses = {s: i + 1 for i, s in enumerate(ret_sessions)}
    word = df[(df["phase"] == "ret-word") & ~df["anchor"]]
    dup = word["mmmId"].duplicated()
    if dup.any():
        sys.exit(f"ERROR: {subject}: {dup.sum()} non-anchor items word-cued more than once")
    item_fold = word.set_index("mmmId")["session"].map(fold_of_ses)
    df["fold"] = df["mmmId"].map(item_fold)
    df.loc[~is_enc, "fold"] = df.loc[~is_enc, "session"].map(fold_of_ses)
    df.loc[df["anchor"], "fold"] = 0
    if df["fold"].isna().any():
        bad = df.loc[df["fold"].isna(), "mmmId"].unique()
        sys.exit(f"ERROR: {subject}: {len(bad)} items with no word-cued retrieval "
                 f"(e.g. mmmId {bad[:5].tolist()})")
    df["fold"] = df["fold"].astype(int)

    # stimulus_id from the registry; nsdId must agree with the events
    df = df.merge(reg.rename(columns={"nsdId": "nsdId_reg"}), on="mmmId", how="left")
    if df["stimulus_id"].isna().any():
        bad = df.loc[df["stimulus_id"].isna(), "mmmId"].unique()
        sys.exit(f"ERROR: {subject}: mmmId not in the registry: {bad[:5].tolist()}")
    if (df["nsdId"] != df["nsdId_reg"]).any():
        bad = df.loc[df["nsdId"] != df["nsdId_reg"], "mmmId"].unique()
        sys.exit(f"ERROR: {subject}: events nsdId disagrees with the registry for "
                 f"mmmId {bad[:5].tolist()}")

    df = df.sort_values(["phase", "session", "run", "onset"], kind="stable")
    out = df[COLUMNS].copy()
    for c in ("ses_date", "enc_date"):
        out[c] = out[c].dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def build_anchor_lags(trials: pd.DataFrame) -> pd.DataFrame:
    """Session pairs (s < s') per anchor x cue: lag and absolute time in days.

    abs_time_* are days since the subject's first TB session (encoding or
    retrieval, whichever is earlier); abs_time_days is the pair's midpoint,
    the regressor the relative-vs-absolute-time readout uses beside lag_days.
    """
    anc = trials[trials["anchor"] & (trials["phase"] != "enc")].copy()
    anc["ses_date"] = pd.to_datetime(anc["ses_date"])
    t0 = pd.to_datetime(trials["ses_date"]).min()
    rows = []
    for (mmm, phase), g in anc.groupby(["mmmId", "phase"]):
        g = g.drop_duplicates("session").sort_values("ses_date")
        recs = list(zip(g["session"], g["ses_date"]))
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                (sa, da), (sb, db) = recs[i], recs[j]
                ta, tb_ = (da - t0).days, (db - t0).days
                rows.append({"subject": trials["subject"].iloc[0], "cue": phase[4:],
                             "mmmId": mmm, "ses_a": sa, "ses_b": sb,
                             "lag_days": tb_ - ta, "abs_time_a_days": ta,
                             "abs_time_b_days": tb_, "abs_time_days": (ta + tb_) / 2,
                             "similarity": np.nan, "plane_angle_delta": np.nan})
    return pd.DataFrame(rows)


# ── self-check ───────────────────────────────────────────────────────────────

def self_check(t: pd.DataFrame, features_csv: Path | None) -> bool:
    """Print the verification block; return True when every line holds."""
    ok_all = True

    def line(label, value, ok):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"  [{'ok' if ok else 'FAIL'}] {label}: {value}")

    non = t[~t["anchor"]]
    enc, w, im = (non[non["phase"] == p] for p in PHASES)
    n_items = t["mmmId"].nunique()
    anchors = sorted(int(v) for v in t.loc[t["anchor"], "mmmId"].unique())
    line("unique items", n_items, n_items == 1000)
    line("anchors", anchors, anchors == list(range(995, 1001)))
    line("non-anchor items", non["mmmId"].nunique(), non["mmmId"].nunique() == 994)
    exp = enc.groupby("mmmId").size()
    encon = enc.groupby("mmmId")["enCon"].first()
    line("enc exposures per item {n: items}", exp.value_counts().sort_index().to_dict(),
         set(exp.unique()) <= {1, 3} and ((encon == 1) == (exp == 1)).all())
    line("items x1 / x3", f"{(exp == 1).sum()} / {(exp == 3).sum()}",
         (exp == 1).sum() == 336 and (exp == 3).sum() == 658)
    line("ret-word rows = items", f"{len(w)} rows, {w['mmmId'].nunique()} items",
         len(w) == 994 and w["mmmId"].nunique() == 994)
    line("ret-image rows = items", f"{len(im)} rows, {im['mmmId'].nunique()} items",
         len(im) == 994 and im["mmmId"].nunique() == 994)
    pair = w.set_index("mmmId")[["session", "run"]].join(
        im.set_index("mmmId")[["session", "run"]], rsuffix="_im")
    line("word/image retrievals same session, different run",
         f"{(pair['session'] == pair['session_im']).sum()} same-session, "
         f"{(pair['run'] != pair['run_im']).sum()} different-run",
         (pair["session"] == pair["session_im"]).all() and (pair["run"] != pair["run_im"]).all())
    recon = w.groupby("reCon").size().to_dict()
    line("reCon within/across", recon, recon.get(1.0) == 497 and recon.get(2.0) == 497)
    line("age_days == 0 iff reCon == 1 (word rows)", "",
         ((w["age_days"] == 0) == (w["reCon"] == 1)).all() and (w["age_days"] >= 0).all())
    folds = w.groupby("fold").size()
    line("folds (word-cued non-anchor items per fold)", folds.to_dict(),
         len(folds) == 15 and folds.min() >= 36 and folds.max() <= 72)
    line("enc rows carry their item's retrieval fold", "",
         (enc.groupby("mmmId")["fold"].nunique() == 1).all()
         and enc.groupby("mmmId")["fold"].first().equals(w.set_index("mmmId")["fold"].reindex(enc.groupby("mmmId")["fold"].first().index)))
    if features_csv is not None and features_csv.exists():
        feat = pd.read_csv(features_csv, usecols=["stimulus_id"], dtype=str)
        joined = t["stimulus_id"].drop_duplicates().isin(feat["stimulus_id"]).sum()
        line("stimulus_id joins the feature file", f"{joined}/1000", joined == 1000)
    else:
        line("stimulus_id joins the feature file", "SKIPPED (file absent)", True)
    return ok_all


# ── main ─────────────────────────────────────────────────────────────────────

def ensure_dataset_description(root: Path, bids_root: Path) -> None:
    dd = root / "dataset_description.json"
    if dd.exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    with open(dd, "w") as f:
        json.dump({
            "Name": "Neural-rotation pilot: trial design tables, ROI ladder "
                    "caches and operator fits",
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "mmmdata/scripts/neural_rotation/",
                             "Description": "design record: mmmdata-agents "
                                            "docs/workbench/neural-rotation-pilot/"}],
            "SourceDatasets": [{"URL": str(bids_root)}],
        }, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, help="e.g. sub-##")
    ap.add_argument("--out-root", default=None,
                    help=f"override <output_dir>/{TREE}")
    ap.add_argument("--inventory-dir", default=None, help="override config inventory_dir")
    ap.add_argument("--features", default=None,
                    help="feature CSV whose stimulus_id column the table must join "
                         "(default: <output_dir>/stimuli_features/shared1000/clip.csv)")
    ap.add_argument("--dry-run", action="store_true", help="build and check; write nothing")
    args = ap.parse_args()

    cfg = tb.load_config()
    bids_root = Path(cfg["bids_project_dir"])
    output_dir = Path(cfg["output_dir"])
    inventory_dir = Path(args.inventory_dir or cfg["inventory_dir"])
    out_root = Path(args.out_root) if args.out_root else output_dir / TREE
    features = Path(args.features) if args.features else output_dir / "stimuli_features" / "shared1000" / "clip.csv"

    print(f"=== design table: {args.subject} ===")
    dates = session_dates(inventory_dir, args.subject)
    enc = load_encoding(bids_root, args.subject)
    ret = load_retrieval(bids_root, args.subject)
    trials = build_trials(enc, ret, dates, registry(bids_root), args.subject)
    lags = build_anchor_lags(trials)
    print(f"{len(trials)} rows: " + ", ".join(
        f"{p} {int((trials['phase'] == p).sum())}" for p in PHASES)
        + f"; {len(lags)} anchor session pairs")

    print("self-check:")
    ok = self_check(trials, features)
    if not ok:
        sys.exit(f"ERROR: {args.subject}: self-check FAILED — the events differ from the "
                 "design record; do not build on this table")
    if args.dry_run:
        print("DRY RUN — nothing written.")
        return
    ensure_dataset_description(out_root, bids_root)
    sub_dir = out_root / args.subject
    sub_dir.mkdir(parents=True, exist_ok=True)
    p_trials = sub_dir / f"{args.subject}_desc-trials.tsv"
    p_lags = sub_dir / f"{args.subject}_desc-anchorlags.tsv"
    trials.to_csv(p_trials, sep="\t", index=False, na_rep="n/a")
    lags.to_csv(p_lags, sep="\t", index=False, na_rep="n/a")
    print(f"wrote {p_trials}\nwrote {p_lags}")


if __name__ == "__main__":
    main()
