#!/usr/bin/env python
"""Render falsifier for the composed TB timelines (tb-timelines Settles-when 2).

`psytwill compose` lays item-store features onto the movie grid and CLAIMS
that is what the movie battery would read out from a real movie of the run.
This script tests that claim on a few runs: it synthesizes the movie the
participant saw (full display geometry, spoken words muxed at onset), runs
the *unchanged* movie battery on it through `stimfeat_campaign.py` under a
staging root, and tables a per-model agreement statistic between the
composed and the rendered features.

Staging root layout (STIMFEAT_BIDS_ROOT for the campaign driver):

    <stage>/composed/<stem>/features/movies_*_features.parquet   composed side
    <stage>/composed/<stem>/movies/<stem>/{frames/,audio.m4a}    full-scale render
    <stage>/stimuli/stimulus_registry/movies.tsv                 the runs as "movies"
    <stage>/stimuli/movies/movie_files/<stem>.mp4                the synthesized movies
    <stage>/derivatives/stimuli_features/movies/<stem>/*.csv     battery cells
    <stage>/derivatives/stimuli_features/psytwill/movies_*.parquet  rendered side
    <stage>/agreement.tsv                                        the artifact

Usage
-----
    tb_render_falsifier.py render  --stage DIR [--runs STEM ...]   # CPU, login node ok
    tb_render_falsifier.py battery --stage DIR --unit STEM          # GPU (sbatch wrapper)
    tb_render_falsifier.py probe-crop --stage DIR --unit STEM       # GPU: visual models on the image region only
    tb_render_falsifier.py compare --stage DIR [-o agreement.tsv]

The crop probe separates two causes a visual miss can have: the battery
reads the whole display (the image is 22% of the frame area on a gray
field), or the image itself is read differently at the rendered size. It
re-runs every visual model on the 768 px image region of the same frames;
`compare` reports that as `stat_cropped` and names the cause from it.

Statistic: per model present in both tables, the median over
stimulus-bearing bins (bins where the composed side has values) of the
cosine between composed and rendered feature vectors. Single-feature models
cannot have a cosine; they report the Pearson correlation across those bins
instead, flagged in `stat`. Bar (charter): >= 0.95 for models with an
identical checkpoint and no `window_sec`. Every model below the bar carries
a `cause`; the causes written here are the script's attribution from the
sidecars and the render geometry, to be confirmed or replaced by a reader.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
import tb_compose_campaign as camp  # noqa: E402  (paths, Run, stores)

BAR = 0.95
DEFAULT_RUNS = (
    "sub-03_ses-04_task-TBencoding_run-01",
    "sub-04_ses-10_task-TBencoding_run-02",
    "sub-05_ses-16_task-TBencoding_run-03",
)
FPS = 25  # the real films are 25 fps ProRes; the battery samples every 0.5 s


def _runs_by_stem(stems) -> list[camp.Run]:
    all_runs = {r.stem: r for r in camp.discover_runs()}
    missing = [s for s in stems if s not in all_runs]
    if missing:
        raise SystemExit(f"unknown run stem(s): {missing}")
    return [all_runs[s] for s in stems]


def _frame_time(p: Path) -> float:
    return float(re.match(r"frame_([\d.]+)\.jpg$", p.name).group(1))


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------
def cmd_render(args) -> int:
    stage = args.stage
    stores = camp.resolve_stores(list(camp.DEFAULT_STORES))
    reg_dir = stage / "stimuli" / "stimulus_registry"
    vid_dir = stage / "stimuli" / "movies" / "movie_files"
    reg_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in _runs_by_stem(args.runs):
        out = stage / "composed" / r.stem
        lead_out = r.lead_out()
        cmd = camp.compose_cmd(r, stores, lead_out, media=True, force=False,
                               sparse=False) + ["--media-scale", "1.0"]
        cmd[cmd.index("-o") + 1] = str(out)
        print(f"compose+render {r.stem} (lead-out {lead_out} s, full scale)")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr[-2000:])
            raise SystemExit(f"compose failed for {r.stem}")
        summary = json.loads(proc.stdout)
        med = summary["media"][r.stem]
        if med["unmapped"]:
            raise SystemExit(f"{r.stem}: {med['unmapped']} unmapped media items")
        run_dir = out / "movies" / r.stem
        frames = sorted(run_dir.glob("frames/frame_*.jpg"), key=_frame_time)
        mp4 = vid_dir / f"{r.stem}.mp4"
        if not mp4.exists() or args.force:
            lst = run_dir / "frames_concat.txt"
            with open(lst, "w") as f:
                for p in frames:
                    f.write(f"file '{p}'\nduration 0.5\n")
                f.write(f"file '{frames[-1]}'\n")  # concat demuxer: last frame needs a repeat
            audio = run_dir / "audio.m4a"
            ff = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst)]
            if audio.exists():
                ff += ["-i", str(audio)]
            ff += ["-r", str(FPS), "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                   "-pix_fmt", "yuv420p"]
            if audio.exists():
                ff += ["-c:a", "aac", "-b:a", "192k"]
            ff += [str(mp4)]
            subprocess.run(ff, check=True)
        dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(mp4)], capture_output=True, text=True,
                             check=True).stdout.strip()
        print(f"  {len(frames)} frames, {med['audio_items']} words -> {mp4.name} ({float(dur):.1f} s)")
        rows.append({"stimulus_id": r.stem, "movie_name": r.stem, "movie_name_variants": "",
                     "video_file": f"movie_files/{mp4.name}", "cue_file": "",
                     "annotation_file": "", "style": "rendered TB run", "duration_s": dur})
    # The campaign driver enumerates every source's units even when asked
    # for one, so the other registry tables must resolve under the stage.
    for src_tsv in camp.REGISTRY_DIR.glob("*.tsv"):
        if src_tsv.name != "movies.tsv" and not (reg_dir / src_tsv.name).exists():
            (reg_dir / src_tsv.name).symlink_to(src_tsv)
    reg = reg_dir / "movies.tsv"
    old = {}
    if reg.exists():
        with open(reg) as f:
            old = {row["stimulus_id"]: row for row in csv.DictReader(f, delimiter="\t")}
    for row in rows:
        old[row["stimulus_id"]] = row
    with open(reg, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(old.values())
    print(f"registry: {reg} ({len(old)} rendered runs)")
    return 0


# ---------------------------------------------------------------------------
# battery
# ---------------------------------------------------------------------------
def _campaign(stage: Path, *argv: str) -> int:
    env = dict(os.environ, STIMFEAT_BIDS_ROOT=str(stage))
    cmd = [sys.executable, str(_SCRIPT_DIR / "stimfeat_campaign.py"), *argv]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env).returncode


def cmd_battery(args) -> int:
    rc = 0
    for source in ("frames", "audio"):
        rc |= _campaign(args.stage, "run", "--set", "movies", "--source", source,
                        "--unit", args.unit, *(["--dry-run"] if args.dry_run else []))
    return rc


# ---------------------------------------------------------------------------
# probe-crop
# ---------------------------------------------------------------------------
IMAGE_FRAC = 0.6  # the task program's image height as a fraction of the screen


def _bearing_times(stage: Path, stem: str) -> list[float]:
    import pandas as pd
    comp = pd.read_parquet(stage / "composed" / stem / "features" / "movies_frames_features.parquet",
                           filters=[("model", "==", "clip")])
    w = comp.pivot_table(index="time", columns="feature", values="value")
    return sorted(w.index[w.notna().any(axis=1)])


def cmd_probe_crop(args) -> int:
    from PIL import Image
    import stimfeat_campaign as sc
    stage, stem = args.stage, args.unit
    pdir = stage / "probe_crop" / stem
    fdir = pdir / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    src = stage / "composed" / stem / "movies" / stem / "frames"
    times = _bearing_times(stage, stem)
    for t in times:
        out = fdir / f"t{t:07.3f}.jpg"
        if out.exists():
            continue
        im = Image.open(src / f"frame_{t:.3f}.jpg")
        W, H = im.size
        s_ = int(IMAGE_FRAC * H)
        im.crop(((W - s_) // 2, (H - s_) // 2, (W - s_) // 2 + s_, (H - s_) // 2 + s_)).save(out, quality=95)
    print(f"{len(times)} bearing frames cropped to the image region under {fdir}")
    models = [m for m in sc.registry("viz2psy") if m not in ("motion",)]
    frames = sorted(str(p) for p in fdir.glob("t*.jpg"))
    rc = 0
    for m in models:
        stem_csv = pdir / f"{m}.csv"
        if stem_csv.with_suffix(".meta.json").exists():
            print(f"  {m}: done")
            continue
        cmd = [sc.PY, "-m", "viz2psy.cli", m, *frames, "-o", str(stem_csv),
               "--batch-size", "64", "--no-viz", "--quiet"]
        if args.dry_run:
            print(" ".join(cmd[:6]), "...")
            continue
        print(f"  {m} ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            rc = 1
            print(f"  {m}: FAILED\n{r.stderr[-1500:]}", flush=True)
    return rc


def _cropped_stat(stage: Path, stem: str, model: str, comp_wide):
    """stat of the crop probe's cells against the composed side, or None."""
    import pandas as pd
    p = stage / "probe_crop" / stem / f"{model}.csv"
    if not p.exists():
        return None
    c = pd.read_csv(p)
    c["time"] = c["filename"].astype(str).str.extract(r"t(\d+\.\d+)\.jpg")[0].astype(float)
    feats = [f for f in comp_wide.columns if f in c.columns]
    if not feats:
        return None
    r = c.set_index("time")[feats]
    stat, kind, *_ = _agreement(comp_wide[feats], r)
    return None if stat != stat else round(stat, 4)


def _audio_by_bin_ordinal(stage: Path, stem: str, comp_w, rend_w):
    """Median cosine on the first bin of each word vs its trailing bin.

    Word files are ~0.54 s on a 0.5 s grid: every word fills one bin and
    spills ~40 ms into the next. In the movie that trailing bin is mostly
    silence; in the clipped file it is the file's end. Splitting the
    agreement by bin ordinal separates that from a model whose window is
    wider than a bin."""
    import numpy as np
    import pandas as pd
    run = _runs_by_stem([stem])[0]
    ev = pd.read_csv(run.events, sep="\t")
    onsets = np.sort(ev.loc[ev["trial_type"] == "word", "onset"].to_numpy(float))
    feats = [c for c in comp_w.columns if c in rend_w.columns]
    on = comp_w.index[comp_w[feats].notna().any(axis=1)]
    t = on.intersection(rend_w.index)
    a = comp_w.loc[t, feats].to_numpy(float); b = rend_w.loc[t, feats].to_numpy(float)
    ok = ~(np.isnan(a).any(1) | np.isnan(b).any(1))
    a, b, tt = a[ok], b[ok], np.asarray(t)[ok]
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    keep = (na > 0) & (nb > 0)
    if keep.sum() == 0 or len(onsets) == 0:
        return None, None
    cos = (a[keep] * b[keep]).sum(1) / (na[keep] * nb[keep])
    idx = np.searchsorted(onsets, tt[keep], side="right") - 1
    ordn = np.floor((tt[keep] - onsets[np.clip(idx, 0, None)]) / 0.5).astype(int)
    first = cos[ordn == 0]; trail = cos[ordn >= 1]
    return (round(float(np.median(first)), 4) if len(first) else None,
            round(float(np.median(trail)), 4) if len(trail) else None)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def _cell_meta(stage: Path, stem: str, model: str) -> dict:
    p = stage / "derivatives" / "stimuli_features" / "movies" / stem / f"{model}.meta.json"
    if not p.exists():
        return {}
    m = json.loads(p.read_text())
    return (m.get("models") or {}).get(model, {})


def _wide(df, model):
    d = df[df["model"] == model]
    return d.pivot_table(index="time", columns="feature", values="value", aggfunc="first")


def _agreement(comp, rend):
    """median per-bin cosine on composed-bearing bins; Pearson if 1-D."""
    import numpy as np
    feats = [c for c in comp.columns if c in rend.columns]
    on = comp.index[comp[feats].notna().any(axis=1)]
    times = on.intersection(rend.index)
    a = comp.loc[times, feats].to_numpy(float)
    b = rend.loc[times, feats].to_numpy(float)
    ok = ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1))
    a, b = a[ok], b[ok]
    if len(a) == 0:
        return float("nan"), "none", len(feats), 0, len(on)
    if len(feats) == 1:
        if a.std() == 0 or b.std() == 0:
            return float("nan"), "pearson_1d", 1, len(a), len(on)
        return float(np.corrcoef(a[:, 0], b[:, 0])[0, 1]), "pearson_1d", 1, len(a), len(on)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    keep = (na > 0) & (nb > 0)
    cos = (a[keep] * b[keep]).sum(axis=1) / (na[keep] * nb[keep])
    return float(np.median(cos)) if len(cos) else float("nan"), "median_cosine", len(feats), int(keep.sum()), len(on)


def cmd_compare(args) -> int:
    import pandas as pd
    stage = args.stage
    for source in ("frames", "audio"):
        if _campaign(stage, "aggregate", "--set", "movies", "--source", source):
            raise SystemExit(f"aggregate failed for movies/{source}")
    agg = stage / "derivatives" / "stimuli_features" / "psytwill"
    rendered = {s: pd.read_parquet(agg / f"movies_{s}_features.parquet")
                for s in ("frames", "audio_frames")}
    rows = []
    for stem in sorted(p.name for p in (stage / "composed").iterdir()):
        cdir = stage / "composed" / stem / "features"
        for stream in ("frames", "audio_frames"):
            cp = cdir / f"movies_{stream}_features.parquet"
            if not cp.exists():
                continue
            comp = pd.read_parquet(cp)
            cmeta = json.loads((cdir / f"movies_{stream}_features.meta.json").read_text())["models"]
            rend = rendered[stream][rendered[stream]["stimulus_id"] == stem]
            r_models = set(rend["model"].unique())
            c_models = set(comp["model"].unique())
            for model in sorted(c_models | r_models):
                row = {"run": stem, "stream": stream, "model": model,
                       "in_composed": model in c_models, "in_rendered": model in r_models}
                rmeta = _cell_meta(stage, stem, model)
                ck_c = (cmeta.get(model) or {}).get("checkpoint")
                ck_r = rmeta.get("checkpoint")
                row.update(checkpoint_composed=ck_c, checkpoint_rendered=ck_r,
                           window_sec=rmeta.get("window_sec"))
                if not (model in c_models and model in r_models):
                    row.update(stat=None, stat_kind=None, n_features=None, n_bins=None,
                               n_bearing_bins=None, bar_applies=False, passes=None,
                               cause="not in both tables: " + (
                                   "movie-only model, nothing to compose from"
                                   if model in r_models else
                                   "composed only; the battery did not emit it"))
                    rows.append(row)
                    continue
                comp_w = _wide(comp, model)
                rend_w = _wide(rend, model)
                stat, kind, nf, nb, non = _agreement(comp_w, rend_w)
                cropped = _cropped_stat(stage, stem, model, comp_w) if stream == "frames" else None
                first_bin = trail_bin = None
                if stream == "audio_frames" and kind == "median_cosine":
                    first_bin, trail_bin = _audio_by_bin_ordinal(stage, stem, comp_w, rend_w)
                bar = (ck_c == ck_r) and rmeta.get("window_sec") in (None, 0) and kind == "median_cosine"
                passes = (stat >= BAR) if (bar and stat == stat) else None
                if bar and passes:
                    cause = ""
                elif ck_c != ck_r:
                    cause = "checkpoint differs between the item store and the battery"
                elif rmeta.get("window_sec"):
                    cause = f"windowed audio model ({rmeta['window_sec']} s): the movie window spans silence and neighbours the clipped word file never had"
                elif kind == "none" and model == "speech_emotion":
                    cause = ("VAD gate: the model scores a 4 s context window only where the mean "
                             "Silero speech probability is >= 0.25; a ~0.54 s word inside silence is "
                             "~13% speech, so every rendered bin is gated to NaN, while the store's "
                             "clipped file is edge-clamped to all word")
                elif kind == "none":
                    cause = ("no numeric bins to compare: the model emits text, or NaN on every "
                             "composed-bearing bin on one side")
                elif kind == "pearson_1d":
                    cause = "single feature: no cosine, Pearson across bins reported; bar not applied"
                elif stream == "frames" and cropped is not None and cropped >= BAR:
                    cause = (f"render canvas: the battery reads the whole 2048x1280 display "
                             f"(image = 22% of the frame on a gray field); on the image region "
                             f"alone the model agrees at {cropped}")
                elif stream == "frames" and cropped is not None:
                    cause = (f"not the canvas: agreement is {cropped} on the 768 px image region "
                             f"alone — the render's LANCZOS upsample or its JPEG q85 encoding "
                             f"(measured 2026-09-22: saliency = the upsample, ebind = the JPEG)")
                elif stream == "frames":
                    cause = "likely render canvas (probe-crop not run for this run/model)"
                elif first_bin is not None and first_bin >= BAR:
                    cause = (f"trailing partial bin: the ~0.54 s word fills one bin and spills "
                             f"~40 ms into the next, which is silence in the movie and the file end "
                             f"in the store; full-word bins agree at {first_bin}, trailing bins at {trail_bin}")
                elif first_bin is not None:
                    cause = (f"model window wider than the bin: even full-word bins agree only at "
                             f"{first_bin} (trailing {trail_bin}) — the movie's silence and neighbours "
                             f"enter the window, the clipped file has none")
                else:
                    cause = "likely silence context: word embedded in a gray-screen soundtrack vs the clipped file"
                row.update(stat=None if stat != stat else round(stat, 4), stat_kind=kind,
                           stat_cropped=cropped, stat_first_bin=first_bin, stat_trailing_bin=trail_bin,
                           n_features=nf, n_bins=nb, n_bearing_bins=non,
                           bar_applies=bar, passes=passes, cause=cause)
                rows.append(row)
        # transcript: the composed text stream comes from the word strings; the
        # rendered path is ASR. Report word recovery, not a feature cosine.
        rows += _transcript_rows(stage, stem)
    out = args.output or (stage / "agreement.tsv")
    df = pd.DataFrame(rows)
    df.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}: {len(df)} rows")
    both = df[df.in_composed & df.in_rendered & df.stream.isin(["frames", "audio_frames"])]
    gated = both[both.bar_applies == True]  # noqa: E712
    print(f"models in both: {both.model.nunique()}; bar applies to {gated.model.nunique()}; "
          f"pass {int((gated.passes == True).sum())}/{len(gated)} (run x model)")
    summ = (both.groupby(["stream", "model"])
            .agg(stat_min=("stat", "min"), stat_median=("stat", "median"),
                 bar=("bar_applies", "first"), kind=("stat_kind", "first"))
            .reset_index().sort_values(["stream", "stat_min"]))
    print(summ.to_string(index=False))
    return 0


def _transcript_rows(stage: Path, stem: str) -> list[dict]:
    import difflib
    import pandas as pd
    cdir = stage / "composed" / stem / "features"
    cp = cdir / "movies_transcript_words_features.parquet"
    if not cp.exists():
        return []
    cmeta = json.loads((cdir / "movies_transcript_words_features.meta.json").read_text())["models"]
    run = _runs_by_stem([stem])[0]
    ev = pd.read_csv(run.events, sep="\t")
    truth = [str(w).lower() for w in ev.loc[ev["trial_type"] == "word", "word"]] \
        if "word" in ev.columns else []
    asr_p = stage / "derivatives" / "stimuli_features" / "movies" / stem / "transcribe_transcript_words.csv"
    if asr_p.exists():
        asr = [str(w).lower().strip(".,!?") for w in pd.read_csv(asr_p)["word"]]
        sm = difflib.SequenceMatcher(a=truth, b=asr, autojunk=False)
        matched = sum(t.size for t in sm.get_matching_blocks())
        cause = (f"ASR path: {matched}/{len(truth)} presented words recovered in order "
                 f"({len(asr)} ASR words); text features are the same model on the same "
                 f"string wherever the word is recovered")
        stat = round(matched / len(truth), 4) if truth else None
    else:
        cause, stat, asr = "ASR path: transcribe cell not run", None, []
    return [{"run": stem, "stream": "transcript_words", "model": m, "in_composed": True,
             "in_rendered": asr_p.exists(), "checkpoint_composed": (cmeta.get(m) or {}).get("checkpoint"),
             "checkpoint_rendered": None, "window_sec": None, "stat": stat,
             "stat_kind": "word_recovery", "n_features": None, "n_bins": len(asr),
             "n_bearing_bins": len(truth), "bar_applies": False, "passes": None, "cause": cause}
            for m in sorted(cmeta)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    p = sub.add_parser("render"); p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    p.add_argument("--force", action="store_true"); p.set_defaults(func=cmd_render)
    p = sub.add_parser("battery"); p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--unit", required=True); p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_battery)
    p = sub.add_parser("probe-crop"); p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--unit", required=True); p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_probe_crop)
    p = sub.add_parser("compare"); p.add_argument("--stage", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path); p.set_defaults(func=cmd_compare)
    args = ap.parse_args(argv)
    args.stage = args.stage.resolve()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
