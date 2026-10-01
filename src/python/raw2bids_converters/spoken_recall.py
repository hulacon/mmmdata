#!/usr/bin/env python3
"""Convert spoken-recall transcripts into BIDS TSVs.

One converter for all spoken free-recall material, distinguished by time
base rather than by session:

  recording — out-of-scanner audio; word onsets are relative to the audio
      recording start. Used by ses-29 final free recall ->
      sub-XX/ses-29/beh/sub-XX_ses-29_task-FINrecall_beh.tsv
      Input is the aud2psy `transcribe` output (standard arm) staged at
      <mmmsourcedata>/derivatives/recall_transcripts/sub-XX/ses-29/standard/;
      provenance is copied from its .meta.json into the sidecar.
  scanner — in-scanner NATretrieval recall. Input is the human-vetted
      per-trial word tables already in each session's beh/
      (recording_mic_<datetime>_word_timestamps.csv, one per recall
      trial), which are read and never modified. Output is one
      sub-XX_ses-YY_task-NATretrieval[_run-ZZ]_beh.tsv per BOLD run, whose
      entities copy that run's bold file. Each per-trial recording spans
      the trial's recall routine (recall1.started -> recall1.stopped), so a
      word's onset on the run clock is recall1.started + its recording
      time. See scanner_events() for how gaps in that rule are carried.

Raw audio stays in mmmsourcedata (PII separation); only word tables enter
the BIDS tree.

Usage:
    python spoken_recall.py 03 [04 05] [--dry-run] [--status automatic]
    python spoken_recall.py 03 04 05 --time-base scanner [--sessions 19 20]
"""

import argparse
import contextlib
import glob
import io
import json
import os
import re
import wave

import pandas as pd

from common import (
    BIDS_ROOT, SOURCE_DIR, bids_sub, write_beh_tsv, write_json_sidecar,
)

# SOURCE_DIR resolves from config/base.toml to the post-migration
# mmmsourcedata root (a sibling of the BIDS root, not <bids>/sourcedata).
TRANSCRIPTS_ROOT = f"{SOURCE_DIR}/derivatives/recall_transcripts"

FINAL_RECALL_SESSION = 29
TASK = "FINrecall"
FILLERS = {"um", "uh", "hmm", "mm", "mhm"}


def transcript_dir(sub_num, arm="standard"):
    return f"{TRANSCRIPTS_ROOT}/{bids_sub(sub_num)}/ses-{FINAL_RECALL_SESSION}/{arm}"


def load_words(sub_num, arm="standard"):
    """Load the aud2psy word-level table for one subject."""
    path = f"{transcript_dir(sub_num, arm)}/recall_transcript_words.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No transcript for {bids_sub(sub_num)}: {path}\n"
            f"Generate it with mmmdata/scripts/ses29_recall_transcribe.sbatch"
        )
    return pd.read_csv(path)


def load_provenance(sub_num, arm="standard"):
    """Extraction provenance from the aud2psy .meta.json sidecar."""
    path = f"{transcript_dir(sub_num, arm)}/recall.meta.json"
    with open(path) as f:
        meta = json.load(f)
    model = meta.get("models", {}).get("transcribe", {})
    return {
        "extractor": meta.get("extractor"),
        "extractor_version": meta.get("aud2psy_version"),
        "schema_version": meta.get("schema_version"),
        "checkpoint": model.get("checkpoint"),
        "backend_version": model.get("package_version"),
    }


def build_events(words):
    """aud2psy word table -> BIDS beh table (recording time base)."""
    onset = words["onset"].astype(float)
    offset = words["offset"].astype(float)
    clean = words["word"].astype(str).str.strip(".,!?…'\"").str.lower()
    return pd.DataFrame({
        "onset": onset.round(3),
        "duration": (offset - onset).round(3),
        "word": words["word"].astype(str),
        "segment_idx": words["segment_idx"].astype(int),
        "asr_probability": words["probability"].astype(float).round(4),
        "filler": clean.isin(FILLERS).map({True: 1, False: 0}),
    })


def sidecar(sub_num, status, provenance):
    return {
        "TaskName": TASK,
        "TaskDescription": (
            "Final free recall (out-of-scanner): the participant verbally "
            "recalled everything they remembered from the study while "
            "being audio-recorded. One row per transcribed word."
        ),
        "TimeBase": (
            "Seconds from the start of the audio recording. The recording "
            "is not synchronized to any scanner clock (behavioral-only "
            "session)."
        ),
        "TranscriptStatus": status,
        "TranscriptionPipeline": provenance,
        "SourceAudio": (
            "Raw audio remains in the private mmmsourcedata tree "
            "(PII separation); it is deliberately not part of this dataset."
        ),
        "onset": {"Description": "Word onset in seconds from recording start.",
                  "Units": "s"},
        "duration": {"Description": "Word duration (ASR offset - onset). "
                     "Whisper word timings are approximate (~200 ms scale); "
                     "use a forced aligner if finer timing is needed.",
                     "Units": "s"},
        "word": {"Description": "Transcribed word, punctuation as emitted."},
        "segment_idx": {"Description": "Whisper segment the word belongs to."},
        "asr_probability": {"Description": "ASR per-word probability."},
        "filler": {"Description": "1 if the word is a filled pause "
                   "(um/uh/hmm/mm/mhm), else 0.",
                   "Levels": {"0": "lexical word", "1": "filled pause"}},
    }


def convert_final_recall(sub_num, dry_run=False, status="automatic",
                         arm="standard"):
    """ses-29 (recording time base) -> beh.tsv + sidecar. Returns paths."""
    words = load_words(sub_num, arm)
    events = build_events(words)
    sub = bids_sub(sub_num)
    ses = f"ses-{FINAL_RECALL_SESSION}"
    out_dir = f"{BIDS_ROOT}/{sub}/{ses}/beh"
    tsv = f"{out_dir}/{sub}_{ses}_task-{TASK}_beh.tsv"

    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)
    write_beh_tsv(events, tsv, dry_run=dry_run)
    write_json_sidecar(
        sidecar(sub_num, status, load_provenance(sub_num, arm)),
        tsv.replace("_beh.tsv", "_beh.json"), dry_run=dry_run,
    )
    print(f"{sub}: {len(events)} words -> {tsv}"
          f"{' (dry run)' if dry_run else ''}")
    return tsv


# ── scanner time base: in-scanner NATretrieval recall ───────────────────────

NAT_TASK = "NATretrieval"
NAT_TRANSCRIPT_GLOB = "recording_mic_*_word_timestamps.csv"
_STAMP = re.compile(r"^recording_mic_(\d{4}-\d{2}-\d{2}_\d{2}h\d{2}\.\d{2}\.\d{3})")

# A per-trial recording is the recall routine minus a constant ~0.6 s, so a
# vetted word time can pass the routine end by at most that much.
ROUTINE_SLACK = 1.0
# Past this share of off-window words a trial is not carrying typos, it is
# paired with the wrong transcript -- refuse rather than mask.
MAX_OFF_WINDOW_FRACTION = 0.05

# Sessions whose single, run-less NATretrieval events file belongs to the
# second BOLD run because the first run crashed. Keyed on real labels because
# the events file carries no run entity: nothing in the data says which run.
EVENTS_RUN = {(4, 20): "02", (4, 24): "02"}

# Trials skipped on screen (~1 s each) because their films were recalled in a
# crashed earlier run. The vetted files in those trial slots hold that earlier
# speech, which is not on this run's clock, so they are withheld until the
# earlier recording can be placed on its own run's clock.
WITHHELD_TRIALS = {(4, 24): {1, 2, 3, 4, 5}}


def read_vetted_transcript(path):
    """One vetted word table -> (word/start/end frame, blank rows dropped).

    Read-only. Some files were re-saved from a spreadsheet (cp1252, a BOM,
    trailing empty columns), and NA parsing is off because "None" is a word
    people say.
    """
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    table = pd.read_csv(io.StringIO(text), keep_default_na=False, dtype=str)
    missing = {"word", "start", "end"} - set(table.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    words = table[["word", "start", "end"]].copy()
    blank = words["word"].str.strip() == ""
    words = words[~blank].reset_index(drop=True)
    words["start"] = pd.to_numeric(words["start"], errors="raise")
    words["end"] = pd.to_numeric(words["end"], errors="raise")
    return words, int(blank.sum())


def nat_transcripts(sub, ses):
    """Vetted per-trial transcripts for one session, in recording order."""
    paths = glob.glob(f"{BIDS_ROOT}/{sub}/{ses}/beh/{NAT_TRANSCRIPT_GLOB}")
    stamped = []
    for path in paths:
        m = _STAMP.match(os.path.basename(path))
        if not m:
            raise ValueError(f"unparseable transcript name: {path}")
        stamped.append((m.group(1), path))
    return [(stamp, path) for stamp, path in sorted(stamped)]


def recording_seconds(sub, ses, stamp):
    """Duration of the per-trial source recording, or None if absent."""
    path = f"{SOURCE_DIR}/{sub}/{ses}/audio/recording_mic_{stamp}.wav"
    if not os.path.exists(path):
        return None
    with contextlib.closing(wave.open(path)) as w:
        return w.getnframes() / w.getframerate()


def nat_target_run(sub_num, ses_num, sub, ses):
    """The BOLD run the session's events file belongs to: (events path, run).

    run is None when the bold file has no run entity.
    """
    events = sorted(glob.glob(
        f"{BIDS_ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-{NAT_TASK}*_events.tsv"))
    if len(events) != 1:
        raise ValueError(f"{sub} {ses}: expected one {NAT_TASK} events file, "
                         f"found {len(events)}")
    bolds = glob.glob(
        f"{BIDS_ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-{NAT_TASK}*_bold.nii.gz")
    runs = sorted({(re.search(r"_run-(\d+)_", b) or [None, None])[1]
                   for b in bolds}, key=str)
    if not runs:
        raise ValueError(f"{sub} {ses}: no {NAT_TASK} bold file")
    events_run = re.search(r"_run-(\d+)_", os.path.basename(events[0]))
    if events_run:
        run = events_run.group(1)
    elif len(runs) == 1:
        run = runs[0]
    elif (sub_num, ses_num) in EVENTS_RUN:
        run = EVENTS_RUN[(sub_num, ses_num)]
    else:
        raise ValueError(f"{sub} {ses}: run-less events file but bold runs "
                         f"{runs}; add the session to EVENTS_RUN")
    if run not in runs:
        raise ValueError(f"{sub} {ses}: events mapped to run-{run}, "
                         f"no such bold run among {runs}")
    return events[0], run


def scanner_events(trials, transcripts, withheld=()):
    """Vetted per-trial words -> one word table on the run clock.

    trials: recall rows of the events file (trial_num, movie_name,
        recall1.started, recall1.stopped), in trial order.
    transcripts: [(words frame, recording seconds or None)], same order.

    Vetted times are carried unaltered in recording_onset/recording_offset.
    The run-clock onset is recall1.started + recording_onset, and is n/a
    only where the vetted start falls outside the trial's routine (a typo
    in the vetted file). duration is n/a where the vetted end precedes the
    start or leaves the routine. time_zero is 'routine_start_assumed' when
    the per-trial recording is missing or shorter than the transcript,
    i.e. the words were taken from another recording and their zero is
    inferred rather than shown.
    """
    if len(trials) != len(transcripts):
        raise ValueError(f"{len(transcripts)} transcripts for "
                         f"{len(trials)} recall trials")
    frames = []
    for trial, (words, rec_seconds) in zip(trials.to_dict("records"),
                                           transcripts):
        if int(trial["trial_num"]) in withheld:
            continue
        started = float(trial["recall1.started"])
        window = float(trial["recall1.stopped"]) - started + ROUTINE_SLACK
        start = words["start"].astype(float)  # empty tables arrive as object
        end = words["end"].astype(float)
        in_window = (start >= 0) & (start <= window)
        if len(words) and (~in_window).mean() > MAX_OFF_WINDOW_FRACTION:
            raise ValueError(
                f"trial {trial['trial_num']}: {(~in_window).sum()} of "
                f"{len(words)} words start outside the {window:.1f} s "
                f"routine; transcript is likely paired with the wrong trial")
        good_end = in_window & (end >= start) & (end <= window)
        last_word = float(end[good_end].max()) if good_end.any() else 0.0
        shown = (rec_seconds is not None
                 and last_word <= rec_seconds + ROUTINE_SLACK)
        clean = words["word"].str.strip(".,!?…'\"").str.lower()
        frames.append(pd.DataFrame({
            "onset": (started + start).where(in_window).round(3),
            "duration": (end - start).where(good_end).round(3),
            "word": words["word"],
            "trial_num": int(trial["trial_num"]),
            "movie_name": trial["movie_name"],
            "recording_onset": start.round(3),
            "recording_offset": end.round(3),
            "filler": clean.isin(FILLERS).astype(int),
            "time_zero": "routine_start" if shown else "routine_start_assumed",
        }))
    columns = ["onset", "duration", "word", "trial_num", "movie_name",
               "recording_onset", "recording_offset", "filler", "time_zero"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=columns)


def scanner_sidecar(status, withheld):
    side = {
        "TaskName": NAT_TASK,
        "TaskDescription": (
            "Spoken free recall of a film, in the scanner. Each recall trial "
            "was cued by the film's title; the participant described the "
            "film aloud. One row per transcribed word."
        ),
        "TimeBase": (
            "onset is on the clock of this run's events.tsv (and so of the "
            "BOLD run it names): the trial's recall-routine start "
            "(recall1.started in events.tsv) plus the word's time in the "
            "trial's audio recording. Each recording spans that routine "
            "minus a constant ~0.6 s, so onsets carry up to ~0.6 s of "
            "uncertainty (start latency and end truncation cannot be told "
            "apart)."
        ),
        "TranscriptStatus": status,
        "TranscriptSource": (
            "Human-vetted per-trial word tables in this directory "
            "(recording_mic_*_word_timestamps.csv, in recording order = "
            "trial order), read unchanged."
        ),
        "SourceAudio": (
            "Raw audio remains in the private mmmsourcedata tree "
            "(PII separation); it is deliberately not part of this dataset."
        ),
        "onset": {"Description": "Word onset on the run clock; n/a where "
                  "the vetted start falls outside the trial's recall routine "
                  "(a timing typo in the vetted table, kept as written in "
                  "recording_onset).", "Units": "s"},
        "duration": {"Description": "recording_offset - recording_onset; "
                     "n/a where the vetted end precedes the start or falls "
                     "outside the routine. Such a row has a timing typo in "
                     "one of its two vetted values, so its onset may be off "
                     "too.", "Units": "s"},
        "word": {"Description": "Transcribed word as vetted, punctuation "
                 "included."},
        "trial_num": {"Description": "Recall trial, as trial_num in "
                      "events.tsv."},
        "movie_name": {"Description": "Film cued on this trial, as in "
                       "events.tsv."},
        "recording_onset": {"Description": "Vetted word start, seconds "
                            "from the trial recording's start, unaltered.",
                            "Units": "s"},
        "recording_offset": {"Description": "Vetted word end, seconds from "
                             "the trial recording's start, unaltered.",
                             "Units": "s"},
        "filler": {"Description": "1 if the word is a filled pause "
                   "(um/uh/hmm/mm/mhm), else 0.",
                   "Levels": {"0": "lexical word", "1": "filled pause"}},
        "time_zero": {
            "Description": "Basis of the recording -> run-clock offset.",
            "Levels": {
                "routine_start": "the per-trial recording exists and covers "
                                 "every word: zero is the routine start",
                "routine_start_assumed": "the per-trial recording is missing "
                                         "or shorter than the transcript (the "
                                         "words came from a session-long "
                                         "backup recording); zero is assumed "
                                         "to be the routine start, unverified",
            },
        },
    }
    if withheld:
        side["WithheldTrials"] = {
            "trial_num": sorted(withheld),
            "Reason": ("Skipped on screen in this run because the films were "
                       "recalled in an earlier, crashed run; that speech is "
                       "not on this run's clock and is not included."),
        }
    return side


def nat_sessions(sub_num):
    """Session numbers with a NATretrieval events file."""
    sub = bids_sub(sub_num)
    found = glob.glob(f"{BIDS_ROOT}/{sub}/ses-*/func/"
                      f"{sub}_ses-*_task-{NAT_TASK}*_events.tsv")
    return sorted({int(re.search(r"ses-(\d+)", f).group(1)) for f in found})


def convert_nat_session(sub_num, ses_num, dry_run=False, status="corrected"):
    """One NATretrieval session -> its run's _beh.tsv + sidecar. Returns path."""
    sub, ses = bids_sub(sub_num), f"ses-{ses_num:02d}"
    events_path, run = nat_target_run(sub_num, ses_num, sub, ses)
    events = pd.read_csv(events_path, sep="\t")
    trials = events[events["trial_type"] == "recall"].reset_index(drop=True)
    found = nat_transcripts(sub, ses)
    transcripts, blanks = [], 0
    for stamp, path in found:
        words, n_blank = read_vetted_transcript(path)
        blanks += n_blank
        transcripts.append((words, recording_seconds(sub, ses, stamp)))
    withheld = WITHHELD_TRIALS.get((sub_num, ses_num), set())
    try:
        table = scanner_events(trials, transcripts, withheld)
    except ValueError as e:
        raise ValueError(f"{sub} {ses}: {e}") from None

    run_part = f"_run-{run}" if run else ""
    tsv = f"{BIDS_ROOT}/{sub}/{ses}/beh/{sub}_{ses}_task-{NAT_TASK}{run_part}_beh.tsv"
    write_beh_tsv(table, tsv, dry_run=dry_run)
    write_json_sidecar(scanner_sidecar(status, withheld),
                       tsv.replace("_beh.tsv", "_beh.json"), dry_run=dry_run)
    assumed = table.loc[table["time_zero"] == "routine_start_assumed",
                        "trial_num"].unique().tolist()
    print(f"{sub} {ses}{run_part or ''}: {len(found)} transcripts, "
          f"{table['trial_num'].nunique()} trials, {len(table)} words; "
          f"onset n/a {table['onset'].isna().sum()}, "
          f"duration n/a {table['duration'].isna().sum()}, "
          f"blank dropped {blanks}"
          + (f", assumed zero trials {list(assumed)}" if len(assumed) else "")
          + (f", withheld {sorted(withheld)}" if withheld else "")
          + (" (dry run)" if dry_run else ""))
    return tsv


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("subjects", nargs="+", type=int,
                   help="Subject numbers, e.g. 3 4 5")
    p.add_argument("--time-base", choices=["recording", "scanner"],
                   default="recording")
    p.add_argument("--status", default=None,
                   choices=["automatic", "corrected"],
                   help="Recorded in the sidecar as TranscriptStatus "
                        "(default: automatic for recording, corrected for "
                        "scanner, whose inputs are human-vetted)")
    p.add_argument("--arm", default="standard",
                   help="Transcript arm to convert (default: standard)")
    p.add_argument("--sessions", nargs="+", type=int,
                   help="scanner only: session numbers (default: every "
                        "session with NATretrieval events)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be written without writing")
    args = p.parse_args()

    if args.time_base == "scanner":
        status = args.status or "corrected"
        for sub_num in args.subjects:
            for ses_num in args.sessions or nat_sessions(sub_num):
                convert_nat_session(sub_num, ses_num, dry_run=args.dry_run,
                                    status=status)
        return
    for sub_num in args.subjects:
        convert_final_recall(sub_num, dry_run=args.dry_run,
                             status=args.status or "automatic", arm=args.arm)


if __name__ == "__main__":
    main()
