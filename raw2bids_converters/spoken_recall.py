#!/usr/bin/env python3
"""Convert spoken-recall Whisper transcripts into BIDS TSVs.

One converter for all spoken free-recall material, distinguished by time
base rather than by session:

  recording (implemented) — out-of-scanner audio; word onsets are
      relative to the audio recording start. Used by ses-29 final free
      recall -> sub-XX/ses-29/beh/sub-XX_ses-29_task-FINrecall_beh.tsv
  scanner (planned) — in-scanner NATretrieval recall; the same word
      table, but onsets must be re-based to the scanner clock via
      per-run alignment offsets and split into per-run files under
      func/. Not implemented until the recording->scanner alignment
      procedure exists (see docs/project-todos.md, audio harmonization).

Input is the aud2psy `transcribe` output (standard arm) staged at
<mmmsourcedata>/derivatives/recall_transcripts/sub-XX/ses-29/standard/;
provenance is copied from its .meta.json into the sidecar. Raw audio and
uncurated transcripts stay in mmmsourcedata (PII separation) — only this
curated word table enters the BIDS tree.

Usage:
    python spoken_recall.py 03 [04 05] [--dry-run] [--status automatic]
"""

import argparse
import json
import os

import pandas as pd

from common import (
    BIDS_ROOT, bids_sub, write_beh_tsv, write_json_sidecar,
)

# Real post-migration sourcedata root (common.py's SOURCE_DIR predates the
# migration and points at the dead <bids>/sourcedata path).
MMMSOURCEDATA = "/gpfs/projects/hulacon/shared/mmmsourcedata"
TRANSCRIPTS_ROOT = f"{MMMSOURCEDATA}/derivatives/recall_transcripts"

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


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("subjects", nargs="+", type=int,
                   help="Subject numbers, e.g. 3 4 5")
    p.add_argument("--time-base", choices=["recording", "scanner"],
                   default="recording")
    p.add_argument("--status", default="automatic",
                   choices=["automatic", "corrected"],
                   help="Recorded in the sidecar as TranscriptStatus")
    p.add_argument("--arm", default="standard",
                   help="Transcript arm to convert (default: standard)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be written without writing")
    args = p.parse_args()

    if args.time_base == "scanner":
        raise NotImplementedError(
            "scanner time base (NATretrieval) needs the recording->scanner "
            "alignment procedure first; see the audio-harmonization item in "
            "docs/project-todos.md"
        )
    for sub_num in args.subjects:
        convert_final_recall(sub_num, dry_run=args.dry_run,
                             status=args.status, arm=args.arm)


if __name__ == "__main__":
    main()
