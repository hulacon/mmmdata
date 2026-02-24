#!/usr/bin/env python3
"""Copy voice memos and transcription CSVs to organized sourcedata.

Copies:
- Per-session .m4a voice memos for sub-04 and sub-05 to audio/ directories
- sub-03 backup free recall to ses-29/audio/
- Transcription CSVs to matching audio/ directories

Does NOT touch Final Recall files (moved manually).

Usage:
    python copy_audio_and_transcriptions.py --dry-run   # preview only
    python copy_audio_and_transcriptions.py              # actually copy
"""

import argparse
import re
import shutil
from pathlib import Path

SOURCEDATA = Path("/projects/hulacon/shared/mmmdata/sourcedata")
RAW_BEHAVIORAL = SOURCEDATA / "rawto_be_bids" / "alldata" / "free_recall_behavioral"

SESSION_OFFSET = 18  # task sess N -> BIDS ses-(N+18)


def copy_file(src: Path, dest: Path, dry_run: bool) -> str:
    """Copy a single file, returning a status string."""
    if dest.exists():
        return f"  SKIP (exists): {src.name} -> {dest}"
    if dry_run:
        return f"  COPY: {src.name}\n    -> {dest}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return f"  COPIED: {src.name} -> {dest.parent}"


def build_voice_memo_plan():
    """Build list of (src, dest) for per-session voice memos."""
    plan = []

    # sub-04 per-session memos
    voice_dir = RAW_BEHAVIORAL / "sub_04" / "Recall" / "Voice Memos"
    for m4a in sorted(voice_dir.glob("*.m4a")):
        # Extract session number from filename like Sub4Sess3.m4a
        match = re.match(r"Sub4Sess(\d+)\.m4a$", m4a.name)
        if match:
            sess = int(match.group(1))
            bids_ses = f"ses-{sess + SESSION_OFFSET:02d}"
            dest = SOURCEDATA / "sub-04" / bids_ses / "audio" / m4a.name
            plan.append((m4a, dest))
        else:
            # Anomalous files - determine session from filename
            if "Sess2" in m4a.name:
                dest = SOURCEDATA / "sub-04" / "ses-20" / "audio" / m4a.name
            elif "Sess6" in m4a.name:
                dest = SOURCEDATA / "sub-04" / "ses-24" / "audio" / m4a.name
            else:
                print(f"  WARNING: cannot map {m4a.name}, skipping")
                continue
            plan.append((m4a, dest))

    # sub-05 per-session memos
    voice_dir = RAW_BEHAVIORAL / "sub_05" / "Recall" / "Voice Memos"
    for m4a in sorted(voice_dir.glob("*.m4a")):
        match = re.match(r"Sub5Sess(\d+)\.m4a$", m4a.name)
        if match:
            sess = int(match.group(1))
            bids_ses = f"ses-{sess + SESSION_OFFSET:02d}"
            dest = SOURCEDATA / "sub-05" / bids_ses / "audio" / m4a.name
            plan.append((m4a, dest))

    # sub-03 backup free recall -> ses-29
    backup = RAW_BEHAVIORAL / "sub_03" / "Final Free Recall" / "Backup" / "Sub3Free Recall.m4a"
    if backup.exists():
        dest = SOURCEDATA / "sub-03" / "ses-29" / "audio" / backup.name
        plan.append((backup, dest))

    return plan


def build_transcription_plan():
    """Build list of (src, dest) for transcription CSVs."""
    plan = []

    # sub-03 ses_09 -> ses-27
    csv_dir = RAW_BEHAVIORAL / "sub_03" / "Transcription Results" / "ses_09" / "Edited Word CSV"
    if csv_dir.exists():
        for csv in sorted(csv_dir.glob("*.csv")):
            dest = SOURCEDATA / "sub-03" / "ses-27" / "audio" / csv.name
            plan.append((csv, dest))

    # sub-05 ses-01 -> ses-19
    csv_dir = RAW_BEHAVIORAL / "sub_05" / "Recall" / "Transcription" / "ses-01"
    if csv_dir.exists():
        for csv in sorted(csv_dir.glob("*.csv")):
            dest = SOURCEDATA / "sub-05" / "ses-19" / "audio" / csv.name
            plan.append((csv, dest))

    # sub-05 ses-02 -> ses-20
    csv_dir = RAW_BEHAVIORAL / "sub_05" / "Recall" / "Transcription" / "ses-02"
    if csv_dir.exists():
        for csv in sorted(csv_dir.glob("*.csv")):
            dest = SOURCEDATA / "sub-05" / "ses-20" / "audio" / csv.name
            plan.append((csv, dest))

    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be copied without copying")
    args = parser.parse_args()

    voice_plan = build_voice_memo_plan()
    transcription_plan = build_transcription_plan()

    print("=== Voice Memos ===")
    v_copied, v_skipped = 0, 0
    for src, dest in voice_plan:
        result = copy_file(src, dest, args.dry_run)
        print(result)
        if "SKIP" in result:
            v_skipped += 1
        else:
            v_copied += 1

    print(f"\nVoice memos: {v_copied} {'would copy' if args.dry_run else 'copied'}, "
          f"{v_skipped} already exist\n")

    print("=== Transcription CSVs ===")
    t_copied, t_skipped = 0, 0
    for src, dest in transcription_plan:
        result = copy_file(src, dest, args.dry_run)
        print(result)
        if "SKIP" in result:
            t_skipped += 1
        else:
            t_copied += 1

    print(f"\nTranscriptions: {t_copied} {'would copy' if args.dry_run else 'copied'}, "
          f"{t_skipped} already exist\n")

    total = v_copied + t_copied
    print(f"Total: {total} {'would copy' if args.dry_run else 'copied'}")


if __name__ == "__main__":
    main()
