"""Synthetic BIDS tree for the release de-identification tests.

Everything here is made up: participant labels 01/02, invented dates,
fictional names ("Zorblat", "Quuxville"). Layout::

    tmp/work/      stands in for the configured BIDS root (working tree)
    tmp/source/    stands in for the configured source-data directory
    tmp/stage/     the staged copy (hardlinked or copied from work/)
    tmp/private/   where private outputs go
"""

import datetime as dt
import json
import os
import shutil
from pathlib import Path

from release.stage import ReleaseConfig

EPOCH = dt.date(1900, 1, 1)

SESSIONS_01 = (
    "session_id\tacq_time\tsession_note\n"
    "ses-01\t2031-03-04\tn/a\n"
    "ses-02\t2031-03-11\tfine\n"
    "ses-03\t2031-05-01\tn/a\n"
)
SESSIONS_02 = (
    "session_id\tacq_time\tsession_note\n"
    "ses-02\t2032-02-09T10:15:00\tn/a\n"
    "ses-01\t2032-01-10\tn/a\n"
    "ses-03\tn/a\taborted\n"
)
SCANS = (
    "filename\tacq_time\n"
    "func/sub-01_ses-01_task-rest_bold.nii.gz\t13:45:00.000000\n"
)
DEBRIEF = (
    "participant_id\tsession_id\toverall_experience\tmemory_strategies\tfavorite_least_favorite\n"
    "sub-01\tses-30\t4\tHe likes to repeat words to himself. He is sure it helped.\t"
    "Her favorite part was the films; the scans made her sleepy.\n"
    "sub-02\tses-30\t5\tShe tries to link each image to her own memories.\t"
    "\"Long\" sessions were hard for her.\n"
)
TRANSCRIPT = (
    "onset\tduration\tword\n"
    "0.5\t0.2\tI\n"
    "0.8\t0.3\twent\n"
    "1.2\t0.2\tto\n"
    "1.5\t0.5\tQuuxville\n"
    "2.1\t0.2\twith\n"
    "2.4\t0.2\tmy\n"
    "2.7\t0.3\tfriend\n"
    "3.1\t0.4\tZorblat.\n"
    "3.6\t0.3\tThen\n"
    "4.0\t0.3\twe\n"
    "4.4\t0.3\tleft\n"
)
TRANSCRIPT_REL = "sub-01/ses-29/beh/sub-01_ses-29_task-FINrecall_beh.tsv"


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def build_bids(root: Path) -> Path:
    write(root / "dataset_description.json",
          json.dumps({"Name": "synthetic", "BIDSVersion": "1.9.0"}, indent=4) + "\n")
    write(root / "README", "A synthetic dataset for tests.\n")
    write(root / "sub-01" / "sub-01_sessions.tsv", SESSIONS_01)
    write(root / "sub-02" / "sub-02_sessions.tsv", SESSIONS_02)
    write(root / "sub-01" / "ses-01" / "sub-01_ses-01_scans.tsv", SCANS)
    write(root / "phenotype" / "final_debriefing.tsv", DEBRIEF)
    write(root / "phenotype" / "final_debriefing.json",
          json.dumps({"participant_id": {"Description": "id"}}, indent=4) + "\n")
    write(root / TRANSCRIPT_REL, TRANSCRIPT)
    write(root / TRANSCRIPT_REL.replace(".tsv", ".json"),
          json.dumps({"word": {"Description": "recalled word"}}, indent=4) + "\n")
    write(root / "derivatives" / "fmriprep" / "dataset_description.json",
          json.dumps({"Name": "fMRIPrep", "BIDSVersion": "1.9.0",
                      "DatasetType": "derivative",
                      "DatasetLinks": {"raw": "../.."}}, indent=4) + "\n")
    return root


def hardlink_copy(src: Path, dst: Path) -> Path:
    for dirpath, _, files in os.walk(src):
        rel = Path(dirpath).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            os.link(Path(dirpath) / f, dst / rel / f)
    return dst


def make_env(tmp_path: Path, hardlink: bool = True):
    """Build work/ and a staged copy; return (cfg, work, stage, private)."""
    work = build_bids(tmp_path / "work")
    source = tmp_path / "source"
    source.mkdir()
    stage = tmp_path / "stage"
    if hardlink:
        hardlink_copy(work, stage)
    else:
        shutil.copytree(work, stage)
    private = tmp_path / "private"
    private.mkdir()
    cfg = ReleaseConfig(bids_root=work, source_dir=source, epoch=EPOCH)
    return cfg, work, stage, private


def write_config(tmp_path: Path, cfg: ReleaseConfig, epoch: str | None = "1900-01-01") -> Path:
    """A config dir whose base.toml points at the synthetic roots."""
    cdir = tmp_path / "config"
    cdir.mkdir(exist_ok=True)
    text = (f'[paths]\nbids_project_dir = "{cfg.bids_root}"\n'
            f'source_dir = "{cfg.source_dir}"\n')
    if epoch is not None:
        text += f'\n[release]\ndate_epoch = "{epoch}"\n'
    (cdir / "base.toml").write_text(text)
    return cdir


def snapshot(root: Path) -> dict:
    """{relative path: (bytes, inode)} for every file under root."""
    out = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = Path(dirpath) / f
            out[p.relative_to(root).as_posix()] = (p.read_bytes(), p.stat().st_ino)
    return out
