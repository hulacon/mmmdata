"""release.media: no participant voice recordings in a shared copy."""

import sys
from pathlib import Path

import pytest

from release import media, sweep
from release.export import ExportOptions, run_export
from release.stage import Stage
from tests.release_fixtures import EPOCH, make_env, snapshot, write_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import release_export  # noqa: E402

WAV_HEAD = b"RIFF\x24\x00\x00\x00WAVEfmt "
MP4_HEAD = b"\x00\x00\x00\x18ftypM4A \x00\x00\x00\x00"
NIFTI_GZ_HEAD = b"\x1f\x8b\x08\x00" + b"\x00" * 12


def _put(root: Path, rel: str, data: bytes = b"") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


@pytest.mark.parametrize("rel", [
    "sub-01/ses-29/beh/sub-01_ses-29_recall.wav",
    "sub-01/ses-29/beh/sub-01_ses-29_recall.M4A",          # case-insensitive
    "sub-01/ses-29/beh/sub-01_ses-29_recall.wav.gz",       # under compression
    "derivatives/transcripts/sub-01/recall.mp3",
    "sourcedata/sub-01/audio/memo.mov",                    # a copied source tree
])
def test_audio_and_video_extensions_are_found(tmp_path, rel):
    _put(tmp_path, rel)
    assert [f.path for f in media.find_media(tmp_path)] == [rel]


@pytest.mark.parametrize("name,head", [("recording.bin", WAV_HEAD),
                                       ("memo", MP4_HEAD),
                                       ("clip.dat2", b"ID3\x04\x00")])
def test_renamed_recordings_are_found_by_content(tmp_path, name, head):
    _put(tmp_path, f"sub-01/misc/{name}", head)
    found = media.find_media(tmp_path)
    assert len(found) == 1 and found[0].match.startswith("content ")


def test_dataset_formats_are_not_flagged_or_opened(tmp_path, monkeypatch):
    for rel in ("sub-01/anat/sub-01_T1w.nii.gz", "sub-01/sub-01_sessions.tsv",
                "sub-01/anat/sub-01_T1w.json", "derivatives/x/fig.svg"):
        _put(tmp_path, rel, NIFTI_GZ_HEAD)
    opened = []
    monkeypatch.setattr(media, "_sniff", lambda p: opened.append(p) or None)
    assert media.find_media(tmp_path) == []
    assert opened == []


def test_top_level_stimuli_is_excluded_by_default(tmp_path):
    _put(tmp_path, "stimuli/twp1000/voice1/word.mp3")
    _put(tmp_path, "stimuli/movies/movie_files/film.mov")
    assert media.find_media(tmp_path) == []
    assert len(media.find_media(tmp_path, allowed_dirs=())) == 2


def test_only_the_top_level_stimuli_is_excluded(tmp_path):
    rel = "sub-01/ses-29/stimuli/recall.wav"
    _put(tmp_path, rel)
    assert [f.path for f in media.find_media(tmp_path)] == [rel]


def test_git_dirs_are_skipped(tmp_path):
    _put(tmp_path, ".git/objects/blob.wav")
    assert media.find_media(tmp_path) == []


def test_sweep_reports_media(tmp_path, capsys):
    cfg, work, stage, private = make_env(tmp_path)
    _put(stage, "sub-01/ses-29/beh/sub-01_ses-29_recall.wav", WAV_HEAD)
    findings = sweep.sweep(Stage(stage, cfg, writable=False), EPOCH)
    assert [f for f in findings if f.kind == "media"]


def test_media_blocks_an_export(tmp_path, capsys):
    """A voice recording in the stage is a sweep finding, so a real export
    writes nothing."""
    cfg, work, stage, private = make_env(tmp_path)
    _put(stage, "sub-01/ses-29/beh/sub-01_ses-29_recall.wav", WAV_HEAD)
    before = {p: snapshot(p) for p in (stage, private)}
    code, findings = run_export(stage, cfg, ExportOptions(
        epoch=EPOCH, offsets_out=private / "o.tsv", pronoun_report=private / "p.tsv"))
    assert code == 1
    assert any(f.kind == "media" for f in findings)
    assert {p: snapshot(p) for p in (stage, private)} == before


def test_cli_media_exit_codes(tmp_path, capsys):
    """Read-only: runs on any root, the working tree included."""
    cfg, work, stage, private = make_env(tmp_path)
    cdir = write_config(tmp_path, cfg)
    pre = ["--config-dir", str(cdir), "media"]
    assert release_export.main(pre + [str(work)]) == 0
    _put(work, "sub-02/ses-01/beh/memo.m4a")
    assert release_export.main(pre + [str(work)]) == 1
    assert "sub-02/ses-01/beh/memo.m4a" in capsys.readouterr().out
    _put(work, "sub-02/ses-01/beh/memo2.m4a")
    assert release_export.main(pre + [str(work), "--allow-dir", "sub-02"]) == 0
