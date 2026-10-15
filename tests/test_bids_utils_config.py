"""get_subject_summary must read bids_project_dir from the [paths] table.

load_config() returns the nested TOML, so ``config['bids_project_dir']``
raised KeyError whenever bids_dir was not passed explicitly.
"""

import json

import pytest

pytest.importorskip("bids")

from core import bids_utils  # noqa: E402


def test_subject_summary_resolves_root_from_config(tmp_path, monkeypatch):
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"Name": "synthetic", "BIDSVersion": "1.8.0"}))
    anat = tmp_path / "sub-01" / "ses-01" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-01_ses-01_T1w.nii.gz").write_bytes(b"")
    monkeypatch.setattr(bids_utils, "load_config",
                        lambda: {"paths": {"bids_project_dir": str(tmp_path)}})
    df = bids_utils.get_subject_summary("01")
    assert len(df) >= 1
