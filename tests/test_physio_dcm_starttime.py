"""physio_dcm aligns by ACQUISITION_INFO or not at all.

StartTime comes from the PMU log's volume-0 start tic. Before 2026-10-02 a log
without one was written with ``StartTime = 0.0``, a value the converter made
up. It now raises, and run_all counts that file as failed.
"""

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTERS = REPO_ROOT / "src" / "python" / "raw2bids_converters"
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))

from raw2bids_converters import physio_dcm  # noqa: E402

PULS = "PULS\nSampleTime = 2\n" + "".join(
    f"   {1000 + i}  PULS  {2000 + i}\n" for i in range(10)
)
ACQ = (
    "ACQUISITION_INFO\nNumVolumes = 2\nNumSlices = 1\n"
    "   0   0   1004   1010   0\n"
    "   1   0   1604   1610   0\n"
)


class _Element:
    def __init__(self, value):
        self.value = value


def _fake_pydicom(payload: str):
    def dcmread(path, force=False):
        return {(0x7FE1, 0x1010): _Element(payload.encode("latin-1"))}

    return types.SimpleNamespace(dcmread=dcmread)


def _dicom_dir(tmp_path):
    d = tmp_path / "Series_##_PhysioLog"
    d.mkdir()
    (d / "file.dcm").write_bytes(b"")
    return d


def test_missing_volume_zero_raises(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pydicom", _fake_pydicom(PULS))
    with pytest.raises(ValueError, match="volume-0 start tic"):
        physio_dcm.convert_file(str(_dicom_dir(tmp_path)), str(tmp_path / "out"), dry_run=True)


def test_start_time_from_volume_zero(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pydicom", _fake_pydicom(PULS + ACQ))
    out = tmp_path / "sub-##_ses-##_task-x"
    assert physio_dcm.convert_file(str(_dicom_dir(tmp_path)), str(out))
    import json

    side = json.loads(Path(f"{out}_recording-pulse_physio.json").read_text())
    assert side["StartTime"] == pytest.approx((1000 - 1004) * 2.5 / 1000)
