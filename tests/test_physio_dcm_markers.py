"""PMU marker rows are not samples.

A PhysioLog's PULS and RESP sections interleave the PMU's own beat/breath
detections among the samples, as rows with a fourth SIGNAL column
(``PULS_TRIGGER``, ``RESP_TRIGGER``). Their VALUE is a fixed marker (2048) and
their tic runs backwards into samples already logged. Until 2026-10-02 they
were parsed as samples: the marker went into the signal, and the unsorted tic
array made the nearest-neighbour resampling undefined, so the output changed
with the numpy version.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTERS = REPO_ROOT / "src" / "python" / "raw2bids_converters"
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))

from raw2bids_converters import physio_dcm  # noqa: E402

# Samples every 2 tics; a marker logged after tic 1018 carries tic 1009.
_ROWS = [f"   {1000 + 2 * i}  PULS  {2400 + i}" for i in range(20)]
_ROWS.insert(10, "   1009  PULS  2048  PULS_TRIGGER")
PULS = "PULS\nACQ_TIME_TICS  CHANNEL  VALUE  SIGNAL\nSampleTime = 2\n" + "\n".join(_ROWS) + "\n"
ACQ = "ACQUISITION_INFO\nNumVolumes = 1\nNumSlices = 1\n   0   0   1000   1006   0\n"


def test_marker_rows_kept_apart_from_samples():
    sections, _ = physio_dcm.parse_pmu_text(PULS + ACQ)
    puls = sections["PULS"]
    assert [v for _, v in puls["channels"]["PULS"]] == [2400 + i for i in range(20)]
    assert puls["markers"] == {"PULS_TRIGGER": [1009]}


def test_resampled_signal_has_no_marker_value():
    sections, _ = physio_dcm.parse_pmu_text(PULS + ACQ)
    _, vals = physio_dcm._resample_channel(sections["PULS"]["channels"]["PULS"], 2)
    assert 2048 not in vals
    assert np.all(np.diff(vals) >= 0)  # the synthetic ramp survives intact


def test_unsorted_tics_refused():
    with pytest.raises(ValueError, match="strictly increasing"):
        physio_dcm._resample_channel([(1000, 1), (1004, 2), (1002, 3)], 2)
