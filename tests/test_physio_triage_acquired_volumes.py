"""Physio triage judges a recording against the volumes acquired, not the protocol's.

A free-recall run is stopped by hand: its protocol says ``NumVolumes = 2400``
while ~1000 are acquired. Measured against the header, a recording covering
the whole run read as TRUNCATED and was never converted (2026-10-02). These pin
the acquired count, and that ACQUISITION_INFO is parsed beyond its first
50,000 characters, where the count used to stop.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTERS = REPO_ROOT / "src" / "python" / "raw2bids_converters"
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))

from raw2bids_converters.generate_physio_triage import acquired_volumes  # noqa: E402
from raw2bids_converters.physio_dcm import parse_pmu_text  # noqa: E402

TR_TICS = 600  # 1.5 s at 2.5 ms per tic
N_ACQUIRED, N_SLICES = 50, 60


def _acq_info(protocol_volumes: int) -> str:
    lines = [
        "ACQUISITION_INFO",
        "NumSlices   = %d" % N_SLICES,
        "NumVolumes  = %d" % protocol_volumes,
        "",
        "VOLUME   SLICE   ACQ_START_TICS  ACQ_FINISH_TICS  ECHO",
    ]
    for v in range(N_ACQUIRED):
        for s in range(N_SLICES):
            t = 100000 + v * TR_TICS + s * 10
            lines.append(f"     {v}       {s}         {t}         {t + 14}     0")
    return "\n".join(lines) + "\n"


def test_acquired_volumes_ignore_the_protocol_header():
    _, acq = parse_pmu_text(_acq_info(protocol_volumes=2400))
    assert acq["num_volumes"] == 2400  # the header, kept as num_volumes_protocol
    assert acquired_volumes(acq) == N_ACQUIRED


def test_acquisition_info_is_parsed_past_fifty_thousand_chars():
    text = _acq_info(protocol_volumes=N_ACQUIRED)
    assert len(text) > 50_000  # the old fixed window
    _, acq = parse_pmu_text(text)
    assert max(acq["vol_start_tics"]) == N_ACQUIRED - 1
