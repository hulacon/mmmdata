"""scripts/alignment/accuracy.py helpers: signal-parcel rule and the shifted-mask used by the ribbon shift check."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("nibabel")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "alignment"))
import accuracy as acc  # noqa: E402


def test_signal_parcels_take_typed_schaefer_rows_over_the_threshold(tmp_path):
    t = pd.DataFrame({"beta_type": ["D", "D", "D", "B", "D"],
                      "atlas": ["Schaefer17n400"] * 4 + ["HOSPA"],
                      "parcel": [1, 2, 3, 1, 9], "name": list("abcaz"),
                      "anchor_frac_exceed": [0.5, 0.10, 0.11, 0.9, 0.9]})
    f = tmp_path / "p.tsv"
    t.to_csv(f, sep="\t", index=False)
    assert acc.signal_parcels(f, "anchor")["parcel"].tolist() == [1, 3]
    with pytest.raises(SystemExit, match="no repeat_frac_exceed"):
        acc.signal_parcels(f, "repeat")


def test_shifted_mask_moves_by_whole_voxels_and_keeps_zero_shift():
    m = np.zeros((10, 4, 4), bool)
    m[4] = True
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    assert (acc.shifted_mask(m, aff, 0, 0.0) == m).all()
    moved = acc.shifted_mask(m, aff, 0, 2.0)  # +2 mm on a 2 mm grid = one voxel up the x index
    assert moved[5].all() and not moved[4].any()
