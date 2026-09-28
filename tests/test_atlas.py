"""get_roi_index: a name identifies one parcel, or it is an error."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))

from neuroimaging.atlas import get_roi_index  # noqa: E402


@pytest.fixture
def labels():
    names = ["7Networks_LH_Vis_1", "7Networks_LH_Vis_10", "7Networks_RH_Vis_1", "7Networks_LH_Default_PFC_1"]
    return pd.DataFrame({"index": [1, 2, 3, 4], "name": names})


def test_full_name(labels):
    assert get_roi_index(labels, "7Networks_LH_Vis_1") == 1


def test_tail_is_not_a_substring(labels):
    # "LH_Vis_1" must not pick "LH_Vis_10", which a substring match returned first-come.
    assert get_roi_index(labels, "LH_Vis_1") == 1
    assert get_roi_index(labels, "lh_vis_10") == 2


def test_ambiguous_name_raises(labels):
    with pytest.raises(ValueError, match="matches 2 parcels"):
        get_roi_index(labels, "Vis_1")


def test_no_match(labels):
    assert get_roi_index(labels, "Vis_3") is None
    assert get_roi_index(labels, "Vis") is None  # a network token is not a parcel
