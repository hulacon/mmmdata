"""mmmget.roi() (Schaefer, configured subfields) and mmmget.events() (D6)."""

import numpy as np
import pytest

import mmmget
from tests.mmmget_fixtures import SUBFIELD_TOML, make_env


class TestSchaefer:
    @pytest.fixture(autouse=True)
    def _env(self, tmp_path, monkeypatch):
        make_env(tmp_path, monkeypatch)

    def test_parcel_by_name(self):
        r = mmmget.roi("01", "schaefer100-7:LH_Vis_1")
        assert r.index == 1 and r.space == "MNI152NLin2009cAsym_res-2"
        assert r.path.name.startswith("tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_seg-7n_scale-100")
        assert int(np.asarray(r.mask().dataobj).sum()) == 16
        (rec,) = mmmget.provenance_records()
        assert rec["getter"] == "roi" and rec["standard"] is True

    def test_other_space_is_not_guessed(self):
        with pytest.raises(mmmget.SourceNotConfigured, match="only in MNI152NLin2009cAsym_res-2"):
            mmmget.roi("01", "schaefer100-7:LH_Vis_1", space="T1w")

    def test_unknown_parcel(self):
        with pytest.raises(LookupError, match="no parcel"):
            mmmget.roi("01", "schaefer100-7:LH_Nope_9")

    def test_bad_names(self):
        with pytest.raises(ValueError, match="<atlas>:<region>"):
            mmmget.roi("01", "LH_Vis_1")
        with pytest.raises(ValueError, match="not known"):
            mmmget.roi("01", "glasser:V1")


class TestSubfields:
    def test_unconfigured_source_says_what_is_missing(self, tmp_path, monkeypatch):
        make_env(tmp_path, monkeypatch)
        with pytest.raises(mmmget.SourceNotConfigured, match=r"\[mmmget.hippocampal_subfields\]"):
            mmmget.roi("01", "hipp:LH_CA1", space="T1w")

    def test_configured_source_resolves_from_catalog(self, tmp_path, monkeypatch):
        bids = make_env(tmp_path, monkeypatch, SUBFIELD_TOML)
        r = mmmget.roi("sub-01", "hipp:LH_CA1", space="T1w")
        assert r.index == 2 and r.path.is_relative_to(bids / "derivatives" / "hipposeg")
        assert int(np.asarray(r.mask().dataobj).sum()) == 1

    def test_subject_without_segmentation(self, tmp_path, monkeypatch):
        make_env(tmp_path, monkeypatch, SUBFIELD_TOML)
        with pytest.raises(mmmget.SourceNotConfigured, match="found none"):
            mmmget.roi("02", "hipp:LH_CA1", space="T1w")

    def test_space_mismatch(self, tmp_path, monkeypatch):
        make_env(tmp_path, monkeypatch, SUBFIELD_TOML)
        with pytest.raises(mmmget.SourceNotConfigured, match="no resampling"):
            mmmget.roi("01", "hipp:LH_CA1")


def test_events_raises_naming_d6(tmp_path, monkeypatch):
    make_env(tmp_path, monkeypatch)
    (run,) = mmmget.runs(subject="01", task="memory", run=1)
    with pytest.raises(mmmget.OpenDecisionError, match="open decision D6") as exc:
        mmmget.events(run, scheme="any")
    assert exc.value.decision == "D6"
    assert "_events.tsv" in str(exc.value)
