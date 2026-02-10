"""Tests for the config builder."""

import json

import pytest

from src.python.dcm2bids_config.config_builder import build_config
from src.python.dcm2bids_config.session_defs import (
    SESSION_TYPES,
    AnatDef,
    SessionDef,
    TaskDef,
)


FMAP_INFO_STANDARD = {
    "encoding": {"ap": 10, "pa": 11},
    "retrieval": {"ap": 28, "pa": 30},
}


class TestBuildConfigStructure:
    """Verify the overall shape of generated configs."""

    def test_returns_descriptions_key(self):
        config = build_config(
            "sub-03", "ses-06", SESSION_TYPES["tb_middle"], FMAP_INFO_STANDARD
        )
        assert "descriptions" in config
        assert isinstance(config["descriptions"], list)

    def test_config_is_json_serializable(self):
        config = build_config(
            "sub-03", "ses-06", SESSION_TYPES["tb_middle"], FMAP_INFO_STANDARD
        )
        # Should not raise
        json.dumps(config)


class TestBoldDescriptions:
    def test_every_bold_has_taskname(self):
        """P0 issue: every BOLD must have TaskName in sidecar_changes."""
        config = build_config(
            "sub-03", "ses-06", SESSION_TYPES["tb_middle"], FMAP_INFO_STANDARD
        )
        for desc in config["descriptions"]:
            if desc.get("suffix") == "bold":
                assert "TaskName" in desc["sidecar_changes"], (
                    f"Missing TaskName in {desc['id']}"
                )

    def test_every_sbref_has_taskname(self):
        config = build_config(
            "sub-03", "ses-04", SESSION_TYPES["tb_first"], FMAP_INFO_STANDARD
        )
        for desc in config["descriptions"]:
            if desc.get("suffix") == "sbref":
                assert "TaskName" in desc["sidecar_changes"], (
                    f"Missing TaskName in {desc['id']}"
                )

    def test_task_entity_includes_prefix(self):
        """P0 issue: task labels must use TB/NAT prefix consistently."""
        config = build_config(
            "sub-03", "ses-06", SESSION_TYPES["tb_middle"], FMAP_INFO_STANDARD
        )
        for desc in config["descriptions"]:
            if desc.get("suffix") in ("bold", "sbref"):
                entity = desc["custom_entities"]
                assert entity.startswith("task-TB") or entity.startswith("task-NAT") or entity.startswith("task-INIT"), (
                    f"Task entity {entity} missing TB/NAT prefix"
                )

    def test_tb_middle_bold_count(self):
        """tb_middle: 3 encoding + 1 math + 1 resting + 4 retrieval = 9 BOLD."""
        config = build_config(
            "sub-03", "ses-06", SESSION_TYPES["tb_middle"], FMAP_INFO_STANDARD
        )
        bolds = [d for d in config["descriptions"] if d.get("suffix") == "bold"]
        assert len(bolds) == 9

    def test_tb_first_has_sbref(self):
        """tb_first should have SBRef for every BOLD."""
        config = build_config(
            "sub-03", "ses-04", SESSION_TYPES["tb_first"], FMAP_INFO_STANDARD
        )
        bolds = [d for d in config["descriptions"] if d.get("suffix") == "bold"]
        sbrefs = [d for d in config["descriptions"] if d.get("suffix") == "sbref"]
        assert len(sbrefs) == len(bolds)

    def test_tb_middle_has_sbref(self):
        """tb_middle has SBRef for every BOLD (matching legacy configs)."""
        config = build_config(
            "sub-03", "ses-06", SESSION_TYPES["tb_middle"], FMAP_INFO_STANDARD
        )
        bolds = [d for d in config["descriptions"] if d.get("suffix") == "bold"]
        sbrefs = [d for d in config["descriptions"] if d.get("suffix") == "sbref"]
        assert len(sbrefs) == len(bolds)

    def test_naturalistic_has_sbref(self):
        config = build_config(
            "sub-03", "ses-20", SESSION_TYPES["naturalistic"]
        )
        sbrefs = [d for d in config["descriptions"] if d.get("suffix") == "sbref"]
        bolds = [d for d in config["descriptions"] if d.get("suffix") == "bold"]
        assert len(sbrefs) == len(bolds)


class TestFieldmapDescriptions:
    def test_series_number_fmaps(self):
        config = build_config(
            "sub-03", "ses-06", SESSION_TYPES["tb_middle"], FMAP_INFO_STANDARD
        )
        fmaps = [d for d in config["descriptions"] if d["datatype"] == "fmap"]
        assert len(fmaps) == 4  # 2 groups x 2 directions

    def test_fmap_has_b0_field_identifier(self):
        config = build_config(
            "sub-03", "ses-06", SESSION_TYPES["tb_middle"], FMAP_INFO_STANDARD
        )
        for desc in config["descriptions"]:
            if desc["datatype"] == "fmap":
                assert "B0FieldIdentifier" in desc["sidecar_changes"]

    def test_fmap_no_intended_for(self):
        """We use B0FieldIdentifier/Source, not IntendedFor."""
        config = build_config(
            "sub-03", "ses-06", SESSION_TYPES["tb_middle"], FMAP_INFO_STANDARD
        )
        for desc in config["descriptions"]:
            if desc["datatype"] == "fmap":
                assert "IntendedFor" not in desc["sidecar_changes"]

    def test_series_number_in_criteria(self):
        config = build_config(
            "sub-03", "ses-06", SESSION_TYPES["tb_middle"], FMAP_INFO_STANDARD
        )
        fmaps = [d for d in config["descriptions"] if d["datatype"] == "fmap"]
        series_nums = [d["criteria"]["SeriesNumber"] for d in fmaps]
        assert "10" in series_nums
        assert "11" in series_nums
        assert "28" in series_nums
        assert "30" in series_nums

    def test_series_description_fmaps(self):
        """Naturalistic sessions use SeriesDescription matching."""
        config = build_config(
            "sub-03", "ses-20", SESSION_TYPES["naturalistic"]
        )
        fmaps = [d for d in config["descriptions"] if d["datatype"] == "fmap"]
        assert len(fmaps) == 4
        descs = {d["criteria"]["SeriesDescription"] for d in fmaps}
        assert descs == {
            "se_epi_ap_encoding", "se_epi_pa_encoding",
            "se_epi_ap_retrieval", "se_epi_pa_retrieval",
        }
        # No SeriesNumber in criteria
        for f in fmaps:
            assert "SeriesNumber" not in f["criteria"]

    def test_b0_identifiers_match_sources(self):
        """B0FieldIdentifier on fmaps must match B0FieldSource on BOLDs."""
        config = build_config(
            "sub-03", "ses-06", SESSION_TYPES["tb_middle"], FMAP_INFO_STANDARD
        )
        # Collect all identifiers
        identifiers = set()
        for d in config["descriptions"]:
            if "B0FieldIdentifier" in d.get("sidecar_changes", {}):
                identifiers.add(d["sidecar_changes"]["B0FieldIdentifier"])
        # Collect all sources
        sources = set()
        for d in config["descriptions"]:
            if "B0FieldSource" in d.get("sidecar_changes", {}):
                sources.add(d["sidecar_changes"]["B0FieldSource"])
        # Every source must have a matching identifier
        assert sources <= identifiers, f"Unmatched sources: {sources - identifiers}"

    def test_missing_fmap_info_raises(self):
        with pytest.raises(ValueError, match="fmap_info missing"):
            build_config("sub-03", "ses-06", SESSION_TYPES["tb_middle"])


class TestAnatomySession:
    def test_anatomy_has_anat_descriptions(self):
        config = build_config("sub-03", "ses-01", SESSION_TYPES["anatomy"])
        datatypes = {d["datatype"] for d in config["descriptions"]}
        assert "anat" in datatypes
        assert "dwi" in datatypes

    def test_anatomy_has_init_resting(self):
        config = build_config("sub-03", "ses-01", SESSION_TYPES["anatomy"])
        bolds = [d for d in config["descriptions"] if d.get("suffix") == "bold"]
        assert len(bolds) == 1
        assert bolds[0]["sidecar_changes"]["TaskName"] == "INITresting"

    def test_anatomy_no_fmaps(self):
        config = build_config("sub-03", "ses-01", SESSION_TYPES["anatomy"])
        fmaps = [d for d in config["descriptions"] if d["datatype"] == "fmap"]
        assert len(fmaps) == 0


class TestB0Naming:
    def test_b0_id_format(self):
        """B0FieldIdentifier should follow B0map_{group}_sub{nn}_ses{nn}."""
        config = build_config(
            "sub-03", "ses-06", SESSION_TYPES["tb_middle"], FMAP_INFO_STANDARD
        )
        for d in config["descriptions"]:
            sc = d.get("sidecar_changes", {})
            for key in ("B0FieldIdentifier", "B0FieldSource"):
                if key in sc:
                    val = sc[key]
                    assert val.startswith("B0map_"), f"Bad {key}: {val}"
                    assert "sub03" in val, f"Missing subject in {key}: {val}"
                    assert "ses06" in val, f"Missing session in {key}: {val}"
