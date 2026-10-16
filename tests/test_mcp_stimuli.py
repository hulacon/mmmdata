"""Tests for the MCP stimulus tools (mmmdata_mcp.tools.stimuli_tools).

The tools answer from the Contract B feature store
(``derivatives/stimuli_features/psytwill/*.parquet``). The embedding
filter runs anywhere; the store-backed tests need the real store and
skip off-cluster.
"""

import json

import pytest

from mmmdata_mcp.tools import stimuli_tools


class TestEmbeddingFilter:
    """Embeddings are excluded by default: one EBind row is 1,024 columns."""

    @pytest.mark.parametrize("feature", [
        "clip_000", "dinov2_767", "ebind_1023", "gist_511",
        "saliency_23_23", "fasttext_000",
    ])
    def test_indexed_columns_are_embeddings(self, feature):
        assert stimuli_tools._is_embedding(feature)

    @pytest.mark.parametrize("feature", [
        "resmem_memorability", "aesthetics_score", "emonet_surprise",
        "places_airfield", "yolo_zebra", "llstat_b_mean", "caption_text",
        "beats_is_downbeat",
    ])
    def test_named_scores_are_not_embeddings(self, feature):
        assert not stimuli_tools._is_embedding(feature)


@pytest.fixture
def store_groups():
    """The Contract B groups, or skip if the store is built but empty."""
    groups = stimuli_tools._available_groups()
    if not groups:
        pytest.skip("Contract B feature store not built or empty")
    return groups


@pytest.mark.requires_dataset
class TestAgainstTheStore:
    """Smoke tests against the real aggregate tables."""

    def test_groups_cover_all_three_stimulus_sets(self, store_groups):
        for prefix in ("shared1000_", "twp1000_", "movies_"):
            assert any(g.startswith(prefix) for g in store_groups), prefix

    def test_list_stimulus_sets_reports_groups(self, store_groups):
        result = json.loads(stimuli_tools.list_stimulus_sets())
        assert set(result["groups"]) == set(store_groups)
        assert result["sets"]["shared1000"]["n_files"] > 0

    def test_image_features_excludes_embeddings_by_default(self, store_groups):
        result = json.loads(stimuli_tools.get_image_features(limit=1))
        assert result["rows"] == 1
        assert not any(stimuli_tools._is_embedding(c) for c in result["columns"])

    def test_naming_an_embedding_model_opts_in(self, store_groups):
        result = json.loads(stimuli_tools.get_image_features(models=["ebind"], limit=1))
        assert len([c for c in result["columns"] if c.startswith("ebind_")]) == 1024

    def test_nsd_id_filter_is_zero_based(self, store_groups):
        """The parameter is 0-based; the stimulus id embeds a 1-based nsdId."""
        result = json.loads(stimuli_tools.get_image_features(
            nsd_ids=[2950], models=["resmem"], limit=5))
        assert result["rows"] == 1
        assert "nsd02951" in result["data"][0]["stimulus_id"]

    def test_movie_features_carry_time(self, store_groups):
        result = json.loads(stimuli_tools.get_movie_features(
            "adventure-time", time_start=10, time_end=12, models=["resmem"]))
        assert result["rows"] > 0
        assert all(10 <= r["time"] <= 12 for r in result["data"])

    def test_unknown_movie_lists_what_exists(self, store_groups):
        result = json.loads(stimuli_tools.get_movie_features("not-a-movie"))
        assert "error" in result
        assert len(result["available_stimuli"]) == 60

    def test_transcript_group_keys_on_segments_not_time(self, store_groups):
        """A group's key columns follow its granularity; all-null ones drop."""
        result = json.loads(stimuli_tools.get_movie_features(
            "adventure-time", group="movies_audio_transcript", limit=2))
        assert "chunk_idx" in result["columns"]
        assert "onset" in result["columns"]
        assert "time" not in result["columns"]

    def test_movie_summary_is_one_row_per_film(self, store_groups):
        result = json.loads(stimuli_tools.get_movie_summary(
            models=["resmem"], stats=["mean", "sd"]))
        assert result["rows"] == 60
        assert "resmem_memorability_mean" in result["columns"]

    def test_search_returns_matches_sorted_strongest_first(self, store_groups):
        result = json.loads(stimuli_tools.search_stimuli(
            "resmem_memorability", ">", 0.9, limit=5))
        values = [r["value"] for r in result["data"]]
        assert values == sorted(values, reverse=True)

    def test_search_distinguishes_no_matches_from_no_such_feature(self, store_groups):
        empty = json.loads(stimuli_tools.search_stimuli("resmem_memorability", ">", 99.0))
        assert empty["matches"] == 0 and "error" not in empty
        missing = json.loads(stimuli_tools.search_stimuli("no_such_feature", ">", 0.0))
        assert "error" in missing

    def test_describe_features_reports_checkpoints(self, store_groups):
        result = json.loads(stimuli_tools.describe_features())
        assert result["aggregator"] == "psytwill"
        assert result["models"]["resmem"]["extractor"] == "viz2psy"
        assert result["models"]["clip"]["output_type"] == "embedding"
        assert "checkpoint" in result["models"]["caption"]

    def test_beats_column_carries_its_model_prefix(self, store_groups):
        result = json.loads(stimuli_tools.describe_features(group="movies_audio_beats"))
        assert result["models"]["beats"]["features"] == ["beats_is_downbeat"]
