"""Contract tests for the MMMData MCP server (src/python/mmmdata_mcp).

Pins the advertised tool names, the toolset partition, and the call path
through an in-process MCP client. Runs anywhere: nothing here reads the
dataset.
"""

import json

import anyio
import mcp
import pytest

from mmmdata_mcp import DEFAULT_TOOLSETS, TOOLSETS, build_server
from mmmdata_mcp import server as server_mod
from mmmdata_mcp.paths import DatasetUnavailable

# Renaming or dropping a tool breaks every attached harness -- change
# this list deliberately, and update any agent definitions that name it.
EXPECTED_TOOL_NAMES = {
    "aggregate_iqms",
    "describe_features",
    "detect_iqm_outliers",
    "generate_dcm2bids_config",
    "generate_qc_dashboard",
    "get_file_expectations",
    "get_image_features",
    "get_iqm_metrics",
    "get_motion_summary",
    "get_movie_features",
    "get_movie_summary",
    "get_processing_status",
    "get_run_details",
    "get_session_summary",
    "get_subject_files",
    "job_info",
    "list_jobs",
    "list_qc_reports",
    "list_stimulus_sets",
    "list_subjects",
    "load_qc_decisions",
    "plot_accuracy_by_condition",
    "plot_bold_timeseries",
    "plot_brain_map",
    "plot_carpet",
    "plot_confidence_accuracy",
    "plot_dprime_curve",
    "plot_embedding_scatter",
    "plot_feature_distribution",
    "plot_fin_comparison",
    "plot_image_features",
    "plot_learning_curve",
    "plot_motion_qc",
    "plot_movie_features",
    "plot_rt_distribution",
    "plot_stimulus_similarity",
    "plot_subsequent_memory",
    "plot_timeline_responses",
    "query_catalog",
    "run_behavioral_conversion",
    "save_qc_decision",
    "scan_sourcedata",
    "search_stimuli",
    "submit_dcm2bids",
    "summarize_dataset",
    "validate_conversion",
    "validate_dataset",
}

#: Tools that write into the dataset, read the PII source tree, or submit
#: cluster jobs. None may be reachable without an explicit opt-in.
NOT_BY_DEFAULT = {
    "generate_qc_dashboard",
    "save_qc_decision",
    "scan_sourcedata",
    "run_behavioral_conversion",
    "validate_conversion",
    "generate_dcm2bids_config",
    "submit_dcm2bids",
    "list_jobs",
    "job_info",
}


def _list_tools(toolsets):
    async def run():
        async with mcp.Client(build_server(toolsets)) as client:
            return (await client.list_tools()).tools

    return {t.name: t for t in anyio.run(run)}


@pytest.fixture(scope="module")
def advertised_tools():
    """Every tool, as advertised by a server with all toolsets on."""
    return _list_tools("all")


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------

class TestToolSurface:
    def test_server_builds(self):
        assert build_server("all").name == "mmmdata"

    def test_tool_names_pinned(self, advertised_tools):
        assert set(advertised_tools) == EXPECTED_TOOL_NAMES

    def test_every_tool_has_description_and_schema(self, advertised_tools):
        for name, tool in advertised_tools.items():
            assert tool.description, f"{name} has no description"
            assert tool.input_schema.get("type") == "object", name
            assert "properties" in tool.input_schema, name

    def test_tool_call_round_trip(self):
        async def run():
            async with mcp.Client(build_server("slurm")) as client:
                return await client.call_tool("list_jobs", {})

        result = anyio.run(run)
        assert not result.is_error
        # Every tool returns a JSON string; off-cluster list_jobs returns
        # a JSON error object rather than raising.
        assert isinstance(json.loads(result.content[0].text), (dict, list))

    def test_unknown_tool_is_rejected(self):
        async def run():
            async with mcp.Client(build_server("all")) as client:
                return await client.call_tool("no_such_tool", {})

        assert anyio.run(run).is_error


# ---------------------------------------------------------------------------
# Toolsets
# ---------------------------------------------------------------------------

class TestToolsets:
    def test_toolsets_partition_the_tools(self):
        """Each tool is in exactly one toolset, so selection is unambiguous."""
        names = [fn.__name__ for ts in TOOLSETS.values() for fn in ts.functions]
        assert len(names) == len(set(names))
        assert set(names) == EXPECTED_TOOL_NAMES

    def test_default_set_is_read_only(self):
        default = {fn.__name__ for n in DEFAULT_TOOLSETS
                   for fn in TOOLSETS[n].functions}
        assert not default & NOT_BY_DEFAULT

    def test_default_server_advertises_only_the_default_set(self, monkeypatch):
        monkeypatch.setattr(server_mod, "get_paths", _paths_with({}))
        advertised = set(_list_tools(None))
        expected = {fn.__name__ for n in DEFAULT_TOOLSETS
                    for fn in TOOLSETS[n].functions}
        assert advertised == expected

    def test_config_selects_toolsets(self, monkeypatch):
        monkeypatch.setattr(server_mod, "get_paths",
                            _paths_with({"mcp": {"toolsets": ["slurm"]}}))
        assert server_mod.resolve_toolsets() == ["slurm"]

    def test_explicit_spec_beats_config(self, monkeypatch):
        monkeypatch.setattr(server_mod, "get_paths",
                            _paths_with({"mcp": {"toolsets": ["slurm"]}}))
        assert server_mod.resolve_toolsets("bids, qc") == ["bids", "qc"]

    def test_all_selects_everything(self):
        assert server_mod.resolve_toolsets("all") == list(TOOLSETS)
        assert server_mod.resolve_toolsets(["all"]) == list(TOOLSETS)

    def test_order_follows_the_registry_not_the_spec(self):
        assert server_mod.resolve_toolsets("viz,bids") == ["bids", "viz"]

    def test_unknown_toolset_names_the_valid_ones(self):
        with pytest.raises(ValueError, match="qc_write"):
            server_mod.resolve_toolsets("bids,nope")

    def test_empty_spec_is_an_error(self):
        with pytest.raises(ValueError, match="No toolsets"):
            server_mod.resolve_toolsets(" , ")

    def test_instructions_list_only_enabled_toolsets(self, monkeypatch):
        monkeypatch.setattr(server_mod, "get_paths", _paths_with({}))
        text = server_mod._instructions(["bids"])
        assert "bids:" in text
        assert "conversion" not in text
        assert "dry_run" in server_mod._instructions(["conversion"])

    def test_list_toolsets_cli(self, capsys):
        server_mod.main(["--list-toolsets", "--toolsets", "all"])
        out = capsys.readouterr().out
        for name in TOOLSETS:
            assert name in out


def _paths_with(extra: dict):
    """A get_paths stand-in over a config holding ``extra`` plus [paths]."""
    from mmmdata_mcp.paths import ProjectPaths

    cfg = {"paths": {"bids_project_dir": "/nonexistent/bids"}, **extra}
    return lambda *a, **k: ProjectPaths(cfg)


# ---------------------------------------------------------------------------
# Absent dataset: report, never traceback
# ---------------------------------------------------------------------------

class TestGuard:
    """The MCP layer discards an exception's text. Tools must return it."""

    def test_absent_root_returns_the_key_and_the_fix(self, monkeypatch, tmp_path):
        absent = tmp_path / "not-mounted"
        from mmmdata_mcp.paths import ProjectPaths

        cfg = {"paths": {"bids_project_dir": str(absent)}}
        monkeypatch.setattr(server_mod, "get_paths",
                            lambda *a, **k: ProjectPaths(cfg))

        def tool():
            raise AssertionError("tool body ran against an absent root")

        result = json.loads(server_mod._guarded(tool, "bids_root")())
        assert result["dataset_available"] is False
        assert str(absent) in result["error"]
        assert "[paths] bids_project_dir" in result["error"]
        assert "local.toml" in result["error"]

    def test_error_raised_inside_a_tool_is_reported(self):
        def tool():
            raise DatasetUnavailable("/gpfs/absent", "[paths] output_dir")

        result = json.loads(server_mod._guarded(tool, None)())
        assert result["dataset_available"] is False
        assert "/gpfs/absent" in result["error"]

    def test_transparent_to_success(self):
        assert server_mod._guarded(lambda: '{"ok": true}', None)() == '{"ok": true}'

    def test_other_errors_stay_loud(self):
        """A missing derivative under a present root is a real fault."""
        def tool():
            raise FileNotFoundError("derivatives/fmriprep is missing")

        with pytest.raises(FileNotFoundError):
            server_mod._guarded(tool, None)()

    def test_is_a_file_not_found_error(self):
        assert issubclass(DatasetUnavailable, FileNotFoundError)

    def test_preserves_the_advertised_schema(self, advertised_tools):
        """Registration wraps every tool; schemas come from the signature,
        so the wrapper must stay signature-transparent."""
        props = advertised_tools["get_motion_summary"].input_schema["properties"]
        assert "subject" in props
        assert props["subject"].get("description")


# ---------------------------------------------------------------------------
# SLURM absence
# ---------------------------------------------------------------------------

class TestSlurmAbsence:
    def test_reports_absence_and_names_where_it_works(self, monkeypatch):
        from mmmdata_mcp.tools import slurm_tools as slurm

        monkeypatch.setattr(slurm.shutil, "which", lambda _: None)
        result = json.loads(slurm.list_jobs())
        assert result["slurm_available"] is False
        assert "squeue" in result["error"]
        assert "Talapas" in result["error"]

    def test_job_info_reports_absence_too(self, monkeypatch):
        from mmmdata_mcp.tools import slurm_tools as slurm

        monkeypatch.setattr(slurm.shutil, "which", lambda _: None)
        result = json.loads(slurm.job_info("12345"))
        assert result["slurm_available"] is False
        assert "scontrol" in result["error"]

    def test_unexecutable_binary_reports_rather_than_tracebacks(self, monkeypatch):
        from mmmdata_mcp.tools import slurm_tools as slurm

        monkeypatch.setattr(slurm.shutil, "which", lambda _: "/usr/bin/squeue")

        def boom(*args, **kwargs):
            raise OSError("Exec format error")

        monkeypatch.setattr(slurm.subprocess, "run", boom)
        assert "Exec format error" in json.loads(slurm.list_jobs())["error"]

    def test_absence_is_checked_before_the_subprocess(self, monkeypatch):
        from mmmdata_mcp.tools import slurm_tools as slurm

        monkeypatch.setattr(slurm.shutil, "which", lambda _: None)

        def fail(*args, **kwargs):
            raise AssertionError("subprocess ran with no SLURM on PATH")

        monkeypatch.setattr(slurm.subprocess, "run", fail)
        slurm.list_jobs()


# ---------------------------------------------------------------------------
# Schemas callers depend on
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_scan_sourcedata_requires_subject_id(self, advertised_tools):
        schema = advertised_tools["scan_sourcedata"].input_schema
        assert "subject_id" in schema["required"]
        assert "session_id" in schema["properties"]

    def test_run_behavioral_conversion_dry_run_default(self, advertised_tools):
        schema = advertised_tools["run_behavioral_conversion"].input_schema
        assert schema["properties"]["dry_run"]["default"] is True
        assert schema.get("required", []) == []

    def test_validate_conversion_schema(self, advertised_tools):
        props = advertised_tools["validate_conversion"].input_schema["properties"]
        assert "subjects" in props
        assert "verbose" in props

    def test_qc_enums(self, advertised_tools):
        props = advertised_tools["list_qc_reports"].input_schema["properties"]
        assert props["pipeline"]["enum"] == ["mriqc", "fmriprep", "both"]
        props = advertised_tools["get_iqm_metrics"].input_schema["properties"]
        assert set(props["modality"]["enum"]) == {"bold", "T1w", "T2w", "dwi"}
        props = advertised_tools["detect_iqm_outliers"].input_schema["properties"]
        assert set(props["scope"]["enum"]) == {"global", "within_subject"}
        props = advertised_tools["aggregate_iqms"].input_schema["properties"]
        assert set(props["group_by"]["enum"]) == {"subject", "session", "task", "global"}

    def test_get_run_details_required_fields(self, advertised_tools):
        required = advertised_tools["get_run_details"].input_schema["required"]
        assert set(required) == {"subject", "session", "task"}

    def test_all_optional_tools(self, advertised_tools):
        for name in ("get_motion_summary", "generate_qc_dashboard",
                     "load_qc_decisions", "list_stimulus_sets"):
            schema = advertised_tools[name].input_schema
            assert schema.get("required", []) == [], name

    def test_save_qc_decision_contract(self, advertised_tools):
        schema = advertised_tools["save_qc_decision"].input_schema
        assert set(schema["required"]) == {
            "subject", "session", "task", "decision", "reason"}
        # 'pending' is the only decision an automated record may carry
        assert schema["properties"]["decision"]["enum"] == [
            "keep", "exclude", "investigate", "pending"]

    def test_get_image_features_all_optional(self, advertised_tools):
        schema = advertised_tools["get_image_features"].input_schema
        assert schema.get("required", []) == []
        for prop in ("stimulus_ids", "nsd_ids", "features", "limit"):
            assert prop in schema["properties"]

    def test_get_movie_features_requires_movie_name(self, advertised_tools):
        schema = advertised_tools["get_movie_features"].input_schema
        assert "movie_name" in schema["required"]
        assert "time_start" in schema["properties"]
        assert "time_end" in schema["properties"]

    def test_search_stimuli_required_params(self, advertised_tools):
        schema = advertised_tools["search_stimuli"].input_schema
        assert {"feature", "op", "value"} <= set(schema["required"])
        assert "group" not in schema["required"]
        assert schema["properties"]["group"]["default"] == "shared1000_image"

    def test_every_stimulus_query_tool_takes_a_group(self, advertised_tools):
        for name in ("get_image_features", "get_movie_features",
                     "get_movie_summary", "search_stimuli",
                     "describe_features"):
            props = advertised_tools[name].input_schema["properties"]
            assert "group" in props, f"{name} has no group parameter"
