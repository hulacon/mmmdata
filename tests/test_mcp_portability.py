"""The MCP server runs against a relocated copy of the dataset.

Someone outside Talapas (a collaborator with a copy of the de-identified
export on their own cluster, or a laptop) gets a working server by
changing **config only**: a ``base.toml``/``local.toml`` naming their
roots, found through ``MMMDATA_CONFIG_DIR``. No code edits, no ``/gpfs``.

The fixture is export-shaped rather than a plain copy: session dates are
shifted to a fixed epoch and free-text cells read ``[REDACTED]``, as the
public-release transforms leave them. Each test calls at least one tool
per toolset through a real in-process MCP client.

Not covered: ``conversion`` (its tools run converter scripts that write
into the dataset and submit SLURM jobs) and ``slurm`` (no dataset; its
absence is tested in test_mcp_server.py).
"""

import json
from pathlib import Path

import anyio
import duckdb
import mcp
import pandas as pd
import pytest

from mmmdata_mcp import build_server
from mmmdata_mcp import paths as paths_mod

SUB, SES, TASK, RUN = "01", "01", "encoding", "01"
PREFIX = f"sub-{SUB}_ses-{SES}_task-{TASK}_run-{RUN}"
CONFOUNDS_HEADER = ("framewise_displacement\ttrans_x\ttrans_y\ttrans_z"
                    "\trot_x\trot_y\trot_z\n")


def _write(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _build_export(root: Path) -> dict[str, Path]:
    """A one-subject, one-run dataset shaped like the de-identified export."""
    bids = root / "mmmdata"
    source = root / "mmmsourcedata"

    _write(bids / "dataset_description.json",
           json.dumps({"Name": "MMMData export fixture", "BIDSVersion": "1.9.0"}))
    _write(bids / "participants.tsv", f"participant_id\tage\nsub-{SUB}\tn/a\n")
    # Dates shifted to a fixed epoch per participant
    _write(bids / f"sub-{SUB}" / f"sub-{SUB}_sessions.tsv",
           f"session_id\tacq_time\nses-{SES}\t1900-01-01T09:00:00\n")
    func = bids / f"sub-{SUB}" / f"ses-{SES}" / "func"
    _write(func / f"{PREFIX}_bold.nii.gz")
    _write(func / f"{PREFIX}_events.tsv",
           "onset\tduration\ttrial_type\n0.0\t3.0\timage\n")
    # Free text redacted
    _write(bids / f"sub-{SUB}" / f"ses-{SES}" / "beh"
           / f"sub-{SUB}_ses-{SES}_task-recall_beh.tsv",
           "trial\tresponse_text\n1\t[REDACTED]\n")

    fmriprep_func = bids / "derivatives" / "fmriprep" / f"sub-{SUB}" / f"ses-{SES}" / "func"
    rows = "".join(f"{0.1 + 0.01 * i}\t0.01\t0.02\t0.01\t0.001\t0.002\t0.001\n"
                   for i in range(9))
    _write(fmriprep_func / f"{PREFIX}_desc-confounds_timeseries.tsv",
           CONFOUNDS_HEADER + "n/a\t0\t0\t0\t0\t0\t0\n" + rows)

    _write(bids / "stimuli" / "shared1000" / "images" / "shared0001.png")
    tables = bids / "derivatives" / "stimuli_features" / "psytwill"
    tables.mkdir(parents=True)
    pd.DataFrame({
        "stimulus_id": ["shared0001", "shared0001"],
        "model": ["resmem", "aesthetics"],
        "feature": ["resmem_memorability", "aesthetics_score"],
        "value": [0.7, 5.1],
    }).to_parquet(tables / "shared1000_image_features.parquet")

    inventory = bids / "inventory"
    inventory.mkdir()
    con = duckdb.connect(str(inventory / "catalog.duckdb"))
    con.execute("CREATE TABLE files (sub VARCHAR, ses VARCHAR, task VARCHAR, "
                "suffix VARCHAR, dataset_relpath VARCHAR)")
    con.execute("INSERT INTO files VALUES (?, ?, ?, 'bold', '.')", [SUB, SES, TASK])
    con.close()

    _write(source / f"sub-{SUB}" / f"ses-{SES}" / "behavioral" / "log.csv", "a,b\n")
    return {"bids": bids, "source": source}


def _write_config(config_dir: Path, bids: Path, source: Path, figures: Path) -> None:
    _write(config_dir / "base.toml", f"""\
[paths]
bids_project_dir = "{bids}"
source_dir = "{source}"
output_dir = "{bids / 'derivatives'}"
inventory_dir = "{bids / 'inventory'}"

[qc]
fd_threshold = 0.5
investigate_threshold = 0.5
iqr_multiplier = 1.5

[mcp]
figures_dir = "{figures}"
""")


@pytest.fixture
def relocated(tmp_path, monkeypatch):
    """Config pointing at the export fixture, selected via the env var."""
    roots = _build_export(tmp_path / "export")
    figures = tmp_path / "figures"
    config_dir = tmp_path / "config"
    _write_config(config_dir, roots["bids"], roots["source"], figures)

    monkeypatch.setenv("MMMDATA_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(paths_mod, "_paths", None)
    return {**roots, "figures": figures}


def call(name: str, args: dict | None = None):
    """Call one tool through an all-toolsets server; return parsed JSON."""
    async def run():
        async with mcp.Client(build_server("all")) as client:
            return await client.call_tool(name, args or {})

    result = anyio.run(run)
    assert not result.is_error, result.content[0].text
    text = result.content[0].text
    assert "/gpfs/" not in text, f"{name} resolved a Talapas path: {text[:200]}"
    return json.loads(text)


class TestRelocatedByConfigOnly:
    def test_paths_resolve_to_the_copy(self, relocated):
        assert paths_mod.get_paths().bids_root == relocated["bids"]

    def test_bids(self, relocated):
        assert call("list_subjects") == [SUB]
        summary = call("summarize_dataset")
        assert {"encoding", "recall"} <= set(summary["tasks"])

    def test_catalog(self, relocated):
        result = call("query_catalog", {"sql": "SELECT COUNT(*) AS n FROM files"})
        assert result["rows"] == [{"n": 1}]

    def test_qc(self, relocated):
        result = call("get_motion_summary", {"subject": SUB})
        assert result["n_runs"] == 1

    def test_stimuli(self, relocated):
        result = call("list_stimulus_sets")
        assert result["sets"]["shared1000"]["n_files"] == 1
        assert result["groups"]["shared1000_image"]["n_stimuli"] == 1

    def test_viz_writes_to_the_configured_figures_dir(self, relocated):
        result = call("plot_motion_qc",
                      {"subject": SUB, "session": SES, "task": TASK, "run": RUN})
        out = Path(result["file_path"])
        assert out.is_file()
        assert relocated["figures"] in out.parents

    def test_qc_write_lands_inside_the_copy(self, relocated):
        result = call("save_qc_decision", {
            "subject": SUB, "session": SES, "task": TASK, "run": RUN,
            "decision": "pending", "reason": "relocation test",
        })
        assert result["saved"] is True
        assert result["automated"] is True
        decisions = relocated["bids"] / "derivatives" / "duckbrain" / "qc" / "decisions"
        assert list(decisions.rglob("*.json")), "no decision file under the copy"

    def test_sourcedata(self, relocated):
        result = call("scan_sourcedata", {"subject_id": SUB})
        assert Path(result["path"]) == relocated["source"] / f"sub-{SUB}"
        assert result["totals"] == {"behavioral": 1}


class TestAbsentCopy:
    def test_missing_root_reports_the_key(self, tmp_path, monkeypatch):
        """A config naming a root that is not there says which key to fix."""
        absent = tmp_path / "not-copied-yet"
        _write_config(tmp_path / "config", absent, absent / "src", tmp_path / "fig")
        monkeypatch.setenv("MMMDATA_CONFIG_DIR", str(tmp_path / "config"))
        monkeypatch.setattr(paths_mod, "_paths", None)

        result = call("list_subjects")
        assert result["dataset_available"] is False
        assert str(absent) in result["error"]
        assert "[paths] bids_project_dir" in result["error"]
