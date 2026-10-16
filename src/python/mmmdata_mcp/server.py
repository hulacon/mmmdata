"""Build the MMMData MCP server from the tool modules.

Tools are plain functions whose signatures carry
``Annotated[..., Field(description=...)]`` metadata, so the schema the
server advertises is derived directly from each function.

Tools are registered in **toolsets**. The default set only reads the
dataset and is what anyone with a copy of it should get; the rest write
into the dataset, read the PII source tree, or need the cluster, and
are opt-in. Selection, highest precedence first: ``--toolsets`` on the
command line, ``[mcp] toolsets`` in the config, ``DEFAULT_TOOLSETS``.

Run over stdio (the default; ``bin/mmmdata-mcp`` wraps this):

    python -m mmmdata_mcp
    python -m mmmdata_mcp --toolsets all
    python -m mmmdata_mcp --list-toolsets

Serve over HTTP for remote clients:

    python -m mmmdata_mcp --transport streamable-http --port 8788

The HTTP endpoint is http://<host>:<port>/mcp. It binds to 127.0.0.1
by default and has no auth -- for anything beyond localhost testing,
front it with an SSH tunnel rather than --host 0.0.0.0.
"""

import functools
import json
from dataclasses import dataclass
from typing import Callable, Iterable

from mcp.server import MCPServer

from .paths import DatasetUnavailable, get_paths
from .tools import (
    bids_tools,
    catalog_tools,
    conversion_tools,
    qc_tools,
    slurm_tools,
    sourcedata_tools,
    stimuli_tools,
    visualization_tools,
)


@dataclass(frozen=True)
class Toolset:
    name: str
    functions: tuple[Callable[..., str], ...]
    #: ProjectPaths root that must exist for these tools to mean anything;
    #: None for tools that do not read the dataset.
    needs: str | None
    summary: str


TOOLSETS: dict[str, Toolset] = {t.name: t for t in (
    Toolset("bids", tuple(bids_tools.TOOL_FUNCTIONS), "bids_root",
            "BIDS queries (subjects, sessions, files)"),
    Toolset("catalog", tuple(catalog_tools.TOOL_FUNCTIONS), "inventory_dir",
            "catalog SQL and expectation resolution (Contract A)"),
    Toolset("qc", tuple(qc_tools.TOOL_FUNCTIONS), "bids_root",
            "MRIQC/fMRIPrep QC reads and recorded decisions"),
    Toolset("stimuli", tuple(stimuli_tools.TOOL_FUNCTIONS), "bids_root",
            "stimulus feature store (Contract B)"),
    Toolset("viz", tuple(visualization_tools.TOOL_FUNCTIONS), "bids_root",
            "interactive plots, written to [mcp] figures_dir"),
    Toolset("qc_write", tuple(qc_tools.WRITE_TOOL_FUNCTIONS), "bids_root",
            "QC dashboard and decision writes (into the dataset tree)"),
    Toolset("sourcedata", tuple(sourcedata_tools.TOOL_FUNCTIONS),
            "sourcedata_dir", "raw source tree inspection (holds PII)"),
    Toolset("conversion", tuple(conversion_tools.TOOL_FUNCTIONS), "bids_root",
            "BIDS conversion (writes to the dataset; dry_run first)"),
    Toolset("slurm", tuple(slurm_tools.TOOL_FUNCTIONS), None,
            "SLURM job status (cluster only)"),
)}

#: Read-only toolsets: safe for anyone holding a copy of the dataset.
DEFAULT_TOOLSETS = ("bids", "catalog", "qc", "stimuli", "viz")


def resolve_toolsets(spec: str | Iterable[str] | None = None) -> list[str]:
    """Toolset names to register, in TOOLSETS order.

    ``spec`` is a comma-separated string or an iterable of names; "all"
    selects every toolset. None falls back to ``[mcp] toolsets`` in the
    config, then to DEFAULT_TOOLSETS.
    """
    if spec is None:
        try:
            spec = get_paths().raw_config.get("mcp", {}).get("toolsets")
        except FileNotFoundError:
            spec = None
    if spec is None:
        spec = DEFAULT_TOOLSETS
    if isinstance(spec, str):
        spec = [s for s in (part.strip() for part in spec.split(",")) if s]
    names = set(spec)
    if "all" in names:
        return list(TOOLSETS)
    unknown = names - set(TOOLSETS)
    if unknown:
        raise ValueError(
            f"Unknown toolset(s): {sorted(unknown)}. "
            f"Valid: {list(TOOLSETS)} or 'all'."
        )
    if not names:
        raise ValueError("No toolsets selected.")
    return [name for name in TOOLSETS if name in names]


def _guarded(fn: Callable[..., str], needs: str | None) -> Callable[..., str]:
    """Return DatasetUnavailable's message instead of raising it.

    The MCP layer replaces an uncaught exception with the fixed string
    "Error executing tool <name>" and logs the traceback server-side, so
    a raised DatasetUnavailable would reach the caller with none of its
    text. Off-cluster that is *every* dataset-backed tool. Checking the
    toolset's root up front and returning the error keeps the path, the
    config key and the fix in front of the caller.

    Deliberately narrow: only DatasetUnavailable. Any other
    FileNotFoundError (a missing derivative under a present root) is a
    real fault and stays loud.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            if needs is not None:
                get_paths().require_dir(needs)
            return fn(*args, **kwargs)
        except DatasetUnavailable as exc:
            return json.dumps({
                "error": str(exc),
                "dataset_available": False,
            }, indent=2)

    return wrapper


def _instructions(names: list[str]) -> str:
    try:
        root = str(get_paths().bids_root)
    except (FileNotFoundError, KeyError):
        root = "<unconfigured>"
    groups = "; ".join(f"{n}: {TOOLSETS[n].summary}" for n in names)
    text = (
        "Tools for MMMData, a dense within-subject fMRI memory study "
        "(about 30 sessions per participant) organized as a BIDS dataset. "
        f"Configured dataset root: {root}. Enabled toolsets -- {groups}. "
        "Subject/session arguments take bare labels unless a tool says "
        "otherwise. All results are JSON strings; a result with "
        "dataset_available=false means the configured root is absent "
        "on this host, not that the data is empty."
    )
    if "conversion" in names:
        text += " Conversion tools write to the dataset: run dry_run=true first."
    return text


def build_server(toolsets: str | Iterable[str] | None = None) -> MCPServer:
    """Register every tool function from the selected toolsets."""
    names = resolve_toolsets(toolsets)
    server = MCPServer(
        name="mmmdata",
        instructions=_instructions(names),
        version="0.2.0",
    )

    seen: set[str] = set()
    for name in names:
        toolset = TOOLSETS[name]
        for fn in toolset.functions:
            if fn.__name__ in seen:
                raise RuntimeError(f"Duplicate tool name: {fn.__name__!r}")
            seen.add(fn.__name__)
            server.add_tool(_guarded(fn, toolset.needs))

    return server


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MMMData MCP server")
    parser.add_argument(
        "--toolsets",
        help=("comma-separated toolsets, or 'all'. Default: [mcp] toolsets "
              f"in the config, else {','.join(DEFAULT_TOOLSETS)}"),
    )
    parser.add_argument(
        "--list-toolsets", action="store_true",
        help="print the toolsets and their tools, then exit",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="stdio (default) or streamable-http for remote clients",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8788, help="HTTP bind port")
    args = parser.parse_args(argv)

    if args.list_toolsets:
        selected = set(resolve_toolsets(args.toolsets))
        for name, ts in TOOLSETS.items():
            mark = "*" if name in selected else " "
            tools = ", ".join(fn.__name__ for fn in ts.functions)
            print(f"{mark} {name:<11} {ts.summary}\n      {tools}")
        return

    try:
        server = build_server(args.toolsets)
    except ValueError as exc:
        parser.error(str(exc))
    if args.transport == "streamable-http":
        server.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        server.run(transport="stdio")
