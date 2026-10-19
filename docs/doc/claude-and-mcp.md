---
title: Claude and the MCP Server
parent: Code & Tools
nav_order: 40
---

# Using Claude with MMMData (MCP server)

The [mmmdata](https://github.com/hulacon/mmmdata) code repository includes an
[MCP](https://modelcontextprotocol.io) server that lets an AI agent, such as
Claude Code or Claude Science, query a copy of the dataset: BIDS contents, the
data catalog, QC metrics, stimulus features and plots. This page covers
running it against your own copy.

> **Copies shared with collaborators** have defaced anatomical images and
> contain no recordings of participants' voices. A future public release will
> go through further de-identification. The server works the same on any copy.

## What the server is

- `bin/mmmdata-mcp` starts the server (`src/python/mmmdata_mcp`). It speaks
  MCP over stdio, so the client launches it as a local command.
- Every path comes from `config/base.toml`, overlaid by `config/local.toml`.
  Nothing in the code names a host, so pointing `local.toml` at your copy is
  the only change needed.
- Tools return JSON. Subject and session arguments take bare labels (`"##"`)
  unless a tool's description says otherwise.

## 1. Get the code and an interpreter

The server needs Python **3.11 or newer** with `mcp>=2.0` plus mmmdata's
scientific stack. The list below is what the repository's CI installs to run
the server:

```bash
git clone https://github.com/hulacon/mmmdata.git /path/to/mmmdata-code
python3 -m venv /path/to/mmmdata-mcp-env
/path/to/mmmdata-mcp-env/bin/pip install \
    pandas numpy pybids duckdb matplotlib seaborn nibabel nilearn \
    "mcp>=2.0" plotly pyarrow
```

No package install of mmmdata itself is needed: the launcher puts
`src/python` on the path. `MMMDATA_MCP_PYTHON` tells the launcher which
interpreter to use; without it, it uses `python3` from your `PATH`.

## 2. Point `config/local.toml` at your copy

`base.toml` names the lab's own storage, which does not exist on your
machine. Copy the example and set the roots the server reads:

```bash
cd /path/to/mmmdata-code
cp config/local.toml.example config/local.toml
```

```toml
# config/local.toml (gitignored; deep-merged over base.toml)
[paths]
bids_project_dir = "/path/to/mmmdata"
output_dir = "/path/to/mmmdata/derivatives"
inventory_dir = "/path/to/mmmdata/inventory"
```

Every root a tool uses must exist. As an alternative to editing the
checkout, set `MMMDATA_CONFIG_DIR` to a directory holding your own
`base.toml`.

Check it starts:

```bash
MMMDATA_MCP_PYTHON=/path/to/mmmdata-mcp-env/bin/python \
    /path/to/mmmdata-code/bin/mmmdata-mcp --list-toolsets
```

## 3. Toolsets: read-only by default

Tools are grouped into toolsets. The **default** set only reads the dataset,
so it is safe for anyone holding a copy. The rest are **opt-in**.

| Toolset | Default? | What it does |
|---|---|---|
| `bids` | yes | Subjects, sessions, files |
| `catalog` | yes | Read-only SQL over the data catalog (`inventory/catalog.duckdb`) and expectation checks |
| `qc` | yes | MRIQC/fMRIPrep QC metrics and recorded QC decisions |
| `stimuli` | yes | Stimulus feature store |
| `viz` | yes | Interactive plots, written as HTML under `[mcp] figures_dir` (default `output/figures` in the code checkout) |
| `qc_write` | opt-in | Writes QC dashboards and decisions **into the dataset tree** |
| `conversion` | opt-in | BIDS conversion; **writes to the dataset** |
| `sourcedata` | opt-in | Inspects the raw source tree, which holds identifying data and is **not part of any shared copy** |
| `slurm` | opt-in | Job status on the lab's cluster |

The opt-in sets either modify the dataset, need data a shared copy does not
include, or need the lab's cluster, which is why they are off unless asked
for. Selection, highest precedence first: `--toolsets` on the command line
(comma-separated, or `all`), then `[mcp] toolsets` in the config, then the
default. `--list-toolsets` shows which are enabled and their tools.

## 4. Attach it to Claude Code

Put a `.mcp.json` in the directory you start Claude Code from:

```json
{
  "mcpServers": {
    "mmmdata": {
      "command": "/path/to/mmmdata-code/bin/mmmdata-mcp",
      "env": {
        "MMMDATA_MCP_PYTHON": "/path/to/mmmdata-mcp-env/bin/python"
      }
    }
  }
}
```

Add `"args": ["--toolsets", "bids,catalog"]` to narrow the set. Start Claude
Code in that directory, approve the project server when asked, and check
`/mcp` shows it connected. Restart the session after changing `.mcp.json` or
`local.toml`: the server reads its config once at start.

## 5. Attach it to Claude Science

Add it as a **local-command connector** (Settings > Connectors > Add
connector), with `/path/to/mmmdata-code/bin/mmmdata-mcp` as the command and
`MMMDATA_MCP_PYTHON` set to your interpreter (or use `/usr/bin/env` as the
command with `MMMDATA_MCP_PYTHON=/path/to/mmmdata-mcp-env/bin/python
/path/to/mmmdata-code/bin/mmmdata-mcp` as its arguments). Local-command
connectors run inside Claude Science's sandbox, which has **no network by
default**, and your dataset copy must be readable from inside it. We have
not yet tested this route ourselves.

## If a tool returns `dataset_available: false`

The configured root does not exist on this machine. It is not a sign that
the data is empty. The error names the missing path and the config key it
came from (for example `[paths] bids_project_dir`).

1. Set that key in `config/local.toml` to the matching directory of your
   copy (step 2).
2. Check the directory exists and is readable by the user running the
   client.
3. Restart the Claude session so the server reloads its config.

Of the default toolsets, `catalog` reads `inventory_dir` and the others read
`bids_project_dir` (with QC outputs under `output_dir`). A partial copy can
therefore make one toolset unavailable while the rest work.

The `catalog` tools also need `inventory/catalog.duckdb` itself. It is built
from the dataset by the lab's catalog engine and may not be in your copy;
without it, the catalog calls fail.
