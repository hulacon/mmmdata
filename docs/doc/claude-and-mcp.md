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
joining the lab's Claude workspace, a short start for anyone new to Claude
Code, and running the server against your own copy.

> **Copies shared with collaborators** have defaced anatomical images and
> contain no recordings of participants' voices. Collaborators outside the
> University of Oregon get one through Globus; see [Access](shared/access.md).
> The server works the same on any copy.

## Joining the lab's Claude workspace

The lab's Claude use is supported by Anthropic's
[AI for Science](https://www.anthropic.com/news/ai-for-science-program) program, through a
shared claude.ai **Team workspace**. Collaborators can join it; their use of
claude.ai, Claude Code and Claude Science then draws on the lab's credits
rather than their own plan.

1. **Request an invite** from J. Benjamin Hutchinson
   ([bhutch@uoregon.edu](mailto:bhutch@uoregon.edu)), giving the email address
   you want to use. If your institution blocks outside Claude workspaces, use
   another address (a personal Gmail address works).
2. **Accept the invite**, then wait for Ben to set your usage limit. Until he
   does, the account works but cannot spend credits. Changes can take about a
   day to take effect.
3. **Sign in to every Claude tool with the workspace account**, not a personal
   one. A personal login bills your own plan.
4. **Ask Ben for the program's onboarding material** if it would help: its
   onboarding guide (what the credits cover, setting up Claude Code and Claude
   Science), its community forum and its monthly webinar.

Anything you share with the whole workspace, such as a chat Project, is
visible to every member, including members at other institutions.

## New to Claude Code?

[Claude Code](https://code.claude.com/docs/en/overview) is an AI assistant that
runs in your terminal (there are also desktop and IDE versions). It reads and
edits files in the directory you start it from and runs commands, asking your
permission first. A few things to know before the setup steps below:

- **Install and sign in** with Anthropic's
  [quickstart](https://code.claude.com/docs/en/quickstart). When it asks how to
  sign in, choose your Claude (claude.ai) account and pick the lab workspace.
  `/status` shows which account a session is using.
- **Don't set `ANTHROPIC_API_KEY`** in the shell you start Claude Code from.
  With a key set, Claude Code can bill that key's account instead of your
  workspace login.
- **Read permission prompts before approving them.** Claude asks before it
  runs a command or edits a file. The server's default toolsets only read the
  dataset (step 3), so querying your copy cannot change it.
- **Talk to it in plain language.** Once the server is attached (step 4), try
  "Which subjects and sessions are in my copy?", "What QC metrics are there
  for subject ##?" or "Plot the memorability scores of the shared images".
  Claude picks the tools; you do not need to know their names.
- **Useful commands:** `/mcp` shows whether the server is connected, `/help`
  lists the rest. Start a new session after changing the server's config.

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
