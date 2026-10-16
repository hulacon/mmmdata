# MMMData

MMMData refers to the Multi-Modal Memory Dataset. The Multi-Modal Memory Dataset is a dense (many observations for a smaller number of participants) fMRI study being conducted at the University of Oregon. The goal is to better understand how the brain remembers sights and sounds and how doing so in traditional, experimental settings (unrelated, discrete series of trials) relates to doing so in more naturalistic settings (i.e., watching and recalling a video).

## Overview

This repository hosts project code and a wiki-like documentation site built with Jekyll (Just the Docs theme). Data is currently not publicly available.

- Documentation site: https://hulacon.github.io/mmmdata/
- Project code: root-level folders and files in this repo

## MCP server for agents

`bin/mmmdata-mcp` starts an [MCP](https://modelcontextprotocol.io) server
(`src/python/mmmdata_mcp`) that lets an AI agent query the dataset: BIDS
contents, the data catalog, QC metrics, stimulus features and plots. It reads
every path from `config/base.toml`, so it works against a copy of the dataset
anywhere once `config/local.toml` points at it (see `config/local.toml.example`).

- The default toolsets only read. The toolsets that write into the dataset,
  read the raw source tree or talk to SLURM are opt-in:
  `bin/mmmdata-mcp --list-toolsets` shows them, and `--toolsets` selects them.
- The interpreter needs `mcp>=2.0` plus this repo's scientific stack. Set
  `MMMDATA_MCP_PYTHON` to choose it.
- Claude Code example (`.mcp.json`):
  `{"mcpServers": {"mmmdata": {"command": "/path/to/mmmdata/bin/mmmdata-mcp"}}}`

## Contributing

- For now, contributions are limited to team members

## License

Project **code** in this repository is free software, licensed under the **GNU
General Public License v3.0 or later** — see [LICENSE](LICENSE). It comes with no
warranty; you are free to use, modify, and redistribute it under the terms of
that licence.

The licence matches [duckbrain](https://github.com/hulacon/duckbrain)'s
deliberately: code is expected to move between the two repos, and GPL code cannot
land under a permissive licence without the copyright holder dual-licensing it
first. The corollary is that nothing here can be upstreamed into Apache-2.0
nipreps or MIT nipoppy as-is.

This covers **code only**. The MMMData **dataset** is not distributed in this
repository, is not public, and its licensing is a separate question tracked with
the dataset rather than settled here.
