# Documentation Generation System

Scripts to auto-generate Jekyll-compatible API documentation from Python
docstrings.  The output lives in `docs/doc/code/` and is strictly isolated
from the shared dataset documentation in `docs/doc/shared/` (maintained in
the separate `mmmdata-docs` repository and pulled in as a git submodule).

## Quick Start

```bash
./scripts/update_docs.sh
```

This will:

1. Discover every Python package under `src/python/` (directories with
   `__init__.py`).
2. For each package, extract public functions, classes, and their docstrings.
3. Generate markdown files in `docs/doc/code/` with proper Jekyll front
   matter for the `just-the-docs` theme.
4. Remove stale `.md` files from the output directory before writing.

## Generated File Layout

```
docs/doc/code/
├── code_index.md                          # Level 1 – Code Documentation
├── core.md                                # Level 2 – Core package index
├── core_bids_utils.md                     # Level 3 – module page (all items inline)
├── core_config.md
├── dcm2bids_config.md                     # Level 2 – DCM2BIDS Config package
├── dcm2bids_config_cli.md                 # Level 3
├── dcm2bids_config_config_builder.md
├── ...
├── raw2bids_converters.md                 # Level 2 – Raw-to-BIDS Converters
├── raw2bids_converters_common.md          # Level 3
├── raw2bids_converters_behavioral_to_beh.md
└── ...
```

### Jekyll Navigation Hierarchy

The three levels fit within `just-the-docs`' sidebar depth limit:

| Level | Example | Front Matter |
|-------|---------|-------------|
| 1 | Code Documentation | `nav_order: 50`, `has_children: true` |
| 2 | Core | `parent: Code Documentation`, `has_children: true` |
| 3 | bids_utils | `parent: Core`, `grand_parent: Code Documentation` |

Functions and classes are rendered as sections within each module page
(level 3) rather than as separate pages, keeping the sidebar navigable.

## Manual Usage

```bash
python scripts/generate_docs.py src/python/ \
    --output-dir docs/doc/code \
    --nav-order 50 \
    --clean
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `docs/doc/code` | Where to write markdown files |
| `--nav-order` | `50` | Base sidebar order for Code Documentation |
| `--clean` | off | Delete existing `.md` in output dir first |

## What Gets Documented

- **Public functions** — top-level functions not starting with `_`, excluding
  `main()` entry points.
- **Public classes** — including dataclass fields and public methods.
- **Module docstrings** — shown at the top of each module page.

Private helpers (`_name`) and CLI entry points (`main`) are intentionally
excluded.

## Docstring Format

Functions and classes should use NumPy-style docstrings:

```python
def example_function(param1: str, param2: int = 0) -> dict:
    """
    Brief description of the function.

    Longer description if needed.

    Parameters
    ----------
    param1 : str
        Description of param1.
    param2 : int, default=0
        Description of param2.

    Returns
    -------
    dict
        Description of return value.

    Examples
    --------
    >>> result = example_function("test")
    >>> print(result)
    """
```

Supported sections: Parameters, Returns, Examples, Raises, Notes, See Also,
Attributes, Yields, Warnings.

## Relationship to Shared Docs

| Stream | Location | Source | Update Mechanism |
|--------|----------|--------|-----------------|
| Code docs | `docs/doc/code/` | Python docstrings | `./scripts/update_docs.sh` (manual) |
| Shared docs | `docs/doc/shared/` | `mmmdata-docs` repo | `update-shared-docs.yml` (automatic) |

These two streams occupy separate Jekyll navigation branches ("Code
Documentation" vs "Dataset Description") and should never overlap.

## Workflow

1. Write or update Python code with NumPy-style docstrings.
2. Run `./scripts/update_docs.sh` to regenerate the code docs.
3. Review the generated markdown.
4. Commit both code and documentation changes.
5. The Jekyll site rebuilds automatically on push to `main`.
