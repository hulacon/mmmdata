# Documentation Generation System

This directory contains scripts to automatically generate Jekyll-compatible documentation from Python utility modules.

## Overview

The documentation system parses NumPy-style docstrings from Python source files and generates markdown files that integrate with the Jekyll documentation site using the `just-the-docs` theme.

## Files

- **`generate_docs.py`**: Main documentation generator script
- **`update_docs.sh`**: Convenience script to regenerate all documentation

## Quick Start

To regenerate documentation for all utility modules:

```bash
./scripts/update_docs.sh
```

This will:
1. Find all Python files in `src/python/core/` (excluding private modules)
2. Extract function definitions and docstrings
3. Generate markdown files in `docs/doc/`
4. Create index pages for navigation

## Generated Files

The script generates three types of files:

1. **Main Index** (`docs/doc/code_index.md`): Top-level page listing all modules
2. **Module Index** (e.g., `docs/doc/bids_utils_index.md`): Lists all functions in a module
3. **Function Pages** (e.g., `docs/doc/bids_utils_summarize_bids_dataset.md`): Detailed docs for each function

## Manual Usage

To generate documentation for specific files:

```bash
python scripts/generate_docs.py \
    src/python/core/bids_utils.py \
    src/python/core/another_module.py \
    --output-dir docs/doc \
    --nav-order 50
```

Options:
- `--output-dir`: Where to write markdown files (default: `docs/doc`)
- `--nav-order`: Navigation order in Jekyll sidebar (default: 50)

## Docstring Format

Functions should use NumPy-style docstrings for best results:

```python
def example_function(param1: str, param2: int = 0) -> dict:
    """
    Brief description of the function.

    Longer description if needed.

    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int, default=0
        Description of param2

    Returns
    -------
    dict
        Description of return value

    Examples
    --------
    >>> result = example_function("test")
    >>> print(result)
    """
    pass
```

Supported sections:
- **Parameters**: Function arguments
- **Returns**: Return value description
- **Examples**: Usage examples (shown in code blocks)
- **Raises**: Exceptions that may be raised
- **Notes**: Additional information

## Jekyll Integration

Generated files include proper front matter for the `just-the-docs` theme:

```yaml
---
title: Function Name
parent: Module Name
grand_parent: Code Documentation
nav_order: 51.01
---
```

This creates a three-level hierarchy:
1. Code Documentation (main index)
2. Module Name (module index)
3. Function Name (function page)

## Workflow

1. Write/update Python utility functions with good docstrings
2. Run `./scripts/update_docs.sh` to regenerate documentation
3. Review generated markdown files
4. Commit both code and documentation changes
5. Jekyll site will automatically include the new documentation

## Notes

- Only public functions (not starting with `_`) are documented
- The script preserves existing documentation files not related to code
- You can customize the navigation order by editing the `--nav-order` parameter
- Generated files can be manually edited if needed (but will be overwritten on regeneration)
