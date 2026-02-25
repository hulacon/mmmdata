---
title: cli
parent: DCM2BIDS Config
grand_parent: Code Documentation
nav_order: 52.01
---

# cli

CLI entry point for dcm2bids config generation.

Usage::

    python -m src.python.dcm2bids_config.cli --subject sub-03 --session ses-06
    python -m src.python.dcm2bids_config.cli --subject sub-03 --session all --dry-run

**Source:** `src/python/dcm2bids_config/cli.py`
{: .fs-3 .text-grey-dk-000 }

---

## Functions

### `generate_one`

Generate a dcm2bids config for a single subject/session.

```python
generate_one(subject: str, session: str, bids_root: Path, config_dir: Path, *, dry_run: bool = False, force: bool = False) -> dict
```

**Returns**

dict
    Result with keys: subject, session, status, config (if generated),
    output_path (if written), warnings.

---

