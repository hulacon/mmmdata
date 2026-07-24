#!/usr/bin/env python
"""Render the QC guidance registry as a standalone Markdown document.

The registry in ``src/python/neuroimaging/qc_guidance.py`` is the single
source of truth: it drives the dashboard's tooltips and glossary, and this
script renders the same content for the docs site so the guidance can be
read, reviewed, and cited without opening a dashboard.

Regenerate after editing the registry::

    python scripts/generate_qc_guidance_doc.py
    python scripts/generate_qc_guidance_doc.py --stdout   # preview only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_PYTHON = REPO_ROOT / "src" / "python"
if str(REPO_PYTHON) not in sys.path:
    sys.path.insert(0, str(REPO_PYTHON))

from neuroimaging.qc_guidance import guidance_markdown  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "docs" / "doc" / "qc-guidance.md"

FRONT_MATTER = """---
title: QC Review Guidance
nav_order: 40
---

"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Where to write the document (default: %(default)s)",
    )
    ap.add_argument(
        "--modality", action="append", default=None,
        choices=["bold", "T1w", "T2w", "dwi"],
        help="Restrict to these modalities (repeatable). Default: all.",
    )
    ap.add_argument(
        "--no-front-matter", action="store_true",
        help="Omit the Jekyll front matter (for use outside the docs site).",
    )
    ap.add_argument(
        "--stdout", action="store_true",
        help="Print to stdout instead of writing a file.",
    )
    args = ap.parse_args()

    modalities = args.modality or ["bold", "T1w", "T2w", "dwi"]
    body = guidance_markdown(modalities=modalities)
    text = body if args.no_front_matter else FRONT_MATTER + body

    if args.stdout:
        print(text)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
