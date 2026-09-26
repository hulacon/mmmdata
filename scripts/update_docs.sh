#!/bin/bash
# Regenerate API documentation for all Python packages.
#
# This script discovers every package (and subpackage) under src/python/
# and produces Jekyll-compatible markdown in docs/doc/code/, plus a Tools
# section for the command-line scripts listed in TOOLS below.  Old
# generated files are removed first so stale pages don't linger.
#
# The Pages workflow runs this on every deploy, so docs/doc/code/ is
# gitignored: run it locally only to preview the site.
#
# The shared dataset documentation in docs/doc/shared/ (submodule from
# mmmdata-docs) is NOT touched — these two documentation streams are
# strictly isolated.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC_ROOT="$PROJECT_ROOT/src/python"
OUTPUT_DIR="$PROJECT_ROOT/docs/doc/code"

# Command-line tools with a page of their own (rendered from the module
# docstring). Add a script here when its docstring is a user guide.
TOOLS=(
    "$PROJECT_ROOT/scripts/mmmview.py"
)

echo "Generating API documentation..."
echo "Source root: $SRC_ROOT"
echo "Output:      $OUTPUT_DIR"
echo ""

"${PYTHON:-python3}" "$SCRIPT_DIR/generate_docs.py" "$SRC_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --nav-order 50 \
    --clean \
    --tools "${TOOLS[@]}"

echo ""
echo "Documentation generation complete!"
echo "Generated files: $OUTPUT_DIR"
