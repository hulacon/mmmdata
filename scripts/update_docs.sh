#!/bin/bash
# Regenerate API documentation for all Python packages.
#
# This script discovers every package under src/python/ and produces
# Jekyll-compatible markdown in docs/doc/code/.  Old generated files
# are removed first so stale pages don't linger.
#
# The shared dataset documentation in docs/doc/shared/ (submodule from
# mmmdata-docs) is NOT touched — these two documentation streams are
# strictly isolated.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC_ROOT="$PROJECT_ROOT/src/python"
OUTPUT_DIR="$PROJECT_ROOT/docs/doc/code"

echo "Generating API documentation..."
echo "Source root: $SRC_ROOT"
echo "Output:      $OUTPUT_DIR"
echo ""

python3 "$SCRIPT_DIR/generate_docs.py" "$SRC_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --nav-order 50 \
    --clean

echo ""
echo "Documentation generation complete!"
echo "Generated files: $OUTPUT_DIR"
