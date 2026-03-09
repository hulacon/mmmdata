#!/bin/bash
#==============================================================================
# Rebuild scans.tsv files and manifest database after new data arrives.
#
# Run after dcm2bids or behavioral conversion to keep the inventory current.
#
# Usage:
#   ./rebuild_inventory.sh                     # full rebuild
#   ./rebuild_inventory.sh --fast              # skip NIfTI header reading
#   ./rebuild_inventory.sh --subjects sub-03   # specific subject(s) only
#==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
PYTHON="${REPO_ROOT}/.venv/bin/python3"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python venv not found at ${PYTHON}" >&2
    exit 1
fi

# Parse flags
FAST=""
SUBJECT_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast)
            FAST="--skip-nifti"
            shift
            ;;
        --subjects)
            shift
            SUBJECT_ARGS="--subjects $1"
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--fast] [--subjects SUB]" >&2
            exit 1
            ;;
    esac
done

echo "=== Rebuilding scans.tsv files ==="
$PYTHON "${SCRIPT_DIR}/build_scans_tsv.py" $SUBJECT_ARGS
echo ""

echo "=== Rebuilding manifest database ==="
$PYTHON "${SCRIPT_DIR}/build_manifest.py" $FAST
echo ""

echo "=== Running validation ==="
cd "$REPO_ROOT"
PYTHONPATH=src/python $PYTHON -m validation.run
echo ""

echo "=== Inventory rebuild complete ==="
