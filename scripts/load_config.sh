#!/bin/bash
#==============================================================================
# Configuration Loader for SLURM Scripts
#==============================================================================
# Source this file in SLURM batch scripts to load configuration from TOML files.
#
# Usage:
#   source /path/to/scripts/load_config.sh
#
# This sets the following environment variables:
#   BIDS_DIR - BIDS dataset directory
#   CODE_ROOT - Code repository root
#   SINGULARITY_DIR - Singularity images directory
#   VENV_DIR - Python virtual environment directory
#   OUTPUT_DIR - Derivatives output directory
#   INVENTORY_DIR - Inventory files directory
#   SLURM_EMAIL - Email for notifications
#==============================================================================

# Find config directory. Prefer REPO_ROOT if set (needed when sourced from
# SLURM spool where BASH_SOURCE resolves to the spool copy, not the original).
if [ -n "${REPO_ROOT}" ]; then
    CONFIG_DIR_INTERNAL="${REPO_ROOT}/config"
else
    SCRIPT_DIR_INTERNAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CONFIG_DIR_INTERNAL="${SCRIPT_DIR_INTERNAL}/../config"
fi

# Python script to extract config values from TOML
read -r -d '' PYTHON_EXTRACT_CONFIG <<'EOF' || true  # read -d '' exits 1 at EOF; harmless, but fatal under set -e
import importlib.util
import os
import sys
from pathlib import Path

# One loader for Python and shell: core/config.py (stdlib-only, loaded by
# file so core/__init__'s pandas import is not needed here). It applies the
# local.toml overlay, MMMDATA_BIDS_ROOT, and the dataset-relative paths.
config_dir = Path(sys.argv[1])
loader = config_dir.parent / 'src' / 'python' / 'core' / 'config.py'
spec = importlib.util.spec_from_file_location('_mmmdata_config', loader)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
try:
    config = mod.load_config(config_dir)
except FileNotFoundError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)

# Print in shell-friendly format. Path values may reference environment
# variables (e.g. work_dir = "/tmp/$USER/..."); expand them here so shell
# consumers get a concrete path.
if 'paths' in config:
    for key, value in config['paths'].items():
        print(f"{key.upper()}={os.path.expandvars(str(value))}")

if 'slurm' in config:
    for key, value in config['slurm'].items():
        print(f"SLURM_{key.upper()}={value}")
EOF

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Cannot load configuration." >&2
    return 1
fi

# Export the configuration loading function
export_config_vars() {
    # Run Python script and export results
    local config_output
    config_output=$(python3 -c "$PYTHON_EXTRACT_CONFIG" "$CONFIG_DIR_INTERNAL" 2>&1)

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to load configuration:" >&2
        echo "$config_output" >&2
        return 1
    fi

    # Export each variable
    while IFS='=' read -r key value; do
        if [ -n "$key" ] && [ -n "$value" ]; then
            export "$key"="$value"
        fi
    done <<< "$config_output"

    return 0
}

# Load and export configuration
if ! export_config_vars; then
    echo "ERROR: Configuration loading failed" >&2
    return 1
fi

# Provide user-friendly names for commonly used paths
export BIDS_DIR="${BIDS_PROJECT_DIR:-}"
export SCRIPT_DIR="${CODE_ROOT}/scripts"

echo "✓ Configuration loaded successfully"
echo "  BIDS dataset: ${BIDS_DIR}"
echo "  Code root: ${CODE_ROOT}"
echo "  Singularity images: ${SINGULARITY_DIR}"
