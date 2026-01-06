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

# Find the script directory
SCRIPT_DIR_INTERNAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find config directory (one level up from scripts, then into config)
CONFIG_DIR_INTERNAL="${SCRIPT_DIR_INTERNAL}/../config"

# Python script to extract config values from TOML
read -r -d '' PYTHON_EXTRACT_CONFIG <<'EOF'
import sys
import tomllib
from pathlib import Path

def deep_merge(base, override):
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

config_dir = Path(sys.argv[1])
base_path = config_dir / 'base.toml'
local_path = config_dir / 'local.toml'

if not base_path.exists():
    print(f"ERROR: Config file not found: {base_path}", file=sys.stderr)
    sys.exit(1)

with open(base_path, 'rb') as f:
    config = tomllib.load(f)

if local_path.exists():
    with open(local_path, 'rb') as f:
        local_config = tomllib.load(f)
        config = deep_merge(config, local_config)

# Print in shell-friendly format
if 'paths' in config:
    for key, value in config['paths'].items():
        print(f"{key.upper()}={value}")

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
