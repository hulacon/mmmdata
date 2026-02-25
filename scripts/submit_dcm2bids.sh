#!/bin/bash
#==============================================================================
# Submit dcm2bids conversion jobs to SLURM
#==============================================================================
# Usage:
#   ./submit_dcm2bids.sh sub-03 ses-06        # Single session
#   ./submit_dcm2bids.sh sub-03 all           # All sessions
#   ./submit_dcm2bids.sh sub-03 ses-04 ses-30 # Multiple sessions
#==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 2 ]; then
    echo "Usage: $0 SUBJECT SESSION [SESSION ...]"
    echo "  SUBJECT  - e.g. sub-03"
    echo "  SESSION  - e.g. ses-06, 'all', or multiple sessions"
    exit 1
fi

SUBJECT="$1"
shift

# Ensure logs directory exists
mkdir -p "${SCRIPT_DIR}/../logs"

for SESSION in "$@"; do
    echo "Submitting: ${SUBJECT} / ${SESSION}"
    sbatch --export="ALL,SUBJECT=${SUBJECT},SESSION=${SESSION},REPO_ROOT=${SCRIPT_DIR}/.." \
           --job-name="dcm2bids_${SUBJECT}_${SESSION}" \
           "${SCRIPT_DIR}/dcm2bids.sbatch"
done
