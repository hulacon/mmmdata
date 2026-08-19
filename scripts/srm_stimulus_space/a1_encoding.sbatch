#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=srm_a1
#SBATCH --partition=compute
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/srm_a1_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/srm_a1_%j.err

#==============================================================================
# srm-stimulus-space A1 — within-subject cross-arm encoding (all subjects)
#==============================================================================
# Pre-registered pilot; charter and log in mmmdata-agents
# docs/workbench/srm-stimulus-space/. Step 0 builds the psytwill feature
# parquets (kept if present), then fits/evaluates all three subjects with
# the pre-registered HRF lag model plus the fixed-lag sensitivity variant.
#
# Runs in the shared stimfeat env (psytwill + sklearn + pyarrow).
# Usage: sbatch scripts/srm_stimulus_space/a1_encoding.sbatch
#==============================================================================

set -euo pipefail

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}, $(date)"

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"

PYTHON=/gpfs/projects/hulacon/shared/envs/stimfeat/bin/python
SCRIPT="${REPO_ROOT}/scripts/srm_stimulus_space/a1_encoding.py"

"${PYTHON}" "${SCRIPT}" --dry-run
"${PYTHON}" "${SCRIPT}" --lag-model hrf
"${PYTHON}" "${SCRIPT}" --lag-model fixed
# EVC diagnostic grid (workbench log 2026-08-19): NAT representation x variant
"${PYTHON}" "${SCRIPT}" --nat-source glmsingle
"${PYTHON}" "${SCRIPT}" --pipeline nordic
"${PYTHON}" "${SCRIPT}" --nat-source glmsingle --pipeline nordic

echo "Done, $(date)"
