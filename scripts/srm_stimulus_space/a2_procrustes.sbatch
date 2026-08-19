#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=srm_a2
#SBATCH --partition=compute
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/srm_a2_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/srm_a2_%j.err

#==============================================================================
# srm-stimulus-space A2 — across-subject Procrustes alignment (all subjects)
#==============================================================================
# Pre-registered pilot; charter and log in mmmdata-agents
# docs/workbench/srm-stimulus-space/. Runs the primary arm (TB TYPED betas,
# original pipeline) and the TB-rawTR training sensitivity arm required for
# all subjects by the beta-reliability finding. NORDIC arm is a follow-up.
#
# Runs in the shared stimfeat env (psytwill + sklearn + scipy + pyarrow).
# Usage: sbatch scripts/srm_stimulus_space/a2_procrustes.sbatch
#==============================================================================

set -euo pipefail

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}, $(date)"

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"

PYTHON=/gpfs/projects/hulacon/shared/envs/stimfeat/bin/python
SCRIPT="${REPO_ROOT}/scripts/srm_stimulus_space/a2_procrustes.py"

"${PYTHON}" "${SCRIPT}" --dry-run
"${PYTHON}" "${SCRIPT}"                        # primary: betas / original
"${PYTHON}" "${SCRIPT}" --tb-source rawtr      # sensitivity: rawTR training

echo "Done, $(date)"
