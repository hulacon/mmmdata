#!/bin/bash
#SBATCH --job-name=enc_rescue_cpu
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/enc_rescue_cpu_%j.out
# Sizing: a projection of the films' frames took < 30 min at 64 G (psytwill_projection.sbatch);
# read loads every cached score's per-group sums and rebuilds each feature set's key.

#==============================================================================
# rescue.py project | read (no GPU)
#==============================================================================
#   sbatch --export=ALL,VERB=project,ARGS="--block V --corpus movies --manifest <fit>.json" rescue_cpu.sbatch
#   sbatch --export=ALL,VERB=read,ARGS="--block V --members m1,m2 --register films --out <dir>" rescue_cpu.sbatch
#==============================================================================

set -euo pipefail

REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
PY="/gpfs/projects/hulacon/shared/envs/functional-space/bin/python"
: "${VERB:?set VERB (project|read)}" "${ARGS:?set ARGS}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONNOUSERSITE=1

echo "Job ${SLURM_JOB_ID} ${VERB} ${ARGS} on ${SLURMD_NODENAME}; start $(date)"
# shellcheck disable=SC2086  # ARGS is a list of flags
/usr/bin/time -v "${PY}" "${REPO_ROOT}/scripts/braintwill_encoding/rescue.py" "${VERB}" ${ARGS}
echo "End $(date)"
