#!/bin/bash
#SBATCH --job-name=fs_resp
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_resp_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_resp_%j.err
# CPU only. Sizing (first real job, 50 % draw 0, 8 CPUs): 5 min, 8.8 G peak.

#==============================================================================
# functional-space response route — one partition job
#==============================================================================
# Driver: scripts/functional_space/response_route.py (measured series on the
# shared films; fits on alignment data only; computes no score). Undefined
# where the target shares no film (primary 0 %, secondary).
#
#   sbatch --export=ALL,PCT=50,DRAW=0,TARGET=03 response_route.sbatch
#   sbatch --array=0-179%20 --export=ALL,EXTRA=--save-grams response_route.sbatch
#   (array index = `partitions.py job --index N --shared`)
#
# Outputs: <derivatives>/functional_space/routes/response/<scenario>/
# pct-<pct>/draw-<draw>/target-<sub>/ (cross_sub-*.npz + response.json).
#==============================================================================

set -euo pipefail

REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
PY="/gpfs/projects/hulacon/shared/envs/functional-space/bin/python"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then  # array task: the job comes from the partition table
    read -r SCENARIO PCT DRAW TARGET < <("${PY}" "${REPO_ROOT}/scripts/functional_space/partitions.py" \
        job --index "${SLURM_ARRAY_TASK_ID}" --shared)
fi
: "${PCT:?set PCT}" "${DRAW:?set DRAW}" "${TARGET:?set TARGET}"
SCENARIO="${SCENARIO:-primary}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job ${SLURM_JOB_ID} response ${SCENARIO} ${PCT}% draw ${DRAW} target sub-${TARGET} on ${SLURMD_NODENAME}; start $(date)"
/usr/bin/time -v "${PY}" "${REPO_ROOT}/scripts/functional_space/response_route.py" fit \
    --scenario "${SCENARIO}" --pct "${PCT}" --draw "${DRAW}" --target "${TARGET}" ${EXTRA:-}
echo "End $(date)"
