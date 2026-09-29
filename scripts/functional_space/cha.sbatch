#!/bin/bash
#SBATCH --job-name=fs_cha
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_cha_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_cha_%A_%a.err
# Sizing (sizing fit 47527203, /usr/bin/time -v, 8 CPUs): one target 15 min 22 s
# wall, 80.4 G peak RSS (ico5 profiles dominate: 18,755 x 89,820 float32 = 6.7 G
# each). 120 G / 1 h is headroom.

#==============================================================================
# functional-space CHA route — one target subject per array task
#==============================================================================
# Driver: scripts/functional_space/cha.py (rest data only; computes no score).
#
#   python scripts/functional_space/cha.py plan        # inputs + memory estimate
#   sbatch --array=1-3 cha.sbatch                      # targets 03, 04, 05
#
# Outputs: <derivatives>/functional_space/routes/cha/target-<sub>/
# (cross_sub-*.npz + cha.json). The shared functional-space env carries
# nibabel/scipy; nothing is activated, the interpreter is named absolutely.
#==============================================================================

set -euo pipefail

TARGETS=(03 04 05)
TARGET="${TARGETS[$((SLURM_ARRAY_TASK_ID - 1))]}"
REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
PY="/gpfs/projects/hulacon/shared/envs/functional-space/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} target sub-${TARGET} on ${SLURMD_NODENAME}; start $(date)"
/usr/bin/time -v "${PY}" "${REPO_ROOT}/scripts/functional_space/cha.py" fit --target "${TARGET}" ${EXTRA:-}
echo "End $(date)"
