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
#   sbatch rescue_cpu.sbatch project --block V --corpus movies --manifest <fit>.json
#   sbatch rescue_cpu.sbatch read --block V --members m1,m2 --register films --out <dir>
# The verb and flags are script arguments, not --export: sbatch --export splits on
# commas, which would cut a comma-separated --members list.
#==============================================================================

set -euo pipefail

REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
PY="/gpfs/projects/hulacon/shared/envs/functional-space/bin/python"
[ $# -ge 1 ] || { echo "usage: sbatch rescue_cpu.sbatch project|read <flags>" >&2; exit 2; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONNOUSERSITE=1

echo "Job ${SLURM_JOB_ID} $* on ${SLURMD_NODENAME}; start $(date)"
/usr/bin/time -v "${PY}" "${REPO_ROOT}/scripts/braintwill_encoding/rescue.py" "$@"
echo "End $(date)"
