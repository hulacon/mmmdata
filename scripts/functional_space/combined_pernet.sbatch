#!/bin/bash
#SBATCH --job-name=fs_pernet
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_pernet_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_pernet_%A_%a.err
# CPU only. Pieces run in parallel, BLAS pinned to one thread per worker (as combined.sbatch).

#==============================================================================
# functional-space combined model tuned by segment identification, per network
#==============================================================================
# Driver: scripts/functional_space/combined_pernet.py. Reads the frozen 0 %
# combined job's Grams (CHA + stimulus-<space>) and tuning films; writes the
# combined-<space>-segid and combined-<space>-pernet route trees. No score.
#
#   sbatch --export=ALL,DRAW=0,TARGET=03 combined_pernet.sbatch
#   sbatch --export=ALL,VERB=check,DRAW=0,TARGET=03 combined_pernet.sbatch
#   sbatch --array=0-149%20 --export=ALL,SPACE=psytwill,SCENARIO=primary combined_pernet.sbatch
#   (array index -> draw = index / 3, target = 03 / 04 / 05 by index % 3; 0 % jobs only)
#   (optional: REPO_ROOT=<a worktree> to run unmerged code)
#==============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata}"
PY="/gpfs/projects/hulacon/shared/envs/functional-space/bin/python"
VERB="${VERB:-fit}"
SPACE="${SPACE:-psytwill}"
SCENARIO="${SCENARIO:-primary}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    TARGETS=(03 04 05)
    DRAW=$(( SLURM_ARRAY_TASK_ID / 3 ))
    TARGET="${TARGETS[$(( SLURM_ARRAY_TASK_ID % 3 ))]}"
fi
: "${DRAW:?set DRAW}" "${TARGET:?set TARGET}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Job ${SLURM_JOB_ID} ${VERB} combined-${SPACE}-pernet ${SCENARIO} 0% draw ${DRAW} target sub-${TARGET} on ${SLURMD_NODENAME}; repo ${REPO_ROOT}; start $(date)"
/usr/bin/time -v "${PY}" "${REPO_ROOT}/scripts/functional_space/combined_pernet.py" "${VERB}" --space "${SPACE}" \
    --scenario "${SCENARIO}" --draw "${DRAW}" --target "${TARGET}" --n-jobs "${SLURM_CPUS_PER_TASK}"
echo "End $(date)"
