#!/bin/bash
#SBATCH --job-name=fs_pspace
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-5
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_pspace_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_pspace_%A_%a.err
# Sizing: MEASURED 2026-10-07 (array 47653457): <= 3m45s, peak RSS 37 GB
# (friends A); the friends frames table is the largest input (~3.4 GB parquet).

#==============================================================================
# functional-space psytwill arm — place films + probe clips in the release
#==============================================================================
# Driver: scripts/functional_space/psytwill_projection.py
#
#   sbatch psytwill_projection.sbatch                  # all six corpus x block tasks
#   then: python psytwill_projection.py check          # films vs their release families
#
# Task -> corpus x block: 0 movies V, 1 movies A, 2 movie10 V, 3 movie10 A,
# 4 friends V, 5 friends A (a V task also writes the VL relation's V side).
#
# Outputs: <derivatives>/functional_space/psytwill_projection/<corpus>_<B>.parquet
#==============================================================================

set -euo pipefail

REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
PY="/gpfs/projects/hulacon/shared/envs/functional-space/bin/python"
CORPORA=(movies movies movie10 movie10 friends friends)
BLOCKS=(V A V A V A)
i="${SLURM_ARRAY_TASK_ID:?run as an array}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job ${SLURM_JOB_ID} task ${i}: ${CORPORA[$i]} ${BLOCKS[$i]} on ${SLURMD_NODENAME}; start $(date)"
/usr/bin/time -v "${PY}" "${REPO_ROOT}/scripts/functional_space/psytwill_projection.py" project \
    --corpus "${CORPORA[$i]}" --block "${BLOCKS[$i]}"
echo "End $(date)"
