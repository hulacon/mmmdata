#!/bin/bash
#SBATCH --job-name=fs_srm
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_srm_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_srm_%j.err
# CPU only. Pieces run in parallel, one worker per CPU, each with BLAS pinned
# to one thread (threaded BLAS is slower at piece sizes). Sizing: first real
# job pending; 48 G covers the paired shared-film rows (~9 G in the response
# route) plus per-worker piece copies.

#==============================================================================
# functional-space SRM comparator + PCA control — one partition job
#==============================================================================
# Driver: scripts/functional_space/srm_route.py (fits on alignment data only;
# computes no score). Undefined where the target shares no film (primary 0 %,
# secondary): the job exits without writing.
#
#   sbatch --export=ALL,PCT=50,DRAW=0,TARGET=03 srm_route.sbatch
#   (optional: SCENARIO=secondary, EXTRA="--k 10,20")
#
# Outputs: <derivatives>/functional_space/routes/srm/<scenario>/pct-<pct>/
# draw-<draw>/target-<sub>/ (w_/pca_sub-*_k-*.npz + srm.json).
#==============================================================================

set -euo pipefail

: "${PCT:?set PCT}" "${DRAW:?set DRAW}" "${TARGET:?set TARGET}"
SCENARIO="${SCENARIO:-primary}"
REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
PY="/gpfs/projects/hulacon/shared/envs/functional-space/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Job ${SLURM_JOB_ID} srm ${SCENARIO} ${PCT}% draw ${DRAW} target sub-${TARGET} on ${SLURMD_NODENAME}; start $(date)"
/usr/bin/time -v "${PY}" "${REPO_ROOT}/scripts/functional_space/srm_route.py" fit \
    --scenario "${SCENARIO}" --pct "${PCT}" --draw "${DRAW}" --target "${TARGET}" ${EXTRA:-}
echo "End $(date)"
