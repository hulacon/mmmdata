#!/bin/bash
#SBATCH --job-name=fs_combined
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_combined_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_combined_%j.err
# CPU only. Pieces run in parallel, one worker per CPU, BLAS pinned to one
# thread per worker (as the SRM route). Sizing (first real jobs, 16 CPUs):
# 2 blocks (5 weight vectors) ~4 min / 12 G peak; 3 blocks (15) ~12 min / 17 G.

#==============================================================================
# functional-space combined model — one partition job
#==============================================================================
# Driver: scripts/functional_space/combined.py. Reads the saved Grams of the
# CHA route (per target), the stimulus route and, where the target shares
# films, the response route; tunes (w, lam) on the template subjects' tuning
# films (a fit diagnostic); computes no score.
#
#   python scripts/functional_space/combined.py plan --pct 0 --draw 0 --target 03
#   sbatch --export=ALL,PCT=0,DRAW=0,TARGET=03 combined.sbatch
#   (optional: SCENARIO=secondary, SPACE=vgg19)
#
# Outputs: <derivatives>/functional_space/routes/combined-<space>/<scenario>/
# pct-<pct>/draw-<draw>/target-<sub>/ (tuning.tsv, cross_sub-*.npz, combined.json).
#==============================================================================

set -euo pipefail

: "${PCT:?set PCT}" "${DRAW:?set DRAW}" "${TARGET:?set TARGET}"
SCENARIO="${SCENARIO:-primary}"
SPACE="${SPACE:-ebind}"
REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
PY="/gpfs/projects/hulacon/shared/envs/functional-space/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Job ${SLURM_JOB_ID} combined-${SPACE} ${SCENARIO} ${PCT}% draw ${DRAW} target sub-${TARGET} on ${SLURMD_NODENAME}; start $(date)"
/usr/bin/time -v "${PY}" "${REPO_ROOT}/scripts/functional_space/combined.py" fit --space "${SPACE}" \
    --scenario "${SCENARIO}" --pct "${PCT}" --draw "${DRAW}" --target "${TARGET}" --n-jobs "${SLURM_CPUS_PER_TASK}"
echo "End $(date)"
