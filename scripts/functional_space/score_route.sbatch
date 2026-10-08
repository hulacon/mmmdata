#!/bin/bash
#SBATCH --job-name=fs_score
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_score_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_score_%j.err
# CPU only. Face refits run in parallel over pieces (BLAS pinned to one thread
# per worker, as the combined model). Sizing: first real job pending.

#==============================================================================
# functional-space scoring — one partition job, or one target's MNI baseline
#==============================================================================
# Driver: scripts/functional_space/score_route.py. Needs the job's combined
# output AND the routes' Grams (faces are refitted from them), the VGG19
# stimulus route, and, where the target shares films, the SRM route.
#
#   sbatch --export=ALL,PCT=0,DRAW=0,TARGET=03 score_route.sbatch
#   sbatch --array=0-479%20 score_route.sbatch
#   (array index = `partitions.py job --index N`)
#   sbatch --export=ALL,VERB=mni,TARGET=03 score_route.sbatch
#   psytwill arm (scores/psytwill-arm/): add ARM=psytwill to --export, for score or m4
#   Secondary families + smoothed-anatomical control (after `families.py cache-tb` per subject):
#   sbatch --array=0-479%20 --export=ALL,VERB=families score_route.sbatch
#   M4 (0% jobs only; needs the job's score.json) runs on a GPU, so the
#   resources are given at submission and override the CPU defaults above:
#   sbatch --partition=gpu --gres=gpu:nvidia_a100_80gb_pcie:1 --cpus-per-task=8 \
#          --array=0-149,330-479%6 --export=ALL,VERB=m4 score_route.sbatch
#
# Outputs: <derivatives>/functional_space/scores/<scenario>/pct-<pct>/
# draw-<draw>/target-<sub>/ (m2b.parquet, m2a.tsv, score.json), or
# scores/anatomical-mni/target-<sub>/.
#==============================================================================

set -euo pipefail

REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
PY="/gpfs/projects/hulacon/shared/envs/functional-space/bin/python"
VERB="${VERB:-score}"
ARM="${ARM:-ebind}"  # psytwill: the later arm, written under scores/psytwill-arm/
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then  # array task: the job comes from the partition table
    read -r SCENARIO PCT DRAW TARGET < <("${PY}" "${REPO_ROOT}/scripts/functional_space/partitions.py" \
        job --index "${SLURM_ARRAY_TASK_ID}")
fi
: "${TARGET:?set TARGET}"
SCENARIO="${SCENARIO:-primary}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [[ "${VERB}" == "m4" ]]; then
    : "${PCT:?set PCT}" "${DRAW:?set DRAW}"
    echo "Job ${SLURM_JOB_ID} M4 (${ARM} arm) ${SCENARIO} ${PCT}% draw ${DRAW} target sub-${TARGET} on ${SLURMD_NODENAME}; start $(date)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
    /usr/bin/time -v "${PY}" "${REPO_ROOT}/scripts/functional_space/score_route.py" m4 \
        --scenario "${SCENARIO}" --pct "${PCT}" --draw "${DRAW}" --target "${TARGET}" --arm "${ARM}"
elif [[ "${VERB}" == "families" ]]; then
    : "${PCT:?set PCT}" "${DRAW:?set DRAW}"
    echo "Job ${SLURM_JOB_ID} families ${SCENARIO} ${PCT}% draw ${DRAW} target sub-${TARGET} on ${SLURMD_NODENAME}; start $(date)"
    /usr/bin/time -v "${PY}" "${REPO_ROOT}/scripts/functional_space/score_route.py" families \
        --scenario "${SCENARIO}" --pct "${PCT}" --draw "${DRAW}" --target "${TARGET}" --n-jobs "${SLURM_CPUS_PER_TASK}"
elif [[ "${VERB}" == "mni" ]]; then
    echo "Job ${SLURM_JOB_ID} MNI baseline target sub-${TARGET} on ${SLURMD_NODENAME}; start $(date)"
    /usr/bin/time -v "${PY}" "${REPO_ROOT}/scripts/functional_space/score_route.py" mni --target "${TARGET}"
else
    : "${PCT:?set PCT}" "${DRAW:?set DRAW}"
    echo "Job ${SLURM_JOB_ID} score (${ARM} arm) ${SCENARIO} ${PCT}% draw ${DRAW} target sub-${TARGET} on ${SLURMD_NODENAME}; start $(date)"
    /usr/bin/time -v "${PY}" "${REPO_ROOT}/scripts/functional_space/score_route.py" score \
        --scenario "${SCENARIO}" --pct "${PCT}" --draw "${DRAW}" --target "${TARGET}" --n-jobs "${SLURM_CPUS_PER_TASK}" \
        --arm "${ARM}"
fi
echo "End $(date)"
