#!/bin/bash
#SBATCH --job-name=fs_within
#SBATCH --account=hulacon
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_within_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/functional_space/fs_within_%j.err

#==============================================================================
# functional-space within-subject arm — one subject's own encoder on its own held-out films
#==============================================================================
# Driver: scripts/functional_space/within_subject.py (post hoc; design record in
# mmmdata-agents docs/workbench/functional-space-followups/).
#
#   Full-pool encoder + data ceiling, one subject:
#     sbatch --export=ALL,TRAIN=pool,TARGET=03 within_subject.sbatch
#   Partition encoders at 0% (array index = `partitions.py job --index N`; 0-149 = primary 0%).
#   A 3g.40gb MIG slice is enough for 15 films and starts sooner than a full A100:
#     sbatch --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1 --array=0-149%12 \
#            --export=ALL,TRAIN=partition within_subject.sbatch
#
# Outputs: <derivatives>/functional_space/scores/within/<train>/...
#==============================================================================

set -euo pipefail

REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
PY="/gpfs/projects/hulacon/shared/envs/functional-space/bin/python"
TRAIN="${TRAIN:?set TRAIN to partition or pool}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then  # array task: the job comes from the partition table
    read -r SCENARIO PCT DRAW TARGET < <("${PY}" "${REPO_ROOT}/scripts/functional_space/partitions.py" \
        job --index "${SLURM_ARRAY_TASK_ID}")
fi
: "${TARGET:?set TARGET}"
SCENARIO="${SCENARIO:-primary}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job ${SLURM_JOB_ID} within (${TRAIN}) target sub-${TARGET} ${DRAW:+draw ${DRAW} }on ${SLURMD_NODENAME}; start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [[ "${TRAIN}" == "partition" ]]; then
    : "${PCT:?set PCT}" "${DRAW:?set DRAW}"
    /usr/bin/time -v "${PY}" "${REPO_ROOT}/scripts/functional_space/within_subject.py" score --train partition \
        --scenario "${SCENARIO}" --pct "${PCT}" --draw "${DRAW}" --target "${TARGET}"
else
    /usr/bin/time -v "${PY}" "${REPO_ROOT}/scripts/functional_space/within_subject.py" score --train pool \
        --target "${TARGET}"
fi
echo "End $(date)"
