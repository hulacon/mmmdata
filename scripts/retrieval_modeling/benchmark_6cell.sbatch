#!/bin/bash
#SBATCH --job-name=tb_6cell
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/tb_6cell_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/tb_6cell_%A_%a.err
# 6-cell phase-pair benchmark over the TB ROI caches (retrieval-modeling
# Settles-when 2-4). Array task = subject: 1 sub-03, 2 sub-04, 3 sub-05.
#   sbatch --array=1 benchmark_6cell.sbatch          # pilot
#   sbatch --array=1-3 benchmark_6cell.sbatch
SUBJECTS=(sub-03 sub-04 sub-05)
SUBJECT=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}
echo "== ${SUBJECT} on ${SLURMD_NODENAME} $(date)"
export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"
python ${REPO_ROOT}/scripts/retrieval_modeling/benchmark_6cell.py --subject ${SUBJECT} "$@"
RC=$?; echo "== exit ${RC} $(date)  (seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})"; exit ${RC}
