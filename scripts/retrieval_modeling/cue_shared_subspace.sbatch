#!/bin/bash
#SBATCH --job-name=tb_cue_shared
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/tb_cue_shared_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/tb_cue_shared_%A_%a.err
# Stratified reads of the TB phase-pair benchmark (cue_shared_subspace.py).
# Array task = subject: 1 sub-03, 2 sub-04, 3 sub-05.
#   sbatch --array=1-3 cue_shared_subspace.sbatch
# REPO_ROOT may be overridden (e.g. a worktree) with --export=ALL,REPO_ROOT=<path>.
SUBJECTS=(sub-03 sub-04 sub-05)
SUBJECT=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}
echo "== ${SUBJECT} on ${SLURMD_NODENAME} $(date)"
export REPO_ROOT="${REPO_ROOT:-/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata}"
echo "== code ${REPO_ROOT} $(git -C ${REPO_ROOT} rev-parse --short HEAD)"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
source /gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/scripts/load_config.sh
PY=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/.venv/bin/python
${PY} ${REPO_ROOT}/scripts/retrieval_modeling/cue_shared_subspace.py --subject ${SUBJECT} "$@"
RC=$?; echo "== exit ${RC} $(date)  (seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})"; exit ${RC}
