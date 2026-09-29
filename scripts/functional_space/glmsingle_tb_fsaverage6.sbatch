#!/bin/bash
#SBATCH --job-name=glms_tb_fs6
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glms_tb_fs6_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glms_tb_fs6_%A_%a.err

#==============================================================================
# GLMsingle TB encoding on fsaverage6 cortex (glmsingle_tb_fsaverage6.py)
#==============================================================================
# Sizing: the volume enc fits (glmsingle_tb_siloed.sbatch, 8 CPUs) peaked at
# ~200-213 GB RSS over ~20.5 h on ~1.08 M grid voxels. The cortex mask keeps
# ~75 k vertices (~7 % of that), so ~15 GB / ~1.5 h is the linear estimate;
# 64 G / 8 h leaves room for anything that does not scale with vertices.
# Measured: sub-03 (job 47483348) peaked at 22.5 GB over 1 h 40 min.
#
# Array task mapping:  1: sub-03   2: sub-04   3: sub-05
# Usage:
#   sbatch --array=1 glmsingle_tb_fsaverage6.sbatch     # first subject, check
#   sbatch --array=2-3 glmsingle_tb_fsaverage6.sbatch
#==============================================================================

SUBJECTS=(sub-03 sub-04 sub-05)
SUBJECT=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME}: ${SUBJECT}  start $(date)"

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"

python "${REPO_ROOT}/scripts/functional_space/glmsingle_tb_fsaverage6.py" --subject "${SUBJECT}"
EXIT_STATUS=$?

echo "End $(date)  exit ${EXIT_STATUS}  (seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})"
exit ${EXIT_STATUS}
