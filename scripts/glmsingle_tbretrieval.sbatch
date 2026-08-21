#!/bin/bash
#SBATCH --job-name=glmsingle_tbret
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glmsingle_tbret_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glmsingle_tbret_%A_%a.err
#SBATCH --exclude=n0119,n0123,n0128,n0130,n0131,n0190

#==============================================================================
# GLMsingle TBretrieval — single-trial betas, cue types modelled separately
#==============================================================================
# Workbench: mmmdata-agents/docs/workbench/retrieval-modeling/
#
# Two parameterizations, both run and compared:
#   split    one fit per cue type (28 runs, 1000 conditions, 6 repeated)
#   crossed  one fit, conditions = mmmId x cueType (56 runs, 2000, 12 repeated)
#
# Sizing reference: the full-grid TB *encoding* fit peaked at 200 GB / 18.3 h on
# 8 CPUs (job 43272306) over 42 runs x ~207 vols = ~8.7k volumes. A split arm is
# 28 x 150 = 4.2k volumes (~48%); crossed is 56 x 150 = 8.4k, i.e. encoding
# scale. 220 G / 24 h covers both; revise from the pilot's actual peak.
#
# Array task mapping (submit a subset with --array):
#   1: sub-03/image   2: sub-03/word   3: sub-03/crossed
#   4: sub-04/image   5: sub-04/word   6: sub-04/crossed
#   7: sub-05/image   8: sub-05/word   9: sub-05/crossed
#
# Usage:
#   sbatch --array=1 glmsingle_tbretrieval.sbatch          # pilot: sub-03/image
#   sbatch --array=2,3 glmsingle_tbretrieval.sbatch        # rest of sub-03
#   sbatch --array=4-9 glmsingle_tbretrieval.sbatch        # sub-04 + sub-05
#==============================================================================

SUBJECTS=(sub-03 sub-03 sub-03 sub-04 sub-04 sub-04 sub-05 sub-05 sub-05)
PARAMS=(split split crossed split split crossed split split crossed)
CUES=(image word "" image word "" image word "")

IDX=$((SLURM_ARRAY_TASK_ID - 1))
SUBJECT=${SUBJECTS[$IDX]}
PARAM=${PARAMS[$IDX]}
CUE=${CUES[$IDX]}

CUE_ARG=""
if [ -n "${CUE}" ]; then
    CUE_ARG="--cue ${CUE}"
fi

echo "=========================================="
echo "GLMsingle TBretrieval — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Subject: ${SUBJECT}"
echo "Parameterization: ${PARAM} ${CUE}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"

python ${REPO_ROOT}/scripts/glmsingle_tbretrieval.py \
    --subject ${SUBJECT} \
    --parameterization ${PARAM} \
    ${CUE_ARG}

EXIT_STATUS=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "Peak RSS / elapsed: run 'seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}'"
echo "=========================================="

exit ${EXIT_STATUS}
