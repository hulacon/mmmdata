#!/bin/bash
#SBATCH --job-name=glmsingle_tb_siloed
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glmsingle_tb_siloed_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glmsingle_tb_siloed_%A_%a.err
#SBATCH --exclude=n0119,n0123,n0128,n0130,n0131,n0190

#==============================================================================
# GLMsingle TB — siloed arms (enc, ret-image, ret-word fit independently)
#==============================================================================
# Workbench: mmmdata-agents/docs/workbench/retrieval-modeling/ (3x2 redesign,
# 2026-08-26). Pooled arms run separately: glmsingle_tb_pooled.sbatch.
#
# Sizing reference: the 24.1.1 TB encoding fit peaked at 200 GB / 18.3 h on
# 8 CPUs (job 43272306) over 42 runs x ~207 vols = ~8.7k volumes. The enc arm
# is that scale; a ret arm is 28 x 150 = 4.2k volumes (~48%). 220 G / 24 h
# covers both; revise from the pilot's actual peak via seff.
#
# Array task mapping (submit a subset with --array):
#   1: sub-03/enc   2: sub-03/ret-image   3: sub-03/ret-word
#   4: sub-04/enc   5: sub-04/ret-image   6: sub-04/ret-word
#   7: sub-05/enc   8: sub-05/ret-image   9: sub-05/ret-word
#
# Usage:
#   sbatch --array=2 glmsingle_tb_siloed.sbatch      # pilot: sub-03/ret-image
#   sbatch --array=1,3 glmsingle_tb_siloed.sbatch    # rest of sub-03
#   sbatch --array=4-9 glmsingle_tb_siloed.sbatch    # sub-04 + sub-05
#==============================================================================

SUBJECTS=(sub-03 sub-03 sub-03 sub-04 sub-04 sub-04 sub-05 sub-05 sub-05)
ARMS=(enc ret-image ret-word enc ret-image ret-word enc ret-image ret-word)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
SUBJECT=${SUBJECTS[$IDX]}
ARM=${ARMS[$IDX]}

echo "=========================================="
echo "GLMsingle TB (siloed) — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Subject: ${SUBJECT}"
echo "Arm: ${ARM}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"

python ${REPO_ROOT}/scripts/glmsingle_tb.py \
    --subject ${SUBJECT} \
    --arm ${ARM}

EXIT_STATUS=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "Peak RSS / elapsed: run 'seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}'"
echo "=========================================="

exit ${EXIT_STATUS}
