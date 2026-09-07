#!/bin/bash
#SBATCH --job-name=glmsingle_tb_pooled
#SBATCH --account=hulacon
#SBATCH --partition=memorylong
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=600G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glmsingle_tb_pooled_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glmsingle_tb_pooled_%A_%a.err

#==============================================================================
# GLMsingle TB — pooled arm (one fit: enc + ret-image + ret-word subgroups)
#==============================================================================
# Workbench: mmmdata-agents/docs/workbench/retrieval-modeling/ (3x2 redesign,
# 2026-08-26). Siloed arms run separately: glmsingle_tb_siloed.sbatch.
#
# Sizing: MEASURED on the sub-03 pilot (98 runs x 150/210 vols, 3,000 design
# columns, 8 CPUs): seff peak 411.77 GB, wall 68.8 h of the 72 h asked. The
# header is 600 G / 96 h because seff's MaxRSS undercounts the cgroup — the
# siloed enc arms measured ~190-203 G yet OOM'd under a 220 G header — so a
# ~1.1x-seff margin is not enough; ~1.45x memory and ~1.4x wall is the floor.
# Wall also carries up to ~2x node-to-node variance (siloed arms, same design,
# 5.8 vs 11.7 h). memorylong nodes carry 1-4 TB, so 600 G costs nothing extra.
#
# Array task mapping:
#   1: sub-03/pooled   2: sub-04/pooled   3: sub-05/pooled
#
# Usage:
#   sbatch --array=1 glmsingle_tb_pooled.sbatch      # pooled pilot: sub-03
#   sbatch --array=2-3 glmsingle_tb_pooled.sbatch    # sub-04, sub-05 (Gate B passed)
#==============================================================================

SUBJECTS=(sub-03 sub-04 sub-05)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
SUBJECT=${SUBJECTS[$IDX]}

echo "=========================================="
echo "GLMsingle TB (pooled) — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Subject: ${SUBJECT}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"

python ${REPO_ROOT}/scripts/glmsingle_tb.py \
    --subject ${SUBJECT} \
    --arm pooled

EXIT_STATUS=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "Peak RSS / elapsed: run 'seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}'"
echo "=========================================="

exit ${EXIT_STATUS}
