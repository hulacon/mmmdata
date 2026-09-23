#!/bin/bash
#SBATCH --job-name=nr_fit
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/nr_fit_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/nr_fit_%A_%a.err

#==============================================================================
# Neural-rotation pilot: one operator-fit cell per array task
#==============================================================================
# Workbench: mmmdata-agents/docs/workbench/neural-rotation-pilot/.
# Each task is one line of a cells manifest (make_cells.py writes it):
#   subject <TAB> rung <TAB> roi <TAB> pair
# and runs fit_pair.py on it with the cluster null (--n-perm 1000). Extra
# arguments after the manifest are passed to fit_pair.py.
#
# Sizing: the largest cells (Posterior ~76k voxels, PrfUnionThr2p5 ~58k) hold
# two float64 (trials x voxels) arrays of ~1.6 G each plus the SVD work; 24 G
# is generous. Revise from seff after the sub-03 pass.
#
# Usage:
#   python make_cells.py --subjects sub-03 --out cells_sub03.tsv
#   sbatch --array=1-$(wc -l < cells_sub03.tsv) fit_ladder.sbatch cells_sub03.tsv
#==============================================================================

CELLS=$1; shift
if [ -z "${CELLS}" ] || [ ! -f "${CELLS}" ]; then
    echo "ERROR: cells manifest missing: '${CELLS}' (write it with make_cells.py)" >&2
    exit 1
fi
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${CELLS}")
IFS=$'\t' read -r SUBJECT RUNG ROI PAIR <<< "${LINE}"
if [ -z "${PAIR}" ]; then
    echo "ERROR: no cell on line ${SLURM_ARRAY_TASK_ID} of ${CELLS}" >&2
    exit 1
fi

echo "=========================================="
echo "NR fit — Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME}"
echo "Cell: ${SUBJECT} rung ${RUNG} ${ROI} ${PAIR}"
echo "Start Time: $(date)"
echo "=========================================="

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python ${REPO_ROOT}/scripts/neural_rotation/fit_pair.py \
    --subject "${SUBJECT}" --rung "${RUNG}" --roi "${ROI}" --pair "${PAIR}" \
    --n-perm 1000 --n-perm-rot 100 "$@"

EXIT_STATUS=$?
echo ""
echo "End Time: $(date)   Exit Status: ${EXIT_STATUS}"
echo "Peak RSS / elapsed: run 'seff ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}'"
exit ${EXIT_STATUS}
