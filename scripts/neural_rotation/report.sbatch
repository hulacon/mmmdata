#!/bin/bash
#SBATCH --job-name=nr_report
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/nr_report_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/nr_report_%j.err

#==============================================================================
# Neural-rotation pilot: concatenate the per-cell fit tables into the design
# record's tables + Vega-Lite specs (report.py). Seconds; a login node does
# it too. Arguments: <out dir> [report.py options].
#==============================================================================

OUT=$1; shift
export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"

python ${REPO_ROOT}/scripts/neural_rotation/report.py \
    --in "${OUTPUT_DIR}/neural_rotation" --out "${OUT:-${OUTPUT_DIR}/neural_rotation/report}" "$@"
