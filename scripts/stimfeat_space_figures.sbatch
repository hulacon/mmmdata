#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=stimfeat_space_figures
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/stimfeat_space_figures_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/stimfeat_space_figures_%j.err

#==============================================================================
# Joint stimulus-space figures (TB vs NAT) for EBind and CLIP
#==============================================================================
# Renders plot_stimulus_space.py for both models once their feature
# extractions are done. Typically submitted with
#   --dependency=afterany:<ebind_movies_job>:<clip_job>
# The plot script skips movies that are missing features, so a partial
# extraction still renders (with a warning in the log).
#
# Usage:  sbatch [--dependency=afterany:ID:ID] stimfeat_space_figures.sbatch
#==============================================================================

set -uo pipefail

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date -Is)"

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
PY=/gpfs/projects/hulacon/shared/envs/stimfeat/bin/python
set -e

for model in ebind clip; do
    echo "=== ${model} ==="
    "${PY}" "${REPO_ROOT}/scripts/plot_stimulus_space.py" --model "${model}"
done

echo "Finished $(date -Is)"
