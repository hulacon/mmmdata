#!/bin/bash
#SBATCH --job-name=face_audit
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/face_audit_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/face_audit_%A_%a.err

#==============================================================================
# Face audit of a tree (release.faces): no anatomical face anywhere
#==============================================================================
# Read-only. Raw anatomy is checked against its defacing provenance, and every
# 3-D volume of the subject outside the EPI datatypes (derivatives included,
# whatever pipeline wrote them) is scored for signal inside the subject's
# recorded face mask. Exit 1 on any finding.
#
# One array task = one subject; the per-volume score table goes to
#   <source_dir>/anat_defacing/audits/<tag>/sub-XX_scores.tsv
# (outside the BIDS tree: it is a record, not part of the dataset).
#
# Usage:
#   sbatch --array=1-6 face_audit.sbatch                        # working tree, all subjects
#   sbatch --array=1-6 --export=ALL,AUDIT_ROOT=/path/to/share face_audit.sbatch
#   sbatch --export=ALL,AUDIT_SUBJECT=09 face_audit.sbatch      # one subject, no array
#   REPO_ROOT=<worktree> to run unmerged code.
#==============================================================================

SUBJECTS=(03 04 05 06 07 09)
if [ -n "${AUDIT_SUBJECT}" ]; then
    SUBJECT="${AUDIT_SUBJECT}"
else
    SUBJECT=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}
fi

export REPO_ROOT="${REPO_ROOT:-/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata}"
source "/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/scripts/load_config.sh"
PY="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/.venv/bin/python"
export PYTHONPATH="${REPO_ROOT}/src/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} \
       MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

ROOT="${AUDIT_ROOT:-$BIDS_PROJECT_DIR}"
TAG="${AUDIT_TAG:-$(basename "$ROOT")_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}"
OUT="${SOURCE_DIR}/anat_defacing/audits/${TAG}"
mkdir -p "$OUT"

echo "face audit: root=${ROOT} subject=sub-${SUBJECT} code=${REPO_ROOT} ($(git -C "$REPO_ROOT" rev-parse --short HEAD))"
echo "Node: ${SLURMD_NODENAME}  Start: $(date)"

"$PY" -u "${REPO_ROOT}/scripts/release_export.py" faces "$ROOT" --subject "$SUBJECT" \
    --report "${OUT}/sub-${SUBJECT}_scores.tsv"
EXIT_STATUS=$?

echo "End: $(date)  Exit: ${EXIT_STATUS}  (0 clean, 1 findings, 2 refused)"
exit ${EXIT_STATUS}
