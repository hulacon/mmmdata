#!/bin/bash
#SBATCH --job-name=mriqc_anat
#SBATCH --account=hulacon
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/mriqc_anat_%A_%a.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/mriqc_anat_%A_%a.err
#SBATCH --exclude=n0119,n0123,n0128,n0130,n0131,n0190

#==============================================================================
# MRIQC anatomical pass on the DEFACED anatomy — one array task per
# (subject, session) that carries T1w/T2w
#==============================================================================
# Workbench: mmmdata-agents/docs/workbench/anat-defacing/ (DECIDED 2026-09-11:
# the tree's anatomical QC reports derive from what the tree holds). The
# faced-era IQM JSONs and group tables were moved to
# mmmsourcedata/derivatives_faced/mriqc/ first (recorded in each subject's
# anat_defacing provenance under `iqm_faced`); this pass rewrites the
# per-run JSONs, HTML reports and figures in place. Background-dependent IQMs
# (FBER, QI1/QI2, SNRd, EFC) shift on defaced input — that is expected, and
# the README says which set is which.
#
# Array task mapping (sessions with anatomy):
#   1: sub-03/ses-01   2: sub-03/ses-28   3: sub-03/ses-30
#   4: sub-04/ses-01   5: sub-04/ses-28   6: sub-04/ses-30
#   7: sub-05/ses-01   8: sub-05/ses-28   9: sub-05/ses-30
#  10: sub-06/ses-01  11: sub-07/ses-01
#
# Usage:
#   sbatch --array=1 mriqc_anat_defaced.sbatch      # pilot
#   sbatch --array=2-11 mriqc_anat_defaced.sbatch   # the rest
#   then: sbatch mriqc_group.sbatch                  # regenerate group_*.tsv/html
#==============================================================================

SUBJECTS=(sub-03 sub-03 sub-03 sub-04 sub-04 sub-04 sub-05 sub-05 sub-05 sub-06 sub-07)
SESSIONS=(ses-01 ses-28 ses-30 ses-01 ses-28 ses-30 ses-01 ses-28 ses-30 ses-01 ses-01)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
SUBJECT=${SUBJECTS[$IDX]}
SESSION=${SESSIONS[$IDX]}

echo "=========================================="
echo "MRIQC anat (defaced) — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}   Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Subject: ${SUBJECT}   Session: ${SESSION}"
echo "Start Time: $(date)"
echo "=========================================="

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/.venv/bin/activate"
source "${REPO_ROOT}/scripts/load_config.sh"
export SINGULARITYENV_TEMPLATEFLOW_HOME=${BIDS_DIR}/.cache/templateflow
export SINGULARITY_TMPDIR=${CODE_ROOT}/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p ${SINGULARITY_TMPDIR}

cd ${REPO_ROOT}/scripts
python run_mriqc.py \
    --analysis-level participant \
    --subjects ${SUBJECT} \
    --session ${SESSION} \
    --modalities T1w T2w \
    --nprocs 2 \
    --omp-nthreads 2 \
    --mem-gb $((${SLURM_MEM_PER_NODE} / 1024))
EXIT_STATUS=$?

echo ""
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
[ -d "${SINGULARITY_TMPDIR}" ] && rm -rf "${SINGULARITY_TMPDIR}"
exit ${EXIT_STATUS}
