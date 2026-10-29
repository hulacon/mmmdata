#!/bin/bash
#SBATCH --job-name=glmsingle_tb_space
#SBATCH --account=hulacon
#SBATCH --partition=computelong
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=280G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glmsingle_tb_space_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/glmsingle_tb_space_%j.err

#==============================================================================
# GLMsingle TB — one subject x arm, in a non-default fMRIPrep volume space
#==============================================================================
# glmsingle_tb.py --space <space> --output-base <tree>. The default-space
# (MNI) campaign is glmsingle_tb_siloed.sbatch; a non-default space must go to
# its own tree, which glmsingle_tb.py enforces.
#
# Sizing: the header fits an enc arm. On the MNI res-2 grid the enc arm peaked
# at ~210-230 G over ~20 h (8 CPUs, 42 runs) and ran out of memory at 220 G;
# a ret-image arm (TB + FIN) peaked at ~95-120 G over 6-12 h. Memory and time
# scale roughly with voxel count, so a native grid with fewer voxels needs no
# more. For a ret arm, override on the command line, e.g.
#   sbatch --partition=compute --mem=150G --time=18:00:00 ...
#
# Usage:
#   sbatch glmsingle_tb_space.sbatch <sub-##> <arm> <space> <output_base> [<fmriprep_dir>]
#   sbatch glmsingle_tb_space.sbatch sub-## enc T1w <scratch>/glmsingle_tb_T1w
# <fmriprep_dir> (optional) reads the BOLD from another fMRIPrep tree, e.g. an
# alignment arm's backfill output (default: derivatives/fmriprep).
#==============================================================================

if [ $# -ne 4 ] && [ $# -ne 5 ]; then
    echo "usage: sbatch glmsingle_tb_space.sbatch <subject> <arm> <space> <output_base> [<fmriprep_dir>]" >&2
    exit 2
fi
SUBJECT=$1
ARM=$2
SPACE=$3
OUTPUT_BASE=$4
FMRIPREP_DIR=${5:-}

echo "=========================================="
echo "GLMsingle TB (space) — SLURM Job Info"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Subject: ${SUBJECT}  Arm: ${ARM}  Space: ${SPACE}"
echo "Output base: ${OUTPUT_BASE}"
echo "fMRIPrep dir: ${FMRIPREP_DIR:-<default>}"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/scripts/load_config.sh"
source "${REPO_ROOT}/.venv/bin/activate"

python ${REPO_ROOT}/scripts/glmsingle_tb.py \
    --subject ${SUBJECT} \
    --arm ${ARM} \
    --space ${SPACE} \
    --output-base ${OUTPUT_BASE} \
    ${FMRIPREP_DIR:+--fmriprep-dir ${FMRIPREP_DIR}}

EXIT_STATUS=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Status: ${EXIT_STATUS}"
echo "Peak RSS / elapsed: run 'seff ${SLURM_JOB_ID}'"
echo "=========================================="

exit ${EXIT_STATUS}
