#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=ses29_recall_transcribe
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/ses29_recall_transcribe_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/ses29_recall_transcribe_%j.err

#==============================================================================
# ses-29 final-free-recall transcription (workbench ses29-final-recall)
#==============================================================================
# aud2psy transcribe on the raw recall audio in mmmsourcedata, two arms:
#   standard  faster-whisper large-v3          (clean transcript)
#   verbatim  CrisperWhisper                   (keeps fillers/false starts)
#
# Outputs stay on the mmmsourcedata side: uncurated transcripts of
# participant speech are PII-adjacent and must not enter the shareable
# BIDS tree or any git repo. The curated BIDS beh.tsv is a later,
# reviewed step.
#
# Usage:  sbatch ses29_recall_transcribe.sbatch [subject_label]
# (subject label without prefix, default 03; expects exactly one
#  non-Backup *.m4a in <source_dir>/sub-XX/ses-29/audio/)
#==============================================================================

# set -e comes AFTER load_config.sh: its heredoc `read -r -d ''` idiom
# returns 1 at EOF by design and would abort the job under -e.
set -uo pipefail

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date -Is)"

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
export REPO_ROOT
source "${REPO_ROOT}/scripts/load_config.sh"
set -e

module load ffmpeg/6.0.0

# stimfeat env without conda: the interpreter directly, plus the activate.d
# hooks (shared HF/torch caches, PYTHONNOUSERSITE) which key off CONDA_PREFIX.
STIMFEAT=/gpfs/projects/hulacon/shared/envs/stimfeat
export CONDA_PREFIX="${STIMFEAT}"
for hook in "${STIMFEAT}"/etc/conda/activate.d/*.sh; do source "${hook}"; done
PY="${STIMFEAT}/bin/python"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

SUB="${1:-03}"

SOURCE_DIR=$("${PY}" -c "
import sys; sys.path.insert(0, '${REPO_ROOT}/src/python')
from core.config import load_config
print(load_config()['paths']['source_dir'])
")

AUDIO_DIR="${SOURCE_DIR}/sub-${SUB}/ses-29/audio"
mapfile -t CANDIDATES < <(find "${AUDIO_DIR}" -maxdepth 1 -name '*.m4a' ! -name 'Backup*')
if [ "${#CANDIDATES[@]}" -ne 1 ]; then
    echo "Expected exactly one non-Backup m4a in ${AUDIO_DIR}, found ${#CANDIDATES[@]}:" >&2
    printf '  %s\n' "${CANDIDATES[@]:-none}" >&2
    exit 1
fi
AUDIO="${CANDIDATES[0]}"
echo "Audio: ${AUDIO}"

OUT_ROOT="${SOURCE_DIR}/derivatives/recall_transcripts/sub-${SUB}/ses-29"
mkdir -p "${OUT_ROOT}/standard" "${OUT_ROOT}/verbatim"

echo "=== 1/2 standard (faster-whisper large-v3) ==="
"${PY}" -m aud2psy.cli transcribe "${AUDIO}" \
    -o "${OUT_ROOT}/standard/recall.csv" \
    --stimulus-id "sub-${SUB}_ses-29_recall" --language en

echo "=== 2/2 verbatim (CrisperWhisper, fillers kept) ==="
"${PY}" -m aud2psy.cli transcribe "${AUDIO}" \
    -o "${OUT_ROOT}/verbatim/recall.csv" \
    --stimulus-id "sub-${SUB}_ses-29_recall" --language en --verbatim

echo "Finished $(date -Is)"
