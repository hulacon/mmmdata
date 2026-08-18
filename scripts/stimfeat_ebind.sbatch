#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=stimfeat_ebind
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/stimfeat_ebind_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/stimfeat_ebind_%j.err

#==============================================================================
# EBind shared-space feature extraction (workbench ebind-shared-space)
#==============================================================================
# One cross-modal 1024-d space over all three stimulus sets, checkpoint
# encord-team/ebind-full (Contract B §4.1 sidecars, 4-digit indices):
#   1. viz2psy  ebind        shared1000 images   -> stimuli_features/shared1000/
#   2. word2psy ebind_text   twp1000 words       -> stimuli_features/twp1000/
#   3. viz2psy  ebind        movie frames @0.5s  -> stimuli_features/movies/<id>/
#   4. aud2psy  ebind_audio  movie soundtrack    -> stimuli_features/movies/<id>/
#
# Usage:  sbatch stimfeat_ebind.sbatch [movie_stimulus_id]
# (movie id defaults to adventure-time; must exist in the stimulus registry)
#==============================================================================

# set -e comes AFTER load_config.sh: its heredoc `read -r -d ''` idiom
# returns 1 at EOF by design and would abort the job under -e.
set -uo pipefail

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date -Is)"

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
export REPO_ROOT
source "${REPO_ROOT}/scripts/load_config.sh"
set -e

# stimfeat env without conda: the interpreter directly, plus the activate.d
# hooks (shared HF/torch caches, PYTHONNOUSERSITE) which key off CONDA_PREFIX.
STIMFEAT=/gpfs/projects/hulacon/shared/envs/stimfeat
export CONDA_PREFIX="${STIMFEAT}"
for hook in "${STIMFEAT}"/etc/conda/activate.d/*.sh; do source "${hook}"; done
# torchcodec loads its FFmpeg backend from the env's own conda-bundled libs
export LD_LIBRARY_PATH="${STIMFEAT}/lib:${LD_LIBRARY_PATH:-}"
PY="${STIMFEAT}/bin/python"

STIM_DIR="${BIDS_DIR}/stimuli"
REGISTRY="${STIM_DIR}/stimulus_registry"
OUT_ROOT="${BIDS_DIR}/derivatives/stimuli_features"
MOVIE_ID="${1:-adventure-time}"

MOVIE_FILE=$("${PY}" -c "
import csv, sys
with open('${REGISTRY}/movies.tsv') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        if row['stimulus_id'] == '${MOVIE_ID}':
            print(row['video_file']); sys.exit(0)
sys.exit('${MOVIE_ID} not in the movie registry')
")
echo "Movie: ${MOVIE_ID} -> ${MOVIE_FILE}"

mkdir -p "${OUT_ROOT}/shared1000" "${OUT_ROOT}/twp1000" "${OUT_ROOT}/movies/${MOVIE_ID}"

echo "=== 1/4 viz2psy ebind: shared1000 images ==="
"${PY}" -m viz2psy.cli ebind "${STIM_DIR}"/shared1000/images/*.png \
    -o "${OUT_ROOT}/shared1000/ebind.csv" --batch-size 64 --no-viz --quiet

echo "=== 2/4 word2psy ebind_text: twp1000 words ==="
WORDS_CSV="${OUT_ROOT}/twp1000/.twp1000_words_input.csv"
"${PY}" -c "
import csv
with open('${REGISTRY}/twp1000.tsv') as f, open('${WORDS_CSV}', 'w', newline='') as out:
    w = csv.writer(out); w.writerow(['stimulus_id', 'word'])
    for row in csv.DictReader(f, delimiter='\t'):
        w.writerow([row['stimulus_id'], row['stimulus_id']])
"
"${PY}" -m word2psy.cli ebind_text "${WORDS_CSV}" \
    --text-column word --id-column stimulus_id \
    -o "${OUT_ROOT}/twp1000/ebind_text.csv"

echo "=== 3/4 viz2psy ebind: ${MOVIE_ID} frames (0.5 s grid) ==="
"${PY}" -m viz2psy.cli ebind "${STIM_DIR}/movies/${MOVIE_FILE}" \
    -o "${OUT_ROOT}/movies/${MOVIE_ID}/ebind.csv" \
    --stimulus-id "${MOVIE_ID}" --frame-interval 0.5 --no-save-frames \
    --batch-size 64 --no-viz --quiet

echo "=== 4/4 aud2psy ebind_audio: ${MOVIE_ID} soundtrack (0.5 s grid) ==="
"${PY}" -m aud2psy.cli ebind_audio "${STIM_DIR}/movies/${MOVIE_FILE}" \
    -o "${OUT_ROOT}/movies/${MOVIE_ID}/ebind_audio.csv" \
    --stimulus-id "${MOVIE_ID}" --hop 0.5

echo "Finished $(date -Is)"
