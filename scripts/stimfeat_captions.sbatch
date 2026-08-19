#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --job-name=stimfeat_captions
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/stimfeat_captions_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/stimfeat_captions_%j.err

#==============================================================================
# BLIP captions + text-arm embeddings for images and movie frames
#==============================================================================
# Puts TB and NAT visual content into the same TEXT arm of the joint spaces
# (a within-modality comparison that sidesteps the modality gap):
#   1. viz2psy caption   shared1000 images     -> caption.csv
#   2. viz2psy caption   movie frames @0.5 s   -> movies/<id>/caption.csv
#   3. word2psy ebind_text + clip_text on every caption
#      -> caption_{ebind,clip}_text_chunks.csv next to each caption.csv
# Caption rows get unique ids: image stimulus_id, or "<movie_id>@<time>".
# Every step skips outputs that already exist, so the job is safe to re-run.
#
# Usage:  sbatch stimfeat_captions.sbatch
#==============================================================================

# set -e comes AFTER load_config.sh: its heredoc `read -r -d ''` idiom
# returns 1 at EOF by design and would abort the job under -e.
set -uo pipefail

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date -Is)"

REPO_ROOT=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata
export REPO_ROOT
source "${REPO_ROOT}/scripts/load_config.sh"
set -e

STIMFEAT=/gpfs/projects/hulacon/shared/envs/stimfeat
export CONDA_PREFIX="${STIMFEAT}"
for hook in "${STIMFEAT}"/etc/conda/activate.d/*.sh; do source "${hook}"; done
export LD_LIBRARY_PATH="${STIMFEAT}/lib:${LD_LIBRARY_PATH:-}"
PY="${STIMFEAT}/bin/python"

STIM_DIR="${BIDS_DIR}/stimuli"
REGISTRY="${STIM_DIR}/stimulus_registry"
OUT_ROOT="${BIDS_DIR}/derivatives/stimuli_features"

# Embed one caption CSV through both text arms. $1 = dir holding caption.csv,
# $2 = python expression building the unique id from a csv.DictReader row.
embed_captions() {
    local dir="$1" id_expr="$2"
    local input="${dir}/.captions_input.csv"
    "${PY}" -c "
import csv
with open('${dir}/caption.csv') as f, open('${input}', 'w', newline='') as out:
    w = csv.writer(out); w.writerow(['caption_id', 'text'])
    for row in csv.DictReader(f):
        w.writerow([${id_expr}, row['caption_text']])
"
    for arm in ebind_text clip_text; do
        if [[ -f "${dir}/caption_${arm}_chunks.csv" ]]; then
            echo "    ${arm}: exists, skipping"
        else
            "${PY}" -m word2psy.cli "${arm}" "${input}" \
                --text-column text --id-column caption_id \
                -o "${dir}/caption_${arm}.csv" --quiet
        fi
    done
}

IMG_ID="row['stimulus_id']"
MOV_ID="f\"{row['stimulus_id']}@{row['time']}\""

if [[ -f "${OUT_ROOT}/shared1000/caption.csv" ]]; then
    echo "=== shared1000 captions exist, skipping"
else
    echo "=== viz2psy caption: shared1000 images ==="
    "${PY}" -m viz2psy.cli caption "${STIM_DIR}"/shared1000/images/*.png \
        -o "${OUT_ROOT}/shared1000/caption.csv" --no-viz --quiet
fi
echo "=== text arms: shared1000 captions ==="
embed_captions "${OUT_ROOT}/shared1000" "${IMG_ID}"

MOVIE_LIST=$("${PY}" -c "
import csv
with open('${REGISTRY}/movies.tsv') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        print(f\"{row['stimulus_id']}\t{row['video_file']}\")
")

n_done=0 n_skip=0 n_fail=0
failed_ids=()
while IFS=$'\t' read -r MOVIE_ID MOVIE_FILE; do
    OUT_DIR="${OUT_ROOT}/movies/${MOVIE_ID}"
    if [[ -f "${OUT_DIR}/caption_ebind_text_chunks.csv" \
          && -f "${OUT_DIR}/caption_clip_text_chunks.csv" ]]; then
        echo "--- ${MOVIE_ID}: complete, skipping"
        n_skip=$((n_skip + 1))
        continue
    fi
    echo "=== ${MOVIE_ID} ($(date -Is)) ==="
    mkdir -p "${OUT_DIR}"
    # A failed clip is reported at the end but does not kill the sweep.
    ok=1
    if [[ ! -f "${OUT_DIR}/caption.csv" ]]; then
        "${PY}" -m viz2psy.cli caption "${STIM_DIR}/movies/${MOVIE_FILE}" \
            -o "${OUT_DIR}/caption.csv" \
            --stimulus-id "${MOVIE_ID}" --frame-interval 0.5 --no-save-frames \
            --no-viz --quiet || ok=0
    fi
    if (( ok )); then
        embed_captions "${OUT_DIR}" "${MOV_ID}" || ok=0
    fi
    if (( ok )); then
        n_done=$((n_done + 1))
    else
        echo "!!! ${MOVIE_ID} FAILED"
        n_fail=$((n_fail + 1))
        failed_ids+=("${MOVIE_ID}")
    fi
done <<< "${MOVIE_LIST}"

echo "Finished $(date -Is): ${n_done} captioned+embedded, ${n_skip} skipped, ${n_fail} failed"
if (( n_fail > 0 )); then
    echo "Failed clips: ${failed_ids[*]}"
    exit 1
fi
