#!/bin/bash
#SBATCH --account=hulacon
#SBATCH --partition=gpu,preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-80gb
#SBATCH --requeue
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6
#SBATCH --job-name=nat-encret-rmatch
#SBATCH --output=logs/%x_%A_%a.out
# rMatch prompt + local Gemma 4 31B via vLLM: one shard of NAT recall prompts (rmatch_run.py vgenerate).
#
# Usage (from the mmmdata repo root, so logs/ resolves; WORK is a private scratch dir):
#   python scripts/nat_encret/rmatch_run.py prompts "$WORK" [--subjects 06 07]
#   sbatch --array=0-3 --export=ALL,WORK="$WORK",COND=real,NTASKS=4 scripts/nat_encret/rmatch_vllm.sbatch [--limit K]
#   python scripts/nat_encret/rmatch_run.py collect "$WORK" --condition real
#   python scripts/nat_encret/build_nat_encret.py map "$WORK"
# Shards are by film, so each holds whole films and vLLM's prefix cache reuses a film's story.
# Each task writes its shard only at the end, so a preempt requeue is safe (it reruns the shard whole).
# Cost (A100 80 GB PCIe, 7,467 units over 60 films, 4 shards): 24-46 min per shard, 2.3 GPU-h in total.
#
# RMATCH_ENV is a venv built from rMatch's uv.lock (vLLM 0.31) on the stimfeat env's conda Python.
# On Talapas (glibc 2.28) it needs two settings:
#  - conda's libstdc++ (CXXABI_1.3.15) ahead of /lib64's, for ICU;
#  - VLLM_USE_FLASHINFER_SAMPLER=0: FlashInfer's JIT sampling kernel does not compile against glibc
#    2.28 headers, and greedy decoding does not need it (vLLM's PyTorch sampler is used instead).
set -euo pipefail
: "${WORK:?set WORK=<rmatch_run.py work dir>}"
ENVS=/gpfs/projects/hulacon/shared/envs
RMATCH_ENV=${RMATCH_ENV:-$ENVS/rmatch}
export HF_HOME=${HF_HOME:-$ENVS/cache/huggingface} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/gpfs/scratch/hulacon/$USER/vllm-cache}
export VLLM_USE_FLASHINFER_SAMPLER=0
export LD_LIBRARY_PATH=$ENVS/stimfeat/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/rmatch_run.py" ] || HERE="$SLURM_SUBMIT_DIR/scripts/nat_encret"  # sbatch copies the script
git -C "$HERE" log -1 --format='mmmdata %h %s'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$RMATCH_ENV/bin/python" -u "$HERE/rmatch_run.py" vgenerate "$WORK" \
    --condition "${COND:?set COND=real|pseudo}" --task "${SLURM_ARRAY_TASK_ID:-0}" --n-tasks "${NTASKS:-1}" "$@"
