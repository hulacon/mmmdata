#!/usr/bin/env bash
# Build the shared "functional-space" conda environment on Talapas.
#
# One env for the functional-space alignment tooling in this directory:
# fmralign (Procrustes / ridge / OT, GroupAlignment), neuroboros
# (searchlights, polar decomposition), Connectome Workbench (wb_command, for
# surface sampling), plus the mmmdata scientific stack at the versions the
# mmmdata .venv runs. fmralign pulls torch, which is kept out of the mmmdata
# .venv on purpose; this env takes the CPU build.
#
# Follows psytwill/scripts/setup_env.sh (the stimfeat builder):
#   - `conda create --override-channels -c conda-forge`: the FSL installer's
#     ~/.condarc `channels: #!final` defeats every other way of choosing a
#     channel.
#   - PYTHONNOUSERSITE=1: ~/.local site-packages sit AHEAD of a conda env's
#     own site-packages.
#   - shared prefix, shared pip cache, group rwX.
# Python 3.12 is the constellation version standard (contracts §10); the
# pins below are the mmmdata .venv's, so code behaves the same in both.
#
# mmmdata itself is NOT pip-installed (it has no pyproject). As in the
# .venv, scripts put <mmmdata>/src/python on sys.path themselves.
#
#   ./scripts/functional_space/setup_env.sh                # shared prefix
#   ./scripts/functional_space/setup_env.sh --prefix PATH  # elsewhere
#
# Run on a compute node (srun/sbatch); compute nodes have network access.
# Rebuild policy: single writer. Delete the prefix and rebuild fresh before
# committing a new lock (an incremental solve is not a fresh one).
set -euo pipefail

umask 0002  # group-writable files from birth; parent setgid dirs handle group

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMMDATA="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENVS=/gpfs/projects/hulacon/shared/envs
SHARED_PREFIX=$ENVS/functional-space
PREFIX=$SHARED_PREFIX
CACHE=$ENVS/cache
CONDA_MODULE=miniconda3/20260319
LOCK_DIR="$SCRIPT_DIR/env"
TORCH_INDEX=https://download.pytorch.org/whl/cpu

# fmralign: git main, pinned. PyPI 0.0.5 forces numpy 1.26 and conflicts
# with the pins below; this commit installs cleanly against them.
FMRALIGN='fmralign @ git+https://github.com/Parietal-INRIA/fmralign.git@fc1ffbe40bb64dec4faad986435a316fc8ae00d9'
PINS=(
  numpy==2.3.5 scipy==1.16.3 pandas==2.3.3 scikit-learn==1.8.0
  nilearn==0.13.1 nibabel==5.3.3 joblib==1.5.3 matplotlib==3.10.8
  h5py==3.16.0 nitransforms==25.1.0 pybids==0.21.0 duckdb==1.5.5
)
EXTRAS=(neuroboros==0.1.9 pot pyarrow pytest)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)  PREFIX="${2:?--prefix needs a path}"; shift 2 ;;
    -h|--help) sed -n '2,29p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# conda: a bare `conda` on $PATH here is FSL's — load the module explicitly.
source /etc/profile.d/z00_lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load "$CONDA_MODULE" 2>/dev/null || {
  echo "error: could not 'module load $CONDA_MODULE'" >&2; exit 1
}

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$CACHE/pip"   # shared cache; also spares /home quota
mkdir -p "$CACHE/pip"

echo "prefix : $PREFIX"
echo "conda  : python=3.12 pip connectome-workbench (conda-forge, --override-channels)"
echo "pip    : torch (CPU) + fmralign @ pinned git + neuroboros + mmmdata pins"
echo

if [[ -d "$PREFIX" && -n "$(ls -A "$PREFIX" 2>/dev/null)" ]]; then
  echo "==> prefix exists; conda step skipped (delete it for a fresh solve)"
else
  echo "==> conda create"
  conda create --prefix "$PREFIX" --override-channels -c conda-forge --yes \
    python=3.12 pip connectome-workbench
fi

PY="$PREFIX/bin/python"

# torch first, from the CPU index only, so the resolve below cannot pick the
# PyPI build and its multi-GB CUDA dependencies.
echo
echo "==> pip install torch (CPU)"
"$PY" -m pip install --index-url "$TORCH_INDEX" torch

echo
echo "==> pip install (one resolve)"
"$PY" -m pip install "$FMRALIGN" "${PINS[@]}" "${EXTRAS[@]}"

echo
echo "==> verifying conda channel purity"
strays="$(conda list --prefix "$PREFIX" --show-channel-urls 2>/dev/null \
          | grep -vE '^#' | awk 'NF{print $NF}' \
          | grep -viE 'conda-forge|^pypi$' || true)"
if [[ -n "$strays" ]]; then
  echo "FAIL: packages from a channel other than conda-forge:" >&2
  echo "$strays" | sort | uniq -c >&2
  exit 1
fi
echo "ok — conda-forge (+ pypi payload) only"

# Import check from INSIDE the prefix: assert where each module came from,
# and exercise the entry points the routes use.
echo
"$PY" - "$PREFIX" "$MMMDATA/src/python" <<'PYCHECK'
import sys
prefix, mmmdata_src = sys.argv[1], sys.argv[2]
assert sys.executable.startswith(prefix), sys.executable
import numpy, scipy, pandas, sklearn, nilearn, nibabel, torch, ot
import fmralign, neuroboros
from fmralign import PairwiseAlignment, GroupAlignment  # noqa: F401
from neuroboros import searchlights  # noqa: F401
from neuroboros.linalg import safe_polar  # noqa: F401
print('python        ', sys.version.split()[0])
bad = []
for m in (numpy, scipy, pandas, sklearn, nilearn, nibabel, torch, ot,
          fmralign, neuroboros):
    where = getattr(m, '__file__', '') or ''
    print(f'{m.__name__:<14}', getattr(m, '__version__', 'unknown'))
    if not where.startswith(prefix):
        bad.append(f'{m.__name__} imported from {where}')
print('torch cuda    ', torch.version.cuda)
sys.path.insert(0, mmmdata_src)
from neuroimaging import data_quality  # noqa: F401
from neuroimaging.confounds import regime_design  # noqa: F401
print('mmmdata       ', 'neuroimaging.data_quality imports')
if bad:
    print('\nFAIL: modules resolved from outside the environment:', file=sys.stderr)
    print('\n'.join(bad), file=sys.stderr)
    sys.exit(1)
if torch.version.cuda is not None:
    sys.exit('FAIL: torch is a CUDA build; expected CPU')
# Exercise safe_polar rather than trusting the import.
rng = numpy.random.default_rng(0)
R, _ = safe_polar(rng.standard_normal((20, 20)))  # returns (u, p)
assert numpy.allclose(R @ R.T, numpy.eye(20), atol=1e-8)
print('smoke         ', 'safe_polar returns an orthogonal matrix')
PYCHECK
"$PREFIX/bin/wb_command" -version | sed -n '3p'

# Lock manifests: conda side (explicit URLs) + pip side.
mkdir -p "$LOCK_DIR"
CONDA_LOCK="$LOCK_DIR/lock-linux-64.txt"
PIP_LOCK="$LOCK_DIR/pip-lock-linux-64.txt"
echo
echo "==> writing $CONDA_LOCK"
{
  echo "# functional-space conda layer, resolved by setup_env.sh against a fresh"
  echo "# prefix. Recreate: conda create --prefix <p> --file env/lock-linux-64.txt"
  echo "# then the pip layer per env/pip-lock-linux-64.txt."
  conda list --prefix "$PREFIX" --explicit
} > "$CONDA_LOCK"
echo "==> writing $PIP_LOCK"
{
  echo "# functional-space pip layer, one resolve by setup_env.sh."
  echo "# torch comes from $TORCH_INDEX (CPU build); fmralign is the pinned git commit."
  "$PY" -m pip list --format=freeze
} > "$PIP_LOCK"
echo "    $(grep -cv '^#' "$PIP_LOCK") pip packages pinned"

# activate.d hook: user-site guard for every `conda activate` user.
mkdir -p "$PREFIX/etc/conda/activate.d" "$PREFIX/etc/conda/deactivate.d"
cat > "$PREFIX/etc/conda/activate.d/functional-space-env.sh" <<'HOOK'
# Written by mmmdata/scripts/functional_space/setup_env.sh.
# ~/.local site-packages shadow a conda env's own — disable user site.
export _FUNCSPACE_SAVED_PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-}"
export PYTHONNOUSERSITE=1
HOOK
cat > "$PREFIX/etc/conda/deactivate.d/functional-space-env.sh" <<'HOOK'
if [ -n "${_FUNCSPACE_SAVED_PYTHONNOUSERSITE:-}" ]; then
  export PYTHONNOUSERSITE="$_FUNCSPACE_SAVED_PYTHONNOUSERSITE"
else
  unset PYTHONNOUSERSITE
fi
unset _FUNCSPACE_SAVED_PYTHONNOUSERSITE
HOOK

echo "==> group-permission sweep (g+rwX over prefix)"
chmod -R g+rwX "$PREFIX" 2>/dev/null || true

cat <<EOF

Activate with:
  module load $CONDA_MODULE
  conda activate $PREFIX
or call $PREFIX/bin/python directly (set PYTHONNOUSERSITE=1).
EOF
