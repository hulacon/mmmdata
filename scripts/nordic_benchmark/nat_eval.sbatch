#!/bin/bash
#SBATCH --job-name=nat_eval
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/nat_eval_%j.out
#SBATCH --error=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/logs/nat_eval_%j.err

# NAT benchmark eval — within-movie vs between-movie cross-subject ISC,
# all 10 NAT sessions × both pipelines, all Harvard-Oxford ROIs.
# Runtime estimate: ~15 min.

echo "=========================================="
echo "NAT eval — start $(date)"
echo "Node: ${SLURMD_NODENAME}"
echo "=========================================="

export REPO_ROOT="/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata"
source "${REPO_ROOT}/.venv/bin/activate"

python "${REPO_ROOT}/scripts/nordic_benchmark/nat_eval.py" --pipeline both
EXIT=$?

echo "=========================================="
echo "End $(date)  exit ${EXIT}"
echo "=========================================="
exit ${EXIT}
