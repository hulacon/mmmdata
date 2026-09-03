#!/bin/bash
#SBATCH --job-name=physio_triage
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=physio_triage_%j.out
#SBATCH --error=physio_triage_%j.err

# Scanner-physio triage: parse every PhysioLog DICOM in sourcedata and write
# src/python/raw2bids_converters/physio_triage.csv (generated data, untracked).
# generate_inventory.py refuses to run without that table unless told to.
#
# The generator needs pydicom. mmmdata's own .venv does not carry it; the
# shared stimfeat env does, which is why this driver exists: it is the one
# place the working interpreter is written down. Extra arguments pass through
# (e.g. `sbatch scripts/physio_triage.sbatch --check` to compare against the
# existing table without writing, or `--append` to add new subjects only).

ENV=/gpfs/projects/hulacon/shared/envs/stimfeat
source ${ENV}/etc/conda/activate.d/stimfeat-env.sh  # PYTHONNOUSERSITE=1 + shared model caches
PYTHON=${ENV}/bin/python3
CONVERTER_DIR=/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata/src/python/raw2bids_converters

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Python: $PYTHON"
echo "Working dir: $CONVERTER_DIR"
echo ""

cd "$CONVERTER_DIR"
$PYTHON generate_physio_triage.py "$@"
RC=$?

echo ""
echo "generate_physio_triage.py exit code: $RC"
echo "Job finished: $(date)"
exit $RC
