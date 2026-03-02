#!/bin/bash
#
# submit_mriqc.sh - Helper script to submit MRIQC jobs to SLURM
#
# This script provides a simple interface for submitting MRIQC jobs
# with automatic dependency management.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Submit MRIQC jobs to SLURM cluster

OPTIONS:
    participant     Submit participant-level analysis (all subjects, sequential)
    array           Submit per-session analysis for all subjects [RECOMMENDED]
    group           Submit group-level analysis
    pipeline        Submit full pipeline (participant + group with dependency)
    array-pipeline  Submit full pipeline using per-session array jobs [RECOMMENDED]

    -h, --help      Show this help message

EXAMPLES:
    # Process all subjects per-session in parallel (recommended)
    $0 array

    # Run full pipeline with automatic group report
    $0 array-pipeline

    # Process subjects sequentially
    $0 participant

    # Generate group reports only (after participant-level done)
    $0 group

EOF
}

# Function to check if email is configured
check_email() {
    local batch_file=$1
    if grep -q "your-email@uoregon.edu" "$batch_file"; then
        print_warning "Email not configured in $batch_file"
        print_warning "Edit the file and replace 'your-email@uoregon.edu' with your email"
        return 1
    fi
    return 0
}

# Function to submit participant job
submit_participant() {
    print_info "Submitting participant-level analysis (sequential)..."
    check_email "${SCRIPT_DIR}/mriqc_participant.sbatch"

    JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/mriqc_participant.sbatch")

    if [ $? -eq 0 ]; then
        print_success "Job submitted successfully!"
        print_info "Job ID: ${JOB_ID}"
        print_info "Monitor with: squeue -u \$USER"
        print_info "View logs: tail -f ${SCRIPT_DIR}/../logs/mriqc_participant_${JOB_ID}.out"
        echo "$JOB_ID"
        return 0
    else
        print_error "Job submission failed"
        return 1
    fi
}

# Function to submit per-session array jobs for all subjects
submit_array() {
    print_info "Submitting per-session MRIQC analysis for all subjects..."
    check_email "${SCRIPT_DIR}/mriqc_array.sbatch"

    BIDS_DIR="/projects/hulacon/shared/mmmdata"
    SUBJECTS=($(ls -d ${BIDS_DIR}/sub-* | xargs -n 1 basename | sort))
    print_info "Found ${#SUBJECTS[@]} subjects"

    ALL_JOB_IDS=""

    for SUBJECT in "${SUBJECTS[@]}"; do
        # Count sessions for this subject
        N_SESSIONS=$(ls -d ${BIDS_DIR}/${SUBJECT}/ses-* 2>/dev/null | wc -l)

        if [ ${N_SESSIONS} -eq 0 ]; then
            print_warning "No sessions found for ${SUBJECT}, skipping"
            continue
        fi

        print_info "Submitting ${N_SESSIONS} session jobs for ${SUBJECT}..."

        JOB_ID=$(sbatch --parsable --export=ALL,MRIQC_SUBJECT=${SUBJECT} --array=1-${N_SESSIONS} "${SCRIPT_DIR}/mriqc_array.sbatch")

        if [ $? -eq 0 ]; then
            print_success "${SUBJECT}: submitted job ${JOB_ID} (${N_SESSIONS} sessions)"
            if [ -n "$ALL_JOB_IDS" ]; then
                ALL_JOB_IDS="${ALL_JOB_IDS}:${JOB_ID}"
            else
                ALL_JOB_IDS="${JOB_ID}"
            fi
        else
            print_error "${SUBJECT}: submission failed"
        fi
    done

    print_info "Monitor with: squeue -u \$USER -r"
    echo "$ALL_JOB_IDS"
}

# Function to submit group job
submit_group() {
    local dependency=$1

    if [ -z "$dependency" ]; then
        print_info "Submitting group-level analysis..."
        check_email "${SCRIPT_DIR}/mriqc_group.sbatch"
        JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/mriqc_group.sbatch")
    else
        print_info "Submitting group-level analysis (depends on job ${dependency})..."
        check_email "${SCRIPT_DIR}/mriqc_group.sbatch"
        JOB_ID=$(sbatch --parsable --dependency=afterok:${dependency} "${SCRIPT_DIR}/mriqc_group.sbatch")
    fi

    if [ $? -eq 0 ]; then
        print_success "Group job submitted successfully!"
        print_info "Job ID: ${JOB_ID}"
        if [ -n "$dependency" ]; then
            print_info "Will run after job ${dependency} completes"
        fi
        print_info "Monitor with: squeue -u \$USER"
        print_info "View logs: tail -f ${SCRIPT_DIR}/../logs/mriqc_group_${JOB_ID}.out"
        return 0
    else
        print_error "Job submission failed"
        return 1
    fi
}

# Main script
if [ $# -eq 0 ]; then
    print_error "No option specified"
    usage
    exit 1
fi

case "$1" in
    participant)
        submit_participant
        ;;

    array)
        submit_array
        ;;

    group)
        submit_group
        ;;

    pipeline)
        print_info "=== Submitting Full MRIQC Pipeline ==="
        print_info "Step 1: Participant-level analysis (sequential)"
        PART_JOB=$(submit_participant)

        if [ $? -eq 0 ]; then
            echo ""
            print_info "Step 2: Group-level analysis (will run after participant completes)"
            submit_group "$PART_JOB"

            echo ""
            print_success "=== Pipeline Submitted ==="
            print_info "Participant Job ID: ${PART_JOB}"
            print_info "Group job will automatically run after participant job completes"
        fi
        ;;

    array-pipeline)
        print_info "=== Submitting Full MRIQC Pipeline (Per-Session Array) ==="
        print_info "Step 1: Participant-level analysis (per-session)"
        ALL_JOBS=$(submit_array)

        if [ $? -eq 0 ] && [ -n "$ALL_JOBS" ]; then
            echo ""
            print_info "Step 2: Group-level analysis (will run after all session jobs complete)"
            submit_group "$ALL_JOBS"

            echo ""
            print_success "=== Pipeline Submitted ==="
            print_info "Session Job IDs: ${ALL_JOBS}"
            print_info "Group job will automatically run after all session jobs complete"
        fi
        ;;

    -h|--help)
        usage
        exit 0
        ;;

    *)
        print_error "Unknown option: $1"
        usage
        exit 1
        ;;
esac
