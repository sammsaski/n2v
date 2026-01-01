#!/bin/bash
#
# ACAS Xu VNN-COMP 2025 Benchmark Runner
#
# Runs the full ACAS Xu benchmark suite following NNV's VNN-COMP 2025 strategy.
# Uses bash timeout for reliable process termination.
#
# Usage:
#   ./run_benchmark.sh                      # Run all 186 instances
#   ./run_benchmark.sh --timeout 60         # Custom timeout
#   ./run_benchmark.sh --property 2         # Only prop_2 instances
#   ./run_benchmark.sh --subset 10          # Random 10 instances
#   ./run_benchmark.sh --csv results.csv    # Custom output file
#
# Note: Activate your n2v conda environment before running this script.

set -e

# Default configuration
TIMEOUT=120
WORKERS=$(nproc)
FALSIFY_METHOD="random"
FALSIFY_SAMPLES=500
PGD_RESTARTS=10
PGD_STEPS=50
OUTPUT_CSV="results/benchmark_results.csv"
PROPERTY_FILTER=""
SUBSET=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --falsify-method)
            FALSIFY_METHOD="$2"
            shift 2
            ;;
        --falsify-samples)
            FALSIFY_SAMPLES="$2"
            shift 2
            ;;
        --pgd-restarts)
            PGD_RESTARTS="$2"
            shift 2
            ;;
        --pgd-steps)
            PGD_STEPS="$2"
            shift 2
            ;;
        --csv)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        --property)
            PROPERTY_FILTER="$2"
            shift 2
            ;;
        --subset)
            SUBSET="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --timeout SECONDS      Timeout per instance (default: 120)"
            echo "  --workers N            Number of parallel workers (default: CPU count)"
            echo "  --falsify-method M     Falsification method: random, pgd, random+pgd (default: random)"
            echo "  --falsify-samples N    Random falsification samples (default: 500)"
            echo "  --pgd-restarts N       PGD restarts (default: 10)"
            echo "  --pgd-steps N          PGD steps per restart (default: 50)"
            echo "  --csv FILE             Output CSV file (default: results/benchmark_results.csv)"
            echo "  --property N           Only run property N (1-10)"
            echo "  --subset N             Run N randomly selected instances"
            echo "  -h, --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$(dirname "$OUTPUT_CSV")"

# Build instances list from VNN-COMP format
INSTANCES_CSV="$HOME/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/acasxu_2023/instances.csv"

if [[ ! -f "$INSTANCES_CSV" ]]; then
    echo "Error: instances.csv not found at $INSTANCES_CSV"
    exit 1
fi

# Read instances
declare -a INSTANCES
while IFS=, read -r onnx vnnlib timeout_csv; do
    # Extract just filenames
    onnx_name=$(basename "$onnx")
    vnnlib_name=$(basename "$vnnlib")

    # Check if local files exist
    if [[ -f "$SCRIPT_DIR/onnx/$onnx_name" ]] && [[ -f "$SCRIPT_DIR/vnnlib/$vnnlib_name" ]]; then
        # Apply property filter if set
        if [[ -n "$PROPERTY_FILTER" ]]; then
            if [[ "$vnnlib_name" != *"prop_${PROPERTY_FILTER}."* ]]; then
                continue
            fi
        fi
        INSTANCES+=("$onnx_name,$vnnlib_name")
    fi
done < "$INSTANCES_CSV"

# Apply random subset if requested
if [[ -n "$SUBSET" ]]; then
    # Shuffle and take first N instances
    SHUFFLED=($(printf '%s\n' "${INSTANCES[@]}" | shuf))
    INSTANCES=("${SHUFFLED[@]:0:$SUBSET}")
fi

TOTAL=${#INSTANCES[@]}

if [[ $TOTAL -eq 0 ]]; then
    echo "Error: No instances found"
    exit 1
fi

# Print header
echo "================================================================================"
echo "ACAS Xu VNN-COMP 2025 Benchmark"
echo "================================================================================"
echo "Total instances: $TOTAL"
echo "Timeout: ${TIMEOUT}s"
echo "Workers: $WORKERS"
echo "Falsification: $FALSIFY_METHOD (samples=$FALSIFY_SAMPLES, pgd_restarts=$PGD_RESTARTS, pgd_steps=$PGD_STEPS)"
echo "Output: $OUTPUT_CSV"
echo "================================================================================"
echo ""

# Initialize CSV
echo "onnx_file,vnnlib_file,result,time,method" > "$OUTPUT_CSV"

# Statistics
SAT=0
UNSAT=0
UNKNOWN=0
TIMEOUT_COUNT=0
ERROR=0
TOTAL_TIME=0

# Run each instance
COUNT=0
for instance in "${INSTANCES[@]}"; do
    COUNT=$((COUNT + 1))

    IFS=',' read -r onnx_name vnnlib_name <<< "$instance"

    echo -n "[$COUNT/$TOTAL] $onnx_name + $vnnlib_name ... "

    # Run with timeout
    EXIT_CODE=0
    OUTPUT=$(timeout --kill-after=5 "$TIMEOUT" python "$SCRIPT_DIR/run_instance.py" \
        "$SCRIPT_DIR/onnx/$onnx_name" \
        "$SCRIPT_DIR/vnnlib/$vnnlib_name" \
        --workers "$WORKERS" \
        --falsify-method "$FALSIFY_METHOD" \
        --falsify-samples "$FALSIFY_SAMPLES" \
        --pgd-restarts "$PGD_RESTARTS" \
        --pgd-steps "$PGD_STEPS" 2>/dev/null) || EXIT_CODE=$?

    # Parse output
    if [[ $EXIT_CODE -eq 124 ]] || [[ $EXIT_CODE -eq 137 ]]; then
        # Timeout
        RESULT="TIMEOUT"
        TIME="$TIMEOUT"
        METHOD=""
        TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
    else
        RESULT=$(echo "$OUTPUT" | grep "^RESULT:" | cut -d: -f2)
        TIME=$(echo "$OUTPUT" | grep "^TIME:" | cut -d: -f2)
        METHOD=$(echo "$OUTPUT" | grep "^METHOD:" | cut -d: -f2)

        # Update statistics
        case "$RESULT" in
            SAT) SAT=$((SAT + 1)) ;;
            UNSAT) UNSAT=$((UNSAT + 1)) ;;
            UNKNOWN) UNKNOWN=$((UNKNOWN + 1)) ;;
            ERROR) ERROR=$((ERROR + 1)) ;;
        esac
    fi

    # Default values if parsing failed
    RESULT=${RESULT:-ERROR}
    TIME=${TIME:-0}
    METHOD=${METHOD:-none}

    echo "$RESULT ($METHOD, ${TIME}s)"

    # Append to CSV
    echo "$onnx_name,$vnnlib_name,$RESULT,$TIME,$METHOD" >> "$OUTPUT_CSV"

    # Accumulate total time
    TOTAL_TIME=$(echo "$TOTAL_TIME + $TIME" | bc)
done

# Print summary
SOLVED=$((SAT + UNSAT))
echo ""
echo "================================================================================"
echo "BENCHMARK SUMMARY"
echo "================================================================================"
echo "Total instances: $TOTAL"
printf "SAT:     %3d (%.1f%%)\n" $SAT $(echo "scale=1; 100*$SAT/$TOTAL" | bc)
printf "UNSAT:   %3d (%.1f%%)\n" $UNSAT $(echo "scale=1; 100*$UNSAT/$TOTAL" | bc)
printf "UNKNOWN: %3d (%.1f%%)\n" $UNKNOWN $(echo "scale=1; 100*$UNKNOWN/$TOTAL" | bc)
printf "TIMEOUT: %3d (%.1f%%)\n" $TIMEOUT_COUNT $(echo "scale=1; 100*$TIMEOUT_COUNT/$TOTAL" | bc)
printf "ERROR:   %3d (%.1f%%)\n" $ERROR $(echo "scale=1; 100*$ERROR/$TOTAL" | bc)
echo "----------------------------------------"
printf "Solved:  %3d (%.1f%%)\n" $SOLVED $(echo "scale=1; 100*$SOLVED/$TOTAL" | bc)
printf "Total time: %.1fs (%.1fm)\n" $TOTAL_TIME $(echo "scale=1; $TOTAL_TIME/60" | bc)
echo "================================================================================"
