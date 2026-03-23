#!/bin/bash
#
# Generic VNN-COMP Benchmark Runner
#
# Runs all instances from a VNN-COMP instances.csv file.
#
# Usage:
#   ./run_benchmark.sh <benchmark_dir> [OPTIONS]
#
# Arguments:
#   benchmark_dir   Directory containing instances.csv and model/spec files
#
# Options:
#   --timeout N     Timeout per instance in seconds (default: 120)
#   --output FILE   Output CSV file (default: results.csv)
#   --workers N     Number of parallel LP workers (default: CPU count)
#   --no-falsify    Skip falsification stage
#   --no-approx     Skip approximate reachability stage
#   --no-exact      Skip exact reachability stage
#   --subset N      Run first N instances only
#
# The instances.csv should have lines: onnx_path,vnnlib_path,timeout
# Paths are relative to the benchmark_dir.

set -e

# Default configuration
TIMEOUT=120
OUTPUT_CSV="results.csv"
WORKERS=$(nproc)
NO_FALSIFY=""
NO_APPROX=""
NO_EXACT=""
SUBSET=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for benchmark dir
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <benchmark_dir> [OPTIONS]"
    echo "Run '$0 --help' for more information."
    exit 1
fi

BENCHMARK_DIR="$1"
shift

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --output)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --no-falsify)
            NO_FALSIFY="--no-falsify"
            shift
            ;;
        --no-approx)
            NO_APPROX="--no-approx"
            shift
            ;;
        --no-exact)
            NO_EXACT="--no-exact"
            shift
            ;;
        --subset)
            SUBSET="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 <benchmark_dir> [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  benchmark_dir          Directory with instances.csv and model/spec files"
            echo ""
            echo "Options:"
            echo "  --timeout SECONDS      Timeout per instance (default: 120)"
            echo "  --output FILE          Output CSV file (default: results.csv)"
            echo "  --workers N            Number of parallel LP workers (default: CPU count)"
            echo "  --no-falsify           Skip falsification stage"
            echo "  --no-approx            Skip approximate reachability stage"
            echo "  --no-exact             Skip exact reachability stage"
            echo "  --subset N             Run first N instances only"
            echo "  -h, --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate benchmark directory
INSTANCES_CSV="$BENCHMARK_DIR/instances.csv"
if [[ ! -f "$INSTANCES_CSV" ]]; then
    echo "Error: instances.csv not found at $INSTANCES_CSV"
    exit 1
fi

# Read instances
declare -a INSTANCES
while IFS=, read -r onnx_path vnnlib_path inst_timeout; do
    # Skip empty lines and comments
    [[ -z "$onnx_path" || "$onnx_path" == \#* ]] && continue

    # Resolve paths relative to benchmark dir
    onnx_full="$BENCHMARK_DIR/$onnx_path"
    vnnlib_full="$BENCHMARK_DIR/$vnnlib_path"

    if [[ -f "$onnx_full" ]] && [[ -f "$vnnlib_full" ]]; then
        # Use instance-specific timeout if available, otherwise default
        t="${inst_timeout:-$TIMEOUT}"
        # Clean whitespace from timeout
        t=$(echo "$t" | tr -d '[:space:]')
        INSTANCES+=("$onnx_full,$vnnlib_full,$t")
    else
        echo "Warning: skipping missing files: $onnx_full or $vnnlib_full"
    fi
done < "$INSTANCES_CSV"

# Apply subset limit
if [[ -n "$SUBSET" ]]; then
    INSTANCES=("${INSTANCES[@]:0:$SUBSET}")
fi

TOTAL=${#INSTANCES[@]}

if [[ $TOTAL -eq 0 ]]; then
    echo "Error: No valid instances found in $INSTANCES_CSV"
    exit 1
fi

# Print header
echo "================================================================================"
echo "VNN-COMP Benchmark Runner"
echo "================================================================================"
echo "Benchmark:  $BENCHMARK_DIR"
echo "Instances:  $TOTAL"
echo "Timeout:    ${TIMEOUT}s (default, may be overridden per instance)"
echo "Workers:    $WORKERS"
echo "Output:     $OUTPUT_CSV"
echo "================================================================================"
echo ""

# Initialize CSV
echo "onnx_file,vnnlib_file,result,time" > "$OUTPUT_CSV"

# Statistics
SAT=0
UNSAT=0
UNKNOWN=0
TIMEOUT_COUNT=0
ERROR_COUNT=0
TOTAL_TIME=0

# Run each instance
COUNT=0
for instance in "${INSTANCES[@]}"; do
    COUNT=$((COUNT + 1))

    IFS=',' read -r onnx_full vnnlib_full inst_timeout <<< "$instance"

    onnx_name=$(basename "$onnx_full")
    vnnlib_name=$(basename "$vnnlib_full")

    echo -n "[$COUNT/$TOTAL] $onnx_name + $vnnlib_name ... "

    # Run with timeout
    EXIT_CODE=0
    INST_START=$(date +%s%N)
    OUTPUT=$(timeout --kill-after=5 "$inst_timeout" python "$SCRIPT_DIR/run_instance.py" \
        "$onnx_full" \
        "$vnnlib_full" \
        --workers "$WORKERS" \
        $NO_FALSIFY \
        $NO_APPROX \
        $NO_EXACT \
        2>/dev/null) || EXIT_CODE=$?
    INST_END=$(date +%s%N)

    # Compute wall time
    WALL_TIME=$(echo "scale=3; ($INST_END - $INST_START) / 1000000000" | bc)

    # Parse output
    if [[ $EXIT_CODE -eq 124 ]] || [[ $EXIT_CODE -eq 137 ]]; then
        RESULT="timeout"
        WALL_TIME="$inst_timeout"
        TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
    else
        # First line of output is the result
        RESULT=$(echo "$OUTPUT" | head -n 1 | tr -d '[:space:]')
        RESULT=${RESULT:-error}

        case "$RESULT" in
            sat) SAT=$((SAT + 1)) ;;
            unsat) UNSAT=$((UNSAT + 1)) ;;
            unknown) UNKNOWN=$((UNKNOWN + 1)) ;;
            *) ERROR_COUNT=$((ERROR_COUNT + 1)); RESULT="error" ;;
        esac
    fi

    echo "$RESULT (${WALL_TIME}s)"

    # Append to CSV
    echo "$onnx_name,$vnnlib_name,$RESULT,$WALL_TIME" >> "$OUTPUT_CSV"

    # Accumulate total time
    TOTAL_TIME=$(echo "$TOTAL_TIME + $WALL_TIME" | bc)
done

# Print summary
SOLVED=$((SAT + UNSAT))
echo ""
echo "================================================================================"
echo "BENCHMARK SUMMARY"
echo "================================================================================"
echo "Total instances: $TOTAL"
printf "sat:     %3d (%.1f%%)\n" $SAT $(echo "scale=1; 100*$SAT/$TOTAL" | bc)
printf "unsat:   %3d (%.1f%%)\n" $UNSAT $(echo "scale=1; 100*$UNSAT/$TOTAL" | bc)
printf "unknown: %3d (%.1f%%)\n" $UNKNOWN $(echo "scale=1; 100*$UNKNOWN/$TOTAL" | bc)
printf "timeout: %3d (%.1f%%)\n" $TIMEOUT_COUNT $(echo "scale=1; 100*$TIMEOUT_COUNT/$TOTAL" | bc)
printf "error:   %3d (%.1f%%)\n" $ERROR_COUNT $(echo "scale=1; 100*$ERROR_COUNT/$TOTAL" | bc)
echo "----------------------------------------"
printf "Solved:  %3d (%.1f%%)\n" $SOLVED $(echo "scale=1; 100*$SOLVED/$TOTAL" | bc)
printf "Total time: %.1fs (%.1fm)\n" $TOTAL_TIME $(echo "scale=1; $TOTAL_TIME/60" | bc)
echo "================================================================================"
