#!/bin/bash
#
# VNN-COMP Smoke Test
#
# Runs 1 instance from each benchmark to check compatibility.
# Uses bash timeout to enforce per-instance time limits.
# Picks instances where NNV got a definitive result (sat/unsat) when possible,
# so results can be compared afterwards.
#
# Usage:
#   ./smoke_test.sh <benchmarks_root> [OPTIONS]
#
# Options:
#   --timeout N        Fallback timeout if instance has none (default: 120)
#   --python PATH      Python interpreter to use (default: python)
#   --output FILE      Output CSV file (default: smoke_test_results.csv)
#   --nnv-csv FILE     NNV results.csv — used to pick instances with definitive results

set -e

# Defaults
TIMEOUT=120
PYTHON=python
OUTPUT_CSV="smoke_test_results.csv"
NNV_CSV=""
WORKERS=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for benchmarks root
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <benchmarks_root> [OPTIONS]"
    exit 1
fi

BENCH_ROOT="$1"
shift

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --timeout)  TIMEOUT="$2"; shift 2 ;;
        --python)   PYTHON="$2"; shift 2 ;;
        --output)   OUTPUT_CSV="$2"; shift 2 ;;
        --nnv-csv)  NNV_CSV="$2"; shift 2 ;;
        --workers)  WORKERS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 <benchmarks_root> [OPTIONS]"
            echo "  --timeout N        Timeout per instance (default: 120)"
            echo "  --python PATH      Python interpreter (default: python)"
            echo "  --output FILE      Output CSV (default: smoke_test_results.csv)"
            echo "  --nnv-csv FILE     NNV results.csv for instance selection"
            echo "  --workers N        Parallel LP workers (default: CPU count)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! -d "$BENCH_ROOT" ]]; then
    echo "Error: $BENCH_ROOT is not a directory"
    exit 1
fi

# Verify python works
if ! "$PYTHON" -c "import n2v" 2>/dev/null; then
    echo "Error: '$PYTHON' cannot import n2v. Use --python to specify the correct interpreter."
    exit 1
fi

# Pick the best instance for a benchmark:
# If NNV CSV is provided, find the first instance where NNV got sat or unsat.
# Otherwise, fall back to the first instance in instances.csv.
# Prints: onnx_rel,vnnlib_rel,timeout
pick_instance() {
    local bench_name="$1"
    local instances_csv="$2"

    # Try NNV definitive result first
    if [[ -n "$NNV_CSV" ]] && [[ -f "$NNV_CSV" ]]; then
        local match
        match=$(grep "^${bench_name}," "$NNV_CSV" | grep -E ',sat,|,unsat,' | head -1)
        if [[ -n "$match" ]]; then
            local onnx_path vnnlib_path
            onnx_path=$(echo "$match" | cut -d, -f2)
            vnnlib_path=$(echo "$match" | cut -d, -f3)
            # Extract relative paths (strip benchmark prefix)
            local onnx_rel vnnlib_rel
            onnx_rel=$(echo "$onnx_path" | sed "s|.*benchmarks/${bench_name}/||")
            vnnlib_rel=$(echo "$vnnlib_path" | sed "s|.*benchmarks/${bench_name}/||")
            # Look up timeout from instances.csv
            local onnx_base vnnlib_base inst_line inst_timeout
            onnx_base=$(basename "$onnx_rel")
            vnnlib_base=$(basename "$vnnlib_rel")
            inst_line=$(grep "$onnx_base" "$instances_csv" | grep "$vnnlib_base" | head -1)
            inst_timeout=$(echo "$inst_line" | cut -d, -f3 | tr -d '[:space:]')
            inst_timeout=${inst_timeout:-$TIMEOUT}
            # Truncate to integer
            inst_timeout=${inst_timeout%.*}
            echo "${onnx_rel},${vnnlib_rel},${inst_timeout}"
            return
        fi
    fi

    # Fall back to first instance
    local first_line
    first_line=$(grep -v '^\s*#' "$instances_csv" | grep -v '^\s*$' | head -1)
    if [[ -n "$first_line" ]]; then
        local onnx_rel vnnlib_rel inst_timeout
        onnx_rel=$(echo "$first_line" | cut -d, -f1 | tr -d ' ')
        vnnlib_rel=$(echo "$first_line" | cut -d, -f2 | tr -d ' ')
        inst_timeout=$(echo "$first_line" | cut -d, -f3 | tr -d '[:space:]')
        inst_timeout=${inst_timeout:-$TIMEOUT}
        inst_timeout=${inst_timeout%.*}
        # Strip leading ./
        onnx_rel="${onnx_rel#./}"
        vnnlib_rel="${vnnlib_rel#./}"
        echo "${onnx_rel},${vnnlib_rel},${inst_timeout}"
    fi
}

# Collect benchmark directories
BENCHMARKS=($(ls -d "$BENCH_ROOT"/*/ 2>/dev/null | sort))
TOTAL=${#BENCHMARKS[@]}

echo "================================================================================"
echo "VNN-COMP Smoke Test"
echo "================================================================================"
echo "Benchmarks root: $BENCH_ROOT"
echo "Total benchmarks: $TOTAL"
echo "Fallback timeout: ${TIMEOUT}s (per-instance timeouts from instances.csv)"
echo "Python: $PYTHON"
echo "Workers: ${WORKERS:-auto (CPU count)}"
echo "Output: $OUTPUT_CSV"
if [[ -n "$NNV_CSV" ]]; then
    echo "NNV CSV: $NNV_CSV (for instance selection)"
fi
echo "================================================================================"
echo ""

# Initialize CSV
echo "benchmark,onnx,vnnlib,result,time,error" > "$OUTPUT_CSV"

# Statistics
SAT=0
UNSAT=0
UNKNOWN=0
TIMEOUT_COUNT=0
ERROR_COUNT=0
SKIP_COUNT=0

COUNT=0
for bench_dir in "${BENCHMARKS[@]}"; do
    COUNT=$((COUNT + 1))
    bench_name=$(basename "$bench_dir")
    instances_csv="$bench_dir/instances.csv"

    # Skip if no instances.csv
    if [[ ! -f "$instances_csv" ]]; then
        echo "[$COUNT/$TOTAL] $bench_name ... SKIP (no instances.csv)"
        echo "$bench_name,,,skip,0,no instances.csv" >> "$OUTPUT_CSV"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    # Pick instance
    picked=$(pick_instance "$bench_name" "$instances_csv")
    if [[ -z "$picked" ]]; then
        echo "[$COUNT/$TOTAL] $bench_name ... SKIP (no instance found)"
        echo "$bench_name,,,skip,0,no instance found" >> "$OUTPUT_CSV"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    onnx_rel=$(echo "$picked" | cut -d, -f1)
    vnnlib_rel=$(echo "$picked" | cut -d, -f2)
    INST_TIMEOUT=$(echo "$picked" | cut -d, -f3)
    INST_TIMEOUT=${INST_TIMEOUT:-$TIMEOUT}

    onnx_full="$bench_dir/$onnx_rel"
    vnnlib_full="$bench_dir/$vnnlib_rel"

    onnx_name=$(basename "$onnx_rel")
    vnnlib_name=$(basename "$vnnlib_rel")

    # Decompress if needed
    if [[ ! -f "$onnx_full" ]] && [[ -f "${onnx_full}.gz" ]]; then
        gunzip -k "${onnx_full}.gz" 2>/dev/null || true
    fi
    if [[ ! -f "$vnnlib_full" ]] && [[ -f "${vnnlib_full}.gz" ]]; then
        gunzip -k "${vnnlib_full}.gz" 2>/dev/null || true
    fi

    # Check files exist
    if [[ ! -f "$onnx_full" ]]; then
        echo "[$COUNT/$TOTAL] $bench_name ... ERROR (missing $onnx_rel)"
        echo "$bench_name,$onnx_name,$vnnlib_name,error,0,missing onnx: $onnx_rel" >> "$OUTPUT_CSV"
        ERROR_COUNT=$((ERROR_COUNT + 1))
        continue
    fi
    if [[ ! -f "$vnnlib_full" ]]; then
        echo "[$COUNT/$TOTAL] $bench_name ... ERROR (missing $vnnlib_rel)"
        echo "$bench_name,$onnx_name,$vnnlib_name,error,0,missing vnnlib: $vnnlib_rel" >> "$OUTPUT_CSV"
        ERROR_COUNT=$((ERROR_COUNT + 1))
        continue
    fi

    echo -n "[$COUNT/$TOTAL] $bench_name ($onnx_name + $vnnlib_name) [${INST_TIMEOUT}s] ... "

    # Run with bash timeout — captures both stdout and stderr
    INST_START=$(date +%s%N)
    EXIT_CODE=0
    WORKER_ARGS=""
    if [[ -n "$WORKERS" ]]; then
        WORKER_ARGS="--workers $WORKERS"
    fi
    OUTPUT=$(timeout --kill-after=10 "$INST_TIMEOUT" \
        "$PYTHON" "$SCRIPT_DIR/run_instance.py" \
        "$onnx_full" "$vnnlib_full" \
        --category "$bench_name" \
        $WORKER_ARGS \
        2>&1) || EXIT_CODE=$?
    INST_END=$(date +%s%N)

    WALL_TIME=$(echo "scale=1; ($INST_END - $INST_START) / 1000000000" | bc)

    if [[ $EXIT_CODE -eq 124 ]] || [[ $EXIT_CODE -eq 137 ]]; then
        RESULT="timeout"
        WALL_TIME="$INST_TIMEOUT"
        ERR_MSG="exceeded ${INST_TIMEOUT}s"
        TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
    else
        # First line of stdout is the result
        RESULT=$(echo "$OUTPUT" | grep -E '^(sat|unsat|unknown|error|timeout)$' | head -1)
        RESULT=${RESULT:-error}

        # Capture error message
        ERR_MSG=""
        if [[ "$RESULT" == "error" ]]; then
            ERR_MSG=$(echo "$OUTPUT" | grep -iE '(error|exception|traceback|not implemented|not supported)' | tail -1 | head -c 200)
            ERR_MSG=${ERR_MSG:-"unknown error (exit code $EXIT_CODE)"}
        fi

        case "$RESULT" in
            sat)     SAT=$((SAT + 1)) ;;
            unsat)   UNSAT=$((UNSAT + 1)) ;;
            unknown) UNKNOWN=$((UNKNOWN + 1)) ;;
            *)       ERROR_COUNT=$((ERROR_COUNT + 1)); RESULT="error" ;;
        esac
    fi

    # Build display string
    DISPLAY="$RESULT (${WALL_TIME}s)"
    if [[ -n "$ERR_MSG" ]]; then
        ERR_SHORT=$(echo "$ERR_MSG" | tr ',' ';' | tr '\n' ' ' | head -c 80)
        DISPLAY="$DISPLAY — $ERR_SHORT"
    fi
    echo "$DISPLAY"

    # Sanitize error message for CSV
    ERR_MSG_CSV=$(echo "$ERR_MSG" | tr ',' ';' | tr '\n' ' ' | head -c 200)
    echo "$bench_name,$onnx_name,$vnnlib_name,$RESULT,$WALL_TIME,$ERR_MSG_CSV" >> "$OUTPUT_CSV"
done

# Summary
SOLVED=$((SAT + UNSAT))
echo ""
echo "================================================================================"
echo "SMOKE TEST SUMMARY"
echo "================================================================================"
printf "  sat:     %3d\n" $SAT
printf "  unsat:   %3d\n" $UNSAT
printf "  unknown: %3d\n" $UNKNOWN
printf "  timeout: %3d\n" $TIMEOUT_COUNT
printf "  error:   %3d\n" $ERROR_COUNT
printf "  skip:    %3d\n" $SKIP_COUNT
echo "  ----------------------------------------"
printf "  solved:  %3d / %d\n" $SOLVED $TOTAL
echo "================================================================================"
