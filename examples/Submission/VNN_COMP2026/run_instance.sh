#!/bin/bash
# VNN-COMP 2026 run_instance.sh for n2v.
#
# Args: v1 CATEGORY ONNX VNNLIB RESULTS_FILE TIMEOUT
# Runs n2v on one instance and writes the result (sat|unsat|unknown|timeout,
# plus the counterexample for sat) into RESULTS_FILE.

VERSION_STRING="v1"

if [ "$1" != "${VERSION_STRING}" ]; then
    echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
    exit 1
fi

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4
RESULTS_FILE=$5
TIMEOUT=$6

echo "Running n2v on category '$CATEGORY' (onnx '$ONNX_FILE', vnnlib '$VNNLIB_FILE', results '$RESULTS_FILE', timeout '$TIMEOUT')"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$DIR/vnncomp_runner.py" "$CATEGORY" "$ONNX_FILE" "$VNNLIB_FILE" "$RESULTS_FILE" "$TIMEOUT"
