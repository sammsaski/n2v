#!/bin/bash
# VNN-COMP 2026 prepare_instance.sh for n2v.
#
# Args: v1 CATEGORY ONNX VNNLIB
# Runs before each instance. Kills stray processes from a previous run so
# the instance starts from a clean slate. No per-instance preparation is
# needed (n2v loads the model at run time).

VERSION_STRING="v1"

if [ "$1" != "${VERSION_STRING}" ]; then
    echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
    exit 1
fi

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4

echo "Preparing n2v for category '$CATEGORY' (onnx '$ONNX_FILE', vnnlib '$VNNLIB_FILE')"

# Kill any zombie workers left over from a prior instance.
killall -q python3 || true

exit 0
