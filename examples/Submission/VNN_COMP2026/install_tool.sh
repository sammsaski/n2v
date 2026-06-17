#!/bin/bash
# VNN-COMP 2026 installation script for n2v (NNV-Python).
# Installs into the machine's system python3 via pip (no conda / venv).
#
# Usage: install_tool.sh v1

set -e

TOOL_NAME="n2v"
VERSION_STRING="v1"

if [ "$1" != "${VERSION_STRING}" ]; then
    echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
    exit 1
fi

echo "Installing $TOOL_NAME dependencies"

# Repo root is three levels up from this script
#   (examples/Submission/VNN_COMP2026/ -> repo root).
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
echo "Repo root: $REPO_ROOT"

# System packages (best-effort; ignore failures on machines without apt/sudo).
sudo apt-get update -y || true
sudo apt-get install -y python3-pip git || true

# Initialize the onnx2torch submodule (model loader depends on it).
git submodule update --init --recursive

python3 -m pip install --upgrade pip

# Core + verification dependencies.
python3 -m pip install -r requirements.txt \
    || python3 -m pip install numpy scipy torch cvxpy networkx torchdiffeq onnx onnxruntime

# onnx2torch is installed from the submodule (a fork), not PyPI.
python3 -m pip install -e third_party/onnx2torch

# Install n2v itself (editable).
python3 -m pip install -e .

echo "Verifying installation..."
python3 -c "import n2v, onnx2torch, onnx, torch; print('n2v', n2v.__version__)"

echo "Installation of $TOOL_NAME complete."
