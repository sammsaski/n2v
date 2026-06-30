#!/bin/bash
# VNN-COMP 2026 installation script for n2v (NNV-Python).
# Installs into the machine's system python3 via pip (the dedicated competition
# machine). Local-safe mode: run inside a venv or with N2V_LOCAL=1 to skip
# sudo/apt and install into the active interpreter (safe on a shared machine).
#
# Usage: install_tool.sh v1   (or: N2V_LOCAL=1 install_tool.sh v1)

set -e

TOOL_NAME="n2v"
VERSION_STRING="v1"

if [ "$1" != "${VERSION_STRING}" ]; then
    echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
    exit 1
fi

echo "Installing $TOOL_NAME dependencies"

# This script lives at the repo root, so the repo root is its own directory.
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
echo "Repo root: $REPO_ROOT"

# Local-safe mode: when run inside a venv (VIRTUAL_ENV set) or with N2V_LOCAL=1,
# skip the system apt/sudo steps and install into the ACTIVE interpreter -- so a
# local dry-run on a shared machine never touches system python or runs sudo. The
# dedicated competition machine (no venv, no N2V_LOCAL) does the full system
# install as before. (The pip steps below already use `python3 -m pip`, which
# targets the active venv when one is active.)
if [ -n "$VIRTUAL_ENV" ] || [ "$N2V_LOCAL" = "1" ]; then
    echo "Local mode: skipping sudo apt-get; installing into $(command -v python3)."
else
    # System packages (best-effort; ignore failures on machines without apt/sudo).
    sudo apt-get update -y || true
    sudo apt-get install -y python3-pip git || true
fi

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
