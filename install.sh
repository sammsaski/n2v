#!/bin/bash
# Installation script for n2v (NNV-Python)

set -e  # Exit on error

echo "=================================================="
echo "Installing n2v (NNV-Python)"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Must be run from the n2v directory (where pyproject.toml is)"
    exit 1
fi

# Check if git submodules are initialized
if [ ! -f "third_party/onnx2torch/pyproject.toml" ]; then
    echo "Initializing git submodules..."
    git submodule update --init --recursive
    echo "✓ Submodules initialized"
    echo ""
fi

# Install onnx2torch from submodule
echo "Installing onnx2torch from submodule..."
pip install -e third_party/onnx2torch
echo "✓ onnx2torch installed"
echo ""

# Install n2v with all optional dependencies in editable mode
echo "Installing n2v with all dependencies..."
pip install -e ".[all]"
echo "✓ n2v installed"
echo ""

# Verify installation
echo "Verifying installation..."
python -c "import n2v; print(f'n2v version: {n2v.__version__}')"
python -c "import onnx2torch; print(f'onnx2torch: {onnx2torch.__file__}')"
python -c "
from n2v.utils.lpsolver import _HAS_HIGHSPY
print(f'highspy (direct HiGHS API): {_HAS_HIGHSPY}')
"
echo ""

echo "=================================================="
echo "Installation complete!"
echo "=================================================="
echo ""
echo "Usage:"
echo "  import n2v"
echo "  from n2v.sets import Star, Zono"
echo ""
echo "Install modes:"
echo "  pip install .              # core only"
echo "  pip install '.[accel]'     # + HiGHS speedup"
echo "  pip install '.[dev]'       # + testing tools"
echo "  pip install '.[all]'       # everything"
echo ""
