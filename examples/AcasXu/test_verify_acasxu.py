import os
import sys
import time
import numpy as np
from pathlib import Path
import onnx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nnv_py.sets import Star
from nnv_py.nn.reach.reach_star import reach_star_exact, reach_star_approx
from nnv_py.utils import load_vnnlib, verify_specification, load_onnx

script_dir = Path(__file__).parent

# Example: Verify property 1 with network 1_1
network_file = script_dir / "onnx" / "ACASXU_run2a_1_1_batch_2000.onnx"
property_file = script_dir / "vnnlib" / "prop_1.vnnlib"

# Load network
print("\n1. Loading network...")
net = load_onnx(network_file)