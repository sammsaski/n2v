"""LP-free CROWN verification of the VNN-COMP 2023 ViT (benchmark entry point).

The implementation now lives in the n2v package (:mod:`n2v.nn.vit_crown`) so the
VNN-COMP runner can import it. This module re-exports it for the benchmark scripts
and tests (which build the model via the local ``model.py`` and lower it here).
"""
from n2v.nn.vit_crown import (   # noqa: F401
    MEAN, STD, norm_img, eps_box, margin_spec, to_ops,
    crown_margins, verify_instance, verify_halfspace_group_safe,
    load_vit_onnx,
)
