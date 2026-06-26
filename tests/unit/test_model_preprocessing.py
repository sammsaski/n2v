"""
Tests for model preprocessing utilities (BatchNorm fusion, softmax stripping).
"""

import copy
import pytest
import torch
import torch.nn as nn
import torch.fx

from n2v.utils.model_preprocessing import (
    fuse_batchnorm,
    has_batchnorm,
    has_softmax,
    strip_final_softmax,
)


def _set_bn_nontrivial_stats(bn):
    """Set non-default running stats on a BatchNorm layer for meaningful testing."""
    with torch.no_grad():
        bn.running_mean.uniform_(-2.0, 2.0)
        bn.running_var.uniform_(0.5, 3.0)
        if bn.weight is not None:
            bn.weight.uniform_(0.5, 2.0)
        if bn.bias is not None:
            bn.bias.uniform_(-1.0, 1.0)


class TestFuseBatchNormConv2d:
    """Tests for Conv2d + BatchNorm2d fusion."""

    def test_conv_bn_fusion_matches_original(self):
        """Fused Conv2d+BN2d model produces identical outputs (atol=1e-5)."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        _set_bn_nontrivial_stats(model[1])
        model.eval()

        x = torch.randn(2, 3, 8, 8)
        expected = model(x)

        fused = fuse_batchnorm(model)
        actual = fused(x)

        assert torch.allclose(expected, actual, atol=1e-5), (
            f"Max diff: {(expected - actual).abs().max().item()}"
        )

    def test_conv_bn_fusion_removes_batchnorm(self):
        """No BatchNorm layers remain after fusion."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        _set_bn_nontrivial_stats(model[1])
        model.eval()

        fused = fuse_batchnorm(model)
        assert not has_batchnorm(fused), "Fused model still contains BatchNorm layers"

    def test_conv_bn_no_bias(self):
        """Fusion works when Conv2d has bias=False."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        _set_bn_nontrivial_stats(model[1])
        model.eval()

        x = torch.randn(2, 3, 8, 8)
        expected = model(x)

        fused = fuse_batchnorm(model)
        actual = fused(x)

        assert torch.allclose(expected, actual, atol=1e-5), (
            f"Max diff: {(expected - actual).abs().max().item()}"
        )
        # Fused conv should now have a bias
        fused_conv = list(fused.children()).__iter__().__next__()
        assert fused_conv.bias is not None, "Fused conv should have a bias after fusion"


class TestFuseBatchNormLinear:
    """Tests for Linear + BatchNorm1d fusion."""

    def test_linear_bn_fusion_matches_original(self):
        """Fused Linear+BN1d model produces identical outputs (atol=1e-5)."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(20, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )
        _set_bn_nontrivial_stats(model[1])
        model.eval()

        x = torch.randn(4, 20)
        expected = model(x)

        fused = fuse_batchnorm(model)
        actual = fused(x)

        assert torch.allclose(expected, actual, atol=1e-5), (
            f"Max diff: {(expected - actual).abs().max().item()}"
        )
        assert not has_batchnorm(fused), "Fused model still contains BatchNorm layers"


class TestFuseBatchNormEdgeCases:
    """Edge case tests for BatchNorm fusion."""

    def test_no_batchnorm_returns_copy(self):
        """Model without BN returns a deep copy (different object, same weights)."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        model.eval()

        fused = fuse_batchnorm(model)

        # Different object
        assert fused is not model, "fuse_batchnorm should return a new object"

        # Same weights
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), fused.named_parameters()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Parameter {n1} differs"

    def test_standalone_batchnorm_kept(self):
        """BN without preceding Conv/Linear is NOT fused (remains in model)."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.BatchNorm1d(10),  # standalone, no preceding Linear
            nn.ReLU(),
            nn.Linear(10, 5),
        )
        _set_bn_nontrivial_stats(model[0])
        model.eval()

        fused = fuse_batchnorm(model)

        # The standalone BN should still be present
        assert has_batchnorm(fused), (
            "Standalone BatchNorm should remain (not fused)"
        )

        # Output should still match
        x = torch.randn(4, 10)
        expected = model(x)
        actual = fused(x)
        assert torch.allclose(expected, actual, atol=1e-5)

    def test_original_model_unchanged(self):
        """Original model weights are unchanged after fusion."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        _set_bn_nontrivial_stats(model[1])
        model.eval()

        # Save original weights
        orig_conv_weight = model[0].weight.clone()
        orig_conv_bias = model[0].bias.clone()
        orig_bn_weight = model[1].weight.clone()
        orig_bn_bias = model[1].bias.clone()
        orig_bn_mean = model[1].running_mean.clone()
        orig_bn_var = model[1].running_var.clone()

        _ = fuse_batchnorm(model)

        # Verify original is unchanged
        assert torch.equal(model[0].weight, orig_conv_weight)
        assert torch.equal(model[0].bias, orig_conv_bias)
        assert torch.equal(model[1].weight, orig_bn_weight)
        assert torch.equal(model[1].bias, orig_bn_bias)
        assert torch.equal(model[1].running_mean, orig_bn_mean)
        assert torch.equal(model[1].running_var, orig_bn_var)

    def test_nested_sequential(self):
        """Fusion works through nested Sequential modules."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            ),
        )
        _set_bn_nontrivial_stats(model[0][1])
        _set_bn_nontrivial_stats(model[1][1])
        model.eval()

        x = torch.randn(2, 3, 8, 8)
        expected = model(x)

        fused = fuse_batchnorm(model)
        actual = fused(x)

        assert torch.allclose(expected, actual, atol=1e-5), (
            f"Max diff: {(expected - actual).abs().max().item()}"
        )
        assert not has_batchnorm(fused), "Fused model still contains BatchNorm layers"

    def test_multiple_conv_bn_pairs(self):
        """Multiple Conv+BN pairs all get fused."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
        )
        _set_bn_nontrivial_stats(model[1])
        _set_bn_nontrivial_stats(model[4])
        _set_bn_nontrivial_stats(model[7])
        model.eval()

        x = torch.randn(2, 3, 8, 8)
        expected = model(x)

        fused = fuse_batchnorm(model)
        actual = fused(x)

        assert torch.allclose(expected, actual, atol=1e-5), (
            f"Max diff: {(expected - actual).abs().max().item()}"
        )
        assert not has_batchnorm(fused), "Fused model still contains BatchNorm layers"

        # Count Identity layers (should have replaced 3 BN layers)
        identity_count = sum(
            1 for m in fused.modules() if isinstance(m, nn.Identity)
        )
        assert identity_count == 3, f"Expected 3 Identity layers, got {identity_count}"


class TestHasBatchNorm:
    """Tests for the has_batchnorm helper."""

    def test_model_with_batchnorm(self):
        model = nn.Sequential(nn.Linear(10, 10), nn.BatchNorm1d(10))
        assert has_batchnorm(model) is True

    def test_model_without_batchnorm(self):
        model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        assert has_batchnorm(model) is False

    def test_nested_batchnorm(self):
        model = nn.Sequential(
            nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8)),
            nn.ReLU(),
        )
        assert has_batchnorm(model) is True


class _SoftmaxNet(nn.Module):
    """Small MLP ending in nn.Softmax, used to exercise the fx GraphModule path."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.fc2(torch.relu(self.fc1(x))))


class _FunctionalSoftmaxNet(nn.Module):
    """MLP ending in a *functional* softmax (call_function, not a module).

    Traces to a graph whose output is produced by a torch.nn.functional.softmax
    call rather than an nn.Softmax module.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        return torch.nn.functional.softmax(self.fc2(torch.relu(self.fc1(x))), dim=-1)


class _MethodSoftmaxNet(nn.Module):
    """MLP ending in a tensor-method softmax (``x.softmax(-1)``).

    Traces to a graph whose output is produced by a call_method node (target
    'softmax') rather than a module or a call_function.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).softmax(dim=-1)


class TestStripFinalSoftmax:
    """Tests for strip_final_softmax (Sequential and fx GraphModule paths)."""

    def test_sequential_removes_softmax(self):
        """Trailing nn.Softmax is dropped from an nn.Sequential model."""
        torch.manual_seed(0)

        model = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3), nn.Softmax(dim=-1)
        )
        model.eval()

        stripped = strip_final_softmax(model)

        assert not has_softmax(stripped), "Stripped model still contains Softmax"
        assert len(stripped) == 3, "Trailing Softmax should be removed from Sequential"

    def test_sequential_outputs_logits(self):
        """Stripped Sequential outputs pre-softmax logits (re-applying softmax
        reproduces the original probabilities)."""
        torch.manual_seed(0)

        model = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3), nn.Softmax(dim=-1)
        )
        model.eval()
        x = torch.randn(5, 4)

        stripped = strip_final_softmax(model)
        logits = stripped(x)

        assert torch.allclose(torch.softmax(logits, dim=-1), model(x), atol=1e-6)

    def test_sequential_preserves_argmax_and_argmin(self):
        """Dropping softmax leaves the predicted class unchanged for both the
        argmax and argmin conventions, since softmax is order-preserving."""
        torch.manual_seed(0)

        model = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3), nn.Softmax(dim=-1)
        )
        model.eval()
        x = torch.randn(16, 4)

        stripped = strip_final_softmax(model)

        assert torch.equal(stripped(x).argmax(dim=-1), model(x).argmax(dim=-1))
        assert torch.equal(stripped(x).argmin(dim=-1), model(x).argmin(dim=-1))

    def test_graph_module_removes_softmax(self):
        """Trailing Softmax is dropped from a torch.fx.GraphModule via graph
        surgery (the orphaned submodule is replaced with Identity)."""
        torch.manual_seed(0)

        gm = torch.fx.symbolic_trace(_SoftmaxNet())
        gm.eval()
        assert isinstance(gm, torch.fx.GraphModule)
        assert has_softmax(gm)

        stripped = strip_final_softmax(gm)

        assert not has_softmax(stripped), "Stripped GraphModule still contains Softmax"

    def test_graph_module_outputs_logits(self):
        """Stripped GraphModule outputs pre-softmax logits and preserves the
        predicted class."""
        torch.manual_seed(0)

        gm = torch.fx.symbolic_trace(_SoftmaxNet())
        gm.eval()
        x = torch.randn(5, 4)

        stripped = strip_final_softmax(gm)

        assert torch.allclose(torch.softmax(stripped(x), dim=-1), gm(x), atol=1e-6)
        assert torch.equal(stripped(x).argmin(dim=-1), gm(x).argmin(dim=-1))

    def test_graph_module_removes_functional_softmax(self):
        """A trailing functional softmax (call_function) is detected and
        stripped, leaving the model outputting pre-softmax logits."""
        torch.manual_seed(0)

        gm = torch.fx.symbolic_trace(_FunctionalSoftmaxNet())
        gm.eval()
        x = torch.randn(5, 4)
        assert has_softmax(gm), "Functional softmax should be detected before stripping"

        stripped = strip_final_softmax(gm)

        assert not has_softmax(stripped), "Functional softmax should be stripped"
        assert torch.allclose(torch.softmax(stripped(x), dim=-1), gm(x), atol=1e-6)
        assert torch.equal(stripped(x).argmax(dim=-1), gm(x).argmax(dim=-1))

    def test_graph_module_removes_method_softmax(self):
        """A trailing tensor-method softmax (call_method) is detected and
        stripped, leaving the model outputting pre-softmax logits."""
        torch.manual_seed(0)

        gm = torch.fx.symbolic_trace(_MethodSoftmaxNet())
        gm.eval()
        x = torch.randn(5, 4)
        assert has_softmax(gm), "Method softmax should be detected before stripping"

        stripped = strip_final_softmax(gm)

        assert not has_softmax(stripped), "Method softmax should be stripped"
        assert torch.allclose(torch.softmax(stripped(x), dim=-1), gm(x), atol=1e-6)
        assert torch.equal(stripped(x).argmax(dim=-1), gm(x).argmax(dim=-1))


class TestStripFinalSoftmaxEdgeCases:
    """Edge case tests for strip_final_softmax."""

    def test_no_softmax_is_noop(self):
        """A model without a trailing softmax is returned unchanged in value
        as a new object."""
        torch.manual_seed(0)

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        model.eval()
        x = torch.randn(5, 4)

        stripped = strip_final_softmax(model)

        assert not has_softmax(stripped)
        assert stripped is not model, "strip_final_softmax should return a new object"
        assert torch.allclose(stripped(x), model(x), atol=1e-6)

    def test_original_model_unchanged(self):
        """The original model still contains its softmax after stripping."""
        torch.manual_seed(0)

        model = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3), nn.Softmax(dim=-1)
        )
        model.eval()

        _ = strip_final_softmax(model)

        assert has_softmax(model), "Original model should be left untouched"

    def test_idempotent(self):
        """Stripping an already-stripped model is a no-op."""
        torch.manual_seed(0)

        gm = torch.fx.symbolic_trace(_SoftmaxNet())
        gm.eval()
        x = torch.randn(5, 4)

        once = strip_final_softmax(gm)
        twice = strip_final_softmax(once)

        assert not has_softmax(twice)
        assert torch.allclose(twice(x), once(x), atol=1e-6)

    def test_non_trailing_softmax_kept(self):
        """A softmax that is not the final layer is not removed."""
        torch.manual_seed(0)

        model = nn.Sequential(
            nn.Linear(4, 8), nn.Softmax(dim=-1), nn.Linear(8, 3)
        )
        model.eval()

        stripped = strip_final_softmax(model)

        assert has_softmax(stripped), "Non-trailing Softmax should be kept"


class TestHasSoftmax:
    """Tests for the has_softmax helper."""

    def test_model_with_softmax(self):
        model = nn.Sequential(nn.Linear(10, 10), nn.Softmax(dim=-1))
        assert has_softmax(model) is True

    def test_model_without_softmax(self):
        model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        assert has_softmax(model) is False

    def test_nested_softmax(self):
        model = nn.Sequential(
            nn.Sequential(nn.Linear(10, 8), nn.Softmax(dim=-1)),
            nn.ReLU(),
        )
        assert has_softmax(model) is True

    def test_no_functional_softmax_in_graph_module(self):
        """A GraphModule without any softmax reports False."""
        gm = torch.fx.symbolic_trace(
            nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        )
        assert has_softmax(gm) is False


def _load_onnx_softmax_model(opset, op="Softmax"):
    """Build a tiny Gemm -> {Softmax|LogSoftmax} ONNX model at the given opset
    and load it through n2v's real load_onnx path.

    onnx2torch converts an ONNX Softmax/LogSoftmax at opset <= 11 to its
    OnnxSoftmaxV1V11 wrapper (not an nn.Softmax); opset >= 13 maps Softmax to
    nn.Softmax. Skips if onnx / onnx2torch are unavailable.
    """
    import os
    import tempfile

    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnx2torch")
    import numpy as np
    from onnx import TensorProto, helper

    from n2v.utils.model_loader import load_onnx

    rng = np.random.RandomState(0)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    W = helper.make_tensor(
        "W", TensorProto.FLOAT, [4, 4], rng.randn(4, 4).astype(np.float32).flatten()
    )
    B = helper.make_tensor("B", TensorProto.FLOAT, [4], rng.randn(4).astype(np.float32))
    nodes = [
        helper.make_node("Gemm", ["X", "W", "B"], ["H"]),
        helper.make_node(op, ["H"], ["Y"], axis=1),
    ]
    graph = helper.make_graph(nodes, "g", [X], [Y], initializer=[W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 7

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        path = f.name
    try:
        return load_onnx(path)
    finally:
        os.unlink(path)


class TestStripFinalSoftmaxOnnxWrapper:
    """strip_final_softmax / has_softmax cover onnx2torch's OnnxSoftmaxV1V11,
    the module an ONNX Softmax at opset <= 11 converts to. It is not an
    nn.Softmax subclass, so the plain isinstance check used to miss it.

    Exercised end-to-end through the real load_onnx -> onnx2torch path."""

    def test_graph_module_strips_opset11_softmax(self):
        """A real opset-11 ONNX Softmax (an OnnxSoftmaxV1V11 call_module in the
        onnx2torch GraphModule) is detected and stripped down to logits."""
        gm = _load_onnx_softmax_model(11, "Softmax")
        x = torch.randn(5, 4)
        assert has_softmax(gm) is True

        stripped = strip_final_softmax(gm)

        assert has_softmax(stripped) is False
        assert torch.allclose(torch.softmax(stripped(x), dim=-1), gm(x), atol=1e-5)
        assert torch.equal(stripped(x).argmax(-1), gm(x).argmax(-1))

    def test_graph_module_keeps_opset11_logsoftmax(self):
        """A real opset-11 ONNX LogSoftmax is not a plain softmax; left as-is."""
        gm = _load_onnx_softmax_model(11, "LogSoftmax")
        x = torch.randn(5, 4)
        assert has_softmax(gm) is False

        stripped = strip_final_softmax(gm)

        assert torch.allclose(stripped(x), gm(x), atol=1e-6)


def _build_opset9_upsample_model(path):
    """Write a tiny opset-9 model that fails to convert the way the upsample
    cGANs do, and return its onnxruntime reference outputs.

    The deprecated opset-9 Upsample op has no onnx2torch converter, so the
    loader upgrades the model to a modern opset. onnx.version_converter
    rewrites Upsample -> Resize but emits empty-named Constant nodes whose
    onnx2torch-generated names ("Constant", "Constant_1", ...) collide with
    the graph's explicitly-named Constant_1; the dropped node then surfaces
    as RuntimeError("Got unexpected input value type (ValueType.UNKNOWN)").
    The Reshape fed by an explicit Constant_1 and the Upsample fed by a
    Constant scales node together reproduce that collision.
    """
    import numpy as np
    from onnx import TensorProto, helper

    rng = np.random.RandomState(0)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 2, 4])
    W = helper.make_tensor(
        "W", TensorProto.FLOAT, [4, 4], rng.randn(4, 4).astype(np.float32).flatten()
    )
    B = helper.make_tensor("B", TensorProto.FLOAT, [4], rng.randn(4).astype(np.float32))
    shape_const = helper.make_node(
        "Constant", [], ["shp"], name="Constant_1",
        value=helper.make_tensor(
            "v", TensorProto.INT64, [4], np.array([1, 1, 1, 4], np.int64)),
    )
    scales_const = helper.make_node(
        "Constant", [], ["scales"], name="Constant",
        value=helper.make_tensor(
            "s", TensorProto.FLOAT, [4], np.array([1, 1, 2, 1], np.float32)),
    )
    nodes = [
        helper.make_node("Gemm", ["X", "W", "B"], ["H"]),
        shape_const,
        scales_const,
        helper.make_node("Reshape", ["H", "shp"], ["R"]),
        helper.make_node("Upsample", ["R", "scales"], ["Y"], mode="nearest"),
    ]
    graph = helper.make_graph(nodes, "g", [X], [Y], initializer=[W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 9)])
    model.ir_version = 6

    import onnx
    onnx.save(model, path)


class TestOpset9UpsampleLoadShim:
    """The opset-upgrade shim in load_onnx recovers the upsample cGANs
    (cgan_2023 / cgan2026), whose opset-9 Upsample op leaves a node-name
    collision after onnx.version_converter rewrites it to Resize. Exercised
    end-to-end through the real load_onnx -> onnx2torch path. (Issue #48)"""

    def test_loads_and_matches_onnxruntime(self, tmp_path):
        import warnings

        pytest.importorskip("onnx")
        pytest.importorskip("onnx2torch")
        pytest.importorskip("onnxsim")
        ort = pytest.importorskip("onnxruntime")
        import numpy as np

        from n2v.utils.model_loader import load_onnx

        path = str(tmp_path / "upsample9.onnx")
        _build_opset9_upsample_model(path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = load_onnx(path)

        x = np.random.RandomState(1).randn(1, 4).astype(np.float32)
        sess = ort.InferenceSession(path)
        ref = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        with torch.no_grad():
            out = model(torch.from_numpy(x)).numpy()

        assert out.shape == ref.shape
        assert np.allclose(out, ref, atol=1e-5)

    def test_upgradeable_conversion_error_discriminates(self):
        from n2v.utils.model_loader import _is_upgradeable_conversion_error

        assert _is_upgradeable_conversion_error(
            NotImplementedError("Converter is not implemented (Upsample, version=9)"))
        assert _is_upgradeable_conversion_error(
            RuntimeError("Got unexpected input value type (ValueType.UNKNOWN)"))
        # An unrelated failure must propagate, not trigger an opset upgrade.
        assert not _is_upgradeable_conversion_error(RuntimeError("some other error"))
