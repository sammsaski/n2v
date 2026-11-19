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

    import onnx
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

        import onnxruntime as ort
        pytest.importorskip("onnx2torch")
        pytest.importorskip("onnxsim")
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


# --------------------------------------------------------------------------- #
# has_* on every form, the state fusion and stripping leave, modules that     #
# are not fused, dtypes.                                                      #
# --------------------------------------------------------------------------- #
from tests.known_bugs import known_bug  # noqa: E402

_EPS = 0.75  # with running variances 1/4 and 13/4 the fold is exact


def _dyadic_bn(bn):
    """Dyadic statistics whose fold is exact: var + eps is 1 or 4."""
    n = bn.num_features
    with torch.no_grad():
        bn.running_mean.copy_(torch.arange(n) / 2.0 - 1.0)
        bn.running_var.copy_(torch.tensor([0.25, 3.25] * n)[:n])
        if bn.weight is not None:
            bn.weight.copy_(torch.tensor([2.0, -0.5] * n)[:n])
            bn.bias.copy_(torch.arange(n) / 4.0)
    return bn


def _dyadic_linear(fc):
    with torch.no_grad():
        fc.weight.copy_(torch.arange(fc.weight.numel()).reshape(
            fc.weight.shape) % 7 / 4.0 - 0.75)
        fc.bias.copy_(torch.arange(fc.bias.numel()) / 2.0 - 0.5)
    return fc


class _Block(nn.Module):
    """A custom module holding a Linear and its BatchNorm1d."""

    def __init__(self):
        super().__init__()
        self.fc = _dyadic_linear(nn.Linear(3, 4))
        self.bn = _dyadic_bn(nn.BatchNorm1d(4, eps=_EPS))

    def forward(self, x):
        return self.bn(self.fc(x))


class _HoldsABlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _Block()
        self.head = _dyadic_linear(nn.Linear(4, 2))

    def forward(self, x):
        return self.head(torch.relu(self.block(x)))


class _Untraceable(nn.Module):
    """Data-dependent control flow: torch.fx cannot trace it."""

    def __init__(self):
        super().__init__()
        self.fc = _dyadic_linear(nn.Linear(3, 4))
        self.bn = _dyadic_bn(nn.BatchNorm1d(4, eps=_EPS))

    def forward(self, x):
        y = self.bn(self.fc(x))
        if y.sum() > 0:
            return y
        return -y


def _dyadic_input(rows=8, cols=3):
    return (torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
            % 9 - 4.0) / 2.0


def _module_kinds(model):
    return {type(m) for m in model.modules()}


def _traced_call(form):
    """A traced module whose forward is ``form(x)``."""
    class _Net(nn.Module):
        def forward(self, x):
            return form(x)
    return torch.fx.symbolic_trace(_Net())


class TestHasBatchNormForms:
    """has_batchnorm decides whether reach fuses; every form must be found."""

    def test_every_batchnorm_kind_at_any_depth(self):
        """BatchNorm1d, BatchNorm2d and BatchNorm3d, a model that is itself a
        BatchNorm, one two containers deep and a subclass are found; layers
        with a similar role (LayerNorm, GroupNorm, InstanceNorm2d) are not.
        Catches a kind left out of the checked classes and a search of the
        direct children only."""
        class _MyBatchNorm(nn.BatchNorm2d):
            pass

        found = [nn.BatchNorm1d(3), nn.BatchNorm2d(3), nn.BatchNorm3d(3),
                 nn.Sequential(nn.Conv3d(1, 2, 1), nn.BatchNorm3d(2)),
                 nn.Sequential(nn.Sequential(nn.ReLU(), nn.Sequential(
                     nn.Linear(2, 2), nn.BatchNorm1d(2)))),
                 nn.Sequential(nn.Conv2d(1, 2, 1), _MyBatchNorm(2)),
                 _HoldsABlock()]
        for model in found:
            assert has_batchnorm(model) is True, model
        for model in (nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2)),
                      nn.Sequential(nn.Conv2d(2, 2, 1), nn.GroupNorm(1, 2)),
                      nn.Sequential(nn.Conv2d(2, 2, 1), nn.InstanceNorm2d(2))):
            assert has_batchnorm(model) is False, model


class TestHasSoftmaxForms:
    """has_softmax on every form of a softmax and of a log-softmax."""

    def test_every_softmax_call_form_is_found(self):
        """``torch.softmax``, ``F.softmax`` with its input by keyword and the
        tensor method ``.softmax`` in a traced graph count as softmax.
        Catches a call form that is not recognized (the method form, a match
        on the function object instead of its name)."""
        forms = [lambda z: torch.softmax(z, -1),
                 lambda z: torch.nn.functional.softmax(input=z, dim=-1),
                 lambda z: z.softmax(-1)]
        for form in forms:
            assert has_softmax(_traced_call(form)) is True

    def test_log_softmax_forms_are_not_softmax(self):
        """A LogSoftmax is not a plain softmax in any form: an nn.LogSoftmax
        in a Sequential or a traced graph, ``F.log_softmax`` and the tensor
        method ``.log_softmax``. Catches a LogSoftmax module accepted as a
        softmax and a call matched by a name that contains 'softmax'
        (stripping either would turn log-probabilities into logits)."""
        forms = [lambda z: torch.nn.functional.log_softmax(z, -1),
                 lambda z: z.log_softmax(-1)]
        models = [nn.Sequential(nn.Linear(3, 3), nn.LogSoftmax(dim=-1)),
                  torch.fx.symbolic_trace(
                      nn.Sequential(nn.Linear(3, 3), nn.LogSoftmax(dim=-1)))]
        models += [_traced_call(form) for form in forms]
        for model in models:
            assert has_softmax(model) is False, model


class TestFuseBatchNormState:
    """fuse_batchnorm works on a copy and returns it in eval mode."""

    def test_callers_graph_module_is_left_unchanged(self):
        """A traced GraphModule in training mode: after fusion the caller's
        model has the same module classes, training flags, hooks, parameters,
        buffers and graph code, and the fused copy is another object. Catches
        a fusion that rewrites the caller's graph in place (its BatchNorm
        replaced by an identity, its Linear by a float64 layer with hooks)
        or switches the caller's model to eval mode."""
        gm = torch.fx.symbolic_trace(_HoldsABlock()).train()

        def view(m):
            return ([(n, type(x), x.training, len(x._forward_hooks),
                      len(x._forward_pre_hooks))
                     for n, x in m.named_modules()],
                    {k: v.clone() for k, v in m.state_dict().items()}, m.code)

        before = view(gm)
        fused = fuse_batchnorm(gm)
        after = view(gm)
        assert fused is not gm
        assert after[0] == before[0], "module classes, flags or hooks changed"
        assert after[2] == before[2], "the caller's graph was rewritten"
        assert all(torch.equal(v, after[1][k]) for k, v in before[1].items())
        assert not has_batchnorm(fused)

    def test_result_is_in_eval_mode(self):
        """From a model in training mode (a fused pair, a standalone
        BatchNorm, a Dropout), every module of the result is in eval mode, on
        the Sequential path and on the traced path; the docstring promises
        eval mode. Catches the final eval() left out (the rebuilt Sequential
        and the new fused layer and identity come back in training mode)."""
        for model in (nn.Sequential(
                _dyadic_linear(nn.Linear(3, 4)),
                _dyadic_bn(nn.BatchNorm1d(4, eps=_EPS)), nn.ReLU(),
                nn.Dropout(0.5), _dyadic_bn(nn.BatchNorm1d(4, eps=_EPS))),
                _HoldsABlock()):
            fused = fuse_batchnorm(model.train())
            training = [n for n, m in fused.named_modules() if m.training]
            assert not training, f"modules in training mode: {training}"
            assert model.training, "the caller's model was switched to eval"

    def test_traced_model_in_training_mode_is_fused_for_its_eval_function(
            self):
        """A custom module, which the fusion traces, handed over in training
        mode, whose forward calls ``F.dropout(..., self.training)``: tracing
        reads the training flag as a constant, so the fusion must switch its
        copy to eval mode before it traces. The fused model computes the
        eval-mode function exactly on dyadic data, on two calls in a row, and
        the caller's model stays in training mode. Catches the eval() before
        the trace left out (the fused graph then keeps a dropout with
        training=True and returns random outputs, which the final eval()
        cannot undo)."""
        class _Dropping(_HoldsABlock):
            def forward(self, x):
                return self.head(torch.nn.functional.dropout(
                    torch.relu(self.block(x)), 0.5, self.training))

        model = _Dropping().train()
        x = _dyadic_input().double()
        with torch.no_grad():
            expected = copy.deepcopy(model).eval().double()(x)
            fused = fuse_batchnorm(model).double()
            assert nn.BatchNorm1d not in _module_kinds(fused)
            for _ in range(2):
                assert torch.equal(fused(x), expected)
        assert model.training


class TestFuseBatchNormLeavesUnfused:
    """Modules fuse_batchnorm must return unfused, with their function."""

    def test_untraceable_module_is_returned_unfused(self):
        """A module with data-dependent control flow cannot be traced: it is
        returned unfused (its BatchNorm is reached as a standalone layer) and
        computes what it computed, and the reach refuses it with the
        deliberate TypeError naming tracing. Catches the tracing failure let
        through (fusion then raises torch.fx's TraceError, a ValueError)."""
        from n2v.nn.reach import reach_pytorch_model
        from n2v.sets import Star
        import numpy as np

        model = _Untraceable().eval()
        fused = fuse_batchnorm(model)
        assert nn.BatchNorm1d in _module_kinds(fused)
        x = _dyadic_input()
        with torch.no_grad():
            assert torch.equal(fused(x), model(x))
        star = Star.from_bounds(-np.ones((3, 1)), np.ones((3, 1)))
        with pytest.raises(TypeError, match="traceable"):
            reach_pytorch_model(model, star, method="approx")

    def test_batchnorm_without_running_statistics_is_kept(self):
        """A BatchNorm built with track_running_stats=False normalizes by
        batch statistics even in eval mode, so it is not fused: fusion
        returns the model with that BatchNorm and the same function on a
        batch of 8, after a Linear and after a Conv2d. Catches the
        running-statistics check dropped (the fold then raises
        AttributeError on the missing running variance)."""
        for model, x in (
                (nn.Sequential(nn.Linear(3, 4),
                               nn.BatchNorm1d(4, track_running_stats=False)),
                 _dyadic_input()),
                (nn.Sequential(nn.Conv2d(2, 3, 2),
                               nn.BatchNorm2d(3, track_running_stats=False)),
                 torch.arange(8 * 2 * 3 * 3, dtype=torch.float32).reshape(
                     8, 2, 3, 3) % 5 / 4.0)):
            model.eval()
            fused = fuse_batchnorm(model)
            kept = [m for m in fused.modules()
                    if isinstance(m, nn.modules.batchnorm._BatchNorm)]
            assert len(kept) == 1 and kept[0].running_mean is None
            with torch.no_grad():
                assert torch.equal(fused(x), model(x))

    def test_pair_inside_a_custom_module_is_fused(self):
        """A custom module holding Linear then BatchNorm1d, alone and as a
        child of another custom module (traced, the pair under the dotted
        name ``block.fc``): fusion removes the BatchNorm and the result
        computes the original function exactly on dyadic data. The passing
        neighbour of the marked Sequential case below. Catches a graph fusion
        that skips BatchNorm1d (only BatchNorm2d fused)."""
        for model in (_Block().eval(), _HoldsABlock().eval()):
            fused = fuse_batchnorm(model)
            assert nn.BatchNorm1d not in _module_kinds(fused)
            x = _dyadic_input()
            with torch.no_grad():
                assert torch.equal(fused.double()(x.double()),
                                   copy.deepcopy(model).double()(x.double()))

    @known_bug(12, "fuse_batchnorm does not descend into a child of a "
                   "Sequential that is neither a Sequential nor a fusable "
                   "layer, so a Linear+BatchNorm1d pair inside a custom "
                   "module held by a Sequential stays unfused, although the "
                   "docstring says every pair is fused and replaced by an "
                   "identity")
    def test_pair_inside_a_custom_child_of_a_sequential_is_fused(self):
        """Sequential(custom module holding Linear then BatchNorm1d, ReLU):
        the docstring promises every Linear+BatchNorm1d pair fused and its
        BatchNorm replaced by an identity, as for the same module alone (the
        neighbour above), with the function unchanged. Today the BatchNorm
        stays; reach then takes the standalone BatchNorm rule, which is
        sound, so the harm is the broken promise of the returned model."""
        model = nn.Sequential(_Block(), nn.ReLU()).eval()
        fused = fuse_batchnorm(model)
        assert nn.BatchNorm1d not in _module_kinds(fused)
        x = _dyadic_input().double()
        with torch.no_grad():
            assert torch.equal(fused.double()(x),
                               copy.deepcopy(model).double()(x))


class TestFuseBatchNormNames:
    """fuse_batchnorm keeps the names of a Sequential's children."""

    @known_bug(12, "fuse_batchnorm rebuilds an nn.Sequential with the child "
                   "names '0', '1', ... although _fuse_sequential says it "
                   "keeps the original keys, so a Sequential built from an "
                   "OrderedDict loses its names (fused.fc raises "
                   "AttributeError, the state_dict keys change)")
    def test_named_sequential_keeps_its_child_names(self):
        """Sequential(OrderedDict(fc=Linear, bn=BatchNorm1d, act=ReLU)): the
        fused copy computes the original function exactly on dyadic data and
        keeps the names fc, bn and act (``_fuse_sequential``: "Preserve the
        original keys"). The passing neighbour is
        ``TestFuseBatchNormEdgeCases::test_no_batchnorm_returns_copy``, whose
        integer keys are kept."""
        from collections import OrderedDict

        model = nn.Sequential(OrderedDict(
            fc=_dyadic_linear(nn.Linear(3, 4)),
            bn=_dyadic_bn(nn.BatchNorm1d(4, eps=_EPS)),
            act=nn.ReLU())).eval()
        fused = fuse_batchnorm(model)
        x = _dyadic_input().double()
        with torch.no_grad():
            assert torch.equal(copy.deepcopy(fused).double()(x),
                               copy.deepcopy(model).double()(x))
        assert [n for n, _ in fused.named_children()] == ["fc", "bn", "act"]


class TestFusedLayerDtype:
    """The fused layers keep float64 parameters and take and return the
    network's dtype."""

    def test_fused_layers_are_float64_and_return_the_networks_dtype(self):
        """A float32 network: the fused layers hold float64 weights and
        biases, a fused layer in the middle (followed by a float32 Linear) and
        one at the end take float32 and the model returns float32; after
        ``.double()`` it returns float64 and after ``.float()`` its fused
        layers are float32 and it returns float32; a float64 network fuses to
        a float64 model. Every forward agrees with the original's exactly on
        dyadic data, and fusing the fused model again changes nothing. Catches
        a fused layer whose output is not cast back (a float64 result, or a
        dtype error in the next layer), an input not cast to the parameters'
        dtype, and parameters stored in the network's dtype."""
        def build():
            return nn.Sequential(
                _dyadic_linear(nn.Linear(3, 4)),
                _dyadic_bn(nn.BatchNorm1d(4, eps=_EPS)), nn.ReLU(),
                _dyadic_linear(nn.Linear(4, 2)),
                _dyadic_bn(nn.BatchNorm1d(2, eps=_EPS))).eval()

        model = build()
        x = _dyadic_input()
        with torch.no_grad():
            expected = model(x)
            fused = fuse_batchnorm(model)
            assert fused[0].weight.dtype == torch.float64
            assert fused[0].bias.dtype == torch.float64
            assert fused[3].weight.dtype == torch.float64
            out = fused(x)
            assert out.dtype == torch.float32
            assert torch.equal(out, expected)
            assert fused[0](x).dtype == torch.float32
            double = copy.deepcopy(fused).double()(x.double())
            assert double.dtype == torch.float64
            assert torch.equal(double, expected.double())
            single = copy.deepcopy(fused).float()
            assert single[0].weight.dtype == torch.float32
            assert single(x).dtype == torch.float32
            assert torch.equal(single(x), expected)
            model64 = build().double()
            fused64 = fuse_batchnorm(model64)
            out64 = fused64(x.double())
            assert out64.dtype == torch.float64
            assert torch.equal(out64, model64(x.double()))
            again = fuse_batchnorm(fused)
            assert again(x).dtype == torch.float32
            assert torch.equal(again(x), expected)


    def test_fused_model_scripts_and_float_records_no_casts(self):
        """The cast hooks are typed so that ``torch.jit.script`` accepts the
        fused model, which then computes the original function exactly on
        dyadic data; after ``.float()`` the hooks skip the cast, so a trace
        records no ``aten::to`` (an ONNX export has no Cast nodes there).
        Catches hooks without type annotations and a cast that is applied
        even when the dtypes already match."""
        model = nn.Sequential(
            _dyadic_linear(nn.Linear(3, 4)),
            _dyadic_bn(nn.BatchNorm1d(4, eps=_EPS)), nn.ReLU()).eval()
        x = _dyadic_input()
        with torch.no_grad():
            expected = model(x)
            fused = fuse_batchnorm(model)
            assert torch.equal(torch.jit.script(fused)(x), expected)
            traced = torch.jit.trace(fused.float(), x)
        assert not [n for n in traced.graph.nodes() if n.kind() == "aten::to"]


class TestStripFinalSoftmaxState:
    """strip_final_softmax works on a copy and returns it in eval mode."""

    def test_result_is_in_eval_mode_and_the_caller_keeps_its_mode(self):
        """From a Sequential with a Dropout and a trailing softmax in training
        mode, and from a traced GraphModule in training mode, every module of
        the result is in eval mode (the docstring promises it), while the
        caller's model stays in training mode with its softmax. Catches the
        final eval() left out (the rebuilt Sequential comes back in training
        mode) and a stripping that works on the caller's model."""
        for model in (nn.Sequential(nn.Linear(4, 6), nn.Dropout(0.5),
                                    nn.Linear(6, 3), nn.Softmax(dim=-1)),
                      torch.fx.symbolic_trace(_SoftmaxNet())):
            model.train()
            stripped = strip_final_softmax(model)
            training = [n or "<root>" for n, m in stripped.named_modules()
                        if m.training]
            assert not training, f"modules in training mode: {training}"
            assert all(m.training for m in model.modules())
            assert has_softmax(model)

    @known_bug(12, "strip_final_softmax raises torch.fx's RuntimeError ('Tried "
                   "to erase Node ... but it still had 1 users') when the "
                   "trailing softmax node also feeds a node whose result is "
                   "unused", raises=RuntimeError)
    def test_trailing_softmax_with_a_second_user_is_stripped(self):
        """A traced module whose trailing softmax output also feeds an unused
        ``.sum()``: the output node reads the softmax, so it is a trailing
        softmax and the stripped model must return exactly the logits, not
        crash (a model returned with its softmax would be verified for
        probabilities where logits are specified). The neighbour without the
        second user is
        ``TestStripFinalSoftmax::test_graph_module_removes_softmax``."""
        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 3)
                self.sm = nn.Softmax(dim=-1)

            def forward(self, x):
                p = self.sm(self.fc(x))
                _ = p.sum()
                return p

        net = _Net().eval()
        gm = torch.fx.symbolic_trace(net)
        x = torch.randn(5, 4)
        with torch.no_grad():
            got = strip_final_softmax(gm)(x)
            assert torch.equal(got, net.fc(x))
