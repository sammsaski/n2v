"""onnx2torch converters for QuantizeLinear / DequantizeLinear (QDQ models).

onnx2torch ships no converter for ``QuantizeLinear`` / ``DequantizeLinear``, so
fully QDQ-quantized graphs (e.g. VNN-COMP's ``smart_turn_multimodal_2026``)
fail to even load. These converters add the exact ONNX inference semantics so
the model converts and its forward matches onnxruntime, which is the
prerequisite for any downstream analysis (issue #47, "to load").

Scope: this only makes such models LOAD and FORWARD correctly. SOUND set-based
reachability through quantization (a rounding/relaxation step) is a separate,
much harder problem and is not addressed here — reach will raise on these ops
until that lands.

Registration is done against onnx2torch's converter registry from the n2v side
(``third_party/onnx2torch`` is vendored and must not be edited). Import this
module before calling ``onnx2torch.convert``.

Semantics (ONNX opset 13):
  DequantizeLinear:  y = (x - zero_point) * scale
  QuantizeLinear:    y = clamp(round(x / scale) + zero_point, qmin, qmax)
``round`` is round-half-to-even, matching ``torch.round`` and the ONNX spec.
Per-axis (per-channel) scale/zero_point are reshaped to broadcast along ``axis``.
"""

import numpy as np
import torch
from torch import nn

from onnx2torch.node_converters.registry import (
    add_converter,
    _CONVERTER_REGISTRY,
    OperationDescription,
)
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import (
    OnnxMapping,
    OnnxToTorchModule,
    OperationConverterResult,
)

# Integer ranges by zero_point dtype — the saturation bounds for QuantizeLinear.
_QMIN_QMAX = {
    torch.int8: (-128.0, 127.0),
    torch.uint8: (0.0, 255.0),
    torch.int16: (-32768.0, 32767.0),
    torch.uint16: (0.0, 65535.0),
    torch.int32: (-2147483648.0, 2147483647.0),
}


def _reshape_qparam(param: torch.Tensor, axis: int, ndim: int) -> torch.Tensor:
    """Reshape a per-axis scale/zero_point to broadcast along ``axis``.

    Per-tensor params (scalar or single element) broadcast as-is.
    """
    if param.ndim == 0 or param.numel() == 1:
        return param.reshape(())
    shape = [1] * ndim
    shape[axis % ndim] = -1
    return param.reshape(shape)


class OnnxDequantizeLinear(nn.Module, OnnxToTorchModule):
    """DequantizeLinear: ``(x - zero_point) * scale`` (float32 output)."""

    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis
        self.register_buffer('scale', torch.empty(0))
        self.register_buffer('zero_point', torch.empty(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        s = _reshape_qparam(self.scale, self.axis, x.ndim)
        z = _reshape_qparam(self.zero_point, self.axis, x.ndim)
        return (x.to(s.dtype) - z.to(s.dtype)) * s


class OnnxQuantizeLinear(nn.Module, OnnxToTorchModule):
    """QuantizeLinear: ``clamp(round(x / scale) + zero_point, qmin, qmax)``.

    Output is the integer-valued result kept as float32 (exactly represents the
    int8/uint8 range), so the following DequantizeLinear reproduces onnxruntime
    numerics without integer-dtype plumbing through the rest of the graph.
    """

    def __init__(self, axis: int, qmin: float, qmax: float):
        super().__init__()
        self.axis = axis
        self.qmin = qmin
        self.qmax = qmax
        self.register_buffer('scale', torch.empty(0))
        self.register_buffer('zero_point', torch.empty(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        s = _reshape_qparam(self.scale, self.axis, x.ndim)
        z = _reshape_qparam(self.zero_point, self.axis, x.ndim)
        q = torch.round(x / s) + z.to(s.dtype)
        return torch.clamp(q, self.qmin, self.qmax)


def _require_no_block_quant(node: OnnxNode) -> None:
    # Opset-21 blocked quantization (block_size attr) is a different, unhandled
    # layout — fail loudly rather than compute a wrong function.
    if node.attributes.get('block_size', 0):
        raise NotImplementedError(
            'Blocked QuantizeLinear/DequantizeLinear (block_size) '
            'is not supported')


def _baked_qparams(node: OnnxNode, graph: OnnxGraph):
    """Read scale and zero_point initializers; default zero_point to zeros."""
    ins = node.input_values
    if len(ins) < 2 or ins[1] not in graph.initializers:
        raise NotImplementedError(
            f'{node.operation_type} with a non-constant scale is not supported')
    scale = graph.initializers[ins[1]].to_torch()
    if len(ins) > 2 and ins[2]:
        if ins[2] not in graph.initializers:
            raise NotImplementedError(
                f'{node.operation_type} with a non-constant zero_point is '
                f'not supported')
        zero_point = graph.initializers[ins[2]].to_torch()
    else:
        zero_point = torch.zeros_like(scale)
    return ins[0], scale, zero_point


def _convert_dequantize_linear(
        node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    _require_no_block_quant(node)
    axis = node.attributes.get('axis', 1)
    data_input, scale, zero_point = _baked_qparams(node, graph)
    module = OnnxDequantizeLinear(axis=axis)
    module.scale = scale
    module.zero_point = zero_point
    return OperationConverterResult(
        torch_module=module,
        onnx_mapping=OnnxMapping(
            inputs=(data_input,), outputs=node.output_values),
    )


def _convert_quantize_linear(
        node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    _require_no_block_quant(node)
    axis = node.attributes.get('axis', 1)
    data_input, scale, zero_point = _baked_qparams(node, graph)
    qmin, qmax = _QMIN_QMAX.get(zero_point.dtype, (0.0, 255.0))
    module = OnnxQuantizeLinear(axis=axis, qmin=qmin, qmax=qmax)
    module.scale = scale
    module.zero_point = zero_point
    return OperationConverterResult(
        torch_module=module,
        onnx_mapping=OnnxMapping(
            inputs=(data_input,), outputs=node.output_values),
    )


# DequantizeLinear since-versions: 10, 13, 19, 21. QuantizeLinear: same.
# Register all so opset resolution finds a match regardless of model opset.
_REGISTRATIONS = (
    ('DequantizeLinear', _convert_dequantize_linear),
    ('QuantizeLinear', _convert_quantize_linear),
)
_VERSIONS = (10, 13, 19, 21)


_NP_QMIN_QMAX = {
    np.dtype('int8'): (-128, 127),
    np.dtype('uint8'): (0, 255),
    np.dtype('int16'): (-32768, 32767),
    np.dtype('uint16'): (0, 65535),
    np.dtype('int32'): (-2147483648, 2147483647),
}


def _np_reshape_qparam(param: np.ndarray, axis: int, ndim: int) -> np.ndarray:
    if param.ndim == 0 or param.size == 1:
        return param.reshape(())
    shape = [1] * ndim
    shape[axis % ndim] = -1
    return param.reshape(shape)


def _np_dequantize(x, scale, zero_point, axis):
    s = _np_reshape_qparam(scale, axis, x.ndim)
    z = _np_reshape_qparam(zero_point, axis, x.ndim)
    return (x.astype(np.float32) - z.astype(np.float32)) * s


def _np_quantize(x, scale, zero_point, axis):
    s = _np_reshape_qparam(scale, axis, x.ndim)
    z = _np_reshape_qparam(zero_point, axis, x.ndim)
    qmin, qmax = _NP_QMIN_QMAX.get(zero_point.dtype, (0, 255))
    # round-half-to-even, matching ONNX / torch.round / np.round.
    q = np.round(x / s) + z.astype(np.float32)
    # The graph-fold path returns the true ONNX integer dtype (unlike the
    # runtime module, which keeps float32 to avoid int plumbing through fx):
    # a folded QuantizeLinear becomes an ONNX initializer, and its consumer is
    # always a DequantizeLinear, which dequantizes the integer correctly.
    return np.clip(q, qmin, qmax).astype(zero_point.dtype)


def fold_constant_qdq(onnx_model):
    """Constant-fold Quantize/DequantizeLinear nodes with all-constant inputs.

    In QDQ models the int weights/biases flow ``int_init -> DequantizeLinear ->
    Conv/Gemm/MatMul``. onnx2torch's weight-consuming converters expect a real
    initializer, not a computed value, so those weight-side Q/DQ must be folded
    away into float initializers before conversion. Activation-side Q/DQ (whose
    data input is a dynamic tensor) are left in place for the runtime
    converters. A no-op when the graph has no Q/DQ nodes.
    """
    from onnx import numpy_helper

    graph = onnx_model.graph
    if not any(n.op_type in ('QuantizeLinear', 'DequantizeLinear')
               for n in graph.node):
        return onnx_model

    consts = {init.name: numpy_helper.to_array(init)
              for init in graph.initializer}

    changed = True
    while changed:
        changed = False
        for node in list(graph.node):
            if node.op_type not in ('QuantizeLinear', 'DequantizeLinear'):
                continue
            ins = list(node.input)
            # All inputs (data, scale, zero_point) must be constants to fold.
            if not ins or any(name not in consts for name in ins if name):
                continue
            axis = 1
            for attr in node.attribute:
                if attr.name == 'block_size' and attr.i:
                    raise NotImplementedError(
                        'Blocked Q/DQ (block_size) is not supported')
                if attr.name == 'axis':
                    axis = attr.i
            x = consts[ins[0]]
            scale = consts[ins[1]]
            if len(ins) > 2 and ins[2]:
                zero_point = consts[ins[2]]
            else:
                default_dtype = (np.uint8 if node.op_type == 'QuantizeLinear'
                                 else scale.dtype)
                zero_point = np.zeros_like(scale, dtype=default_dtype)
            if node.op_type == 'DequantizeLinear':
                out = _np_dequantize(x, scale, zero_point, axis)
            else:
                out = _np_quantize(x, scale, zero_point, axis)

            out_name = node.output[0]
            consts[out_name] = out
            graph.initializer.append(
                numpy_helper.from_array(out, name=out_name))
            graph.node.remove(node)
            changed = True

    return onnx_model


def _register() -> None:
    for op_type, converter in _REGISTRATIONS:
        for version in _VERSIONS:
            desc = OperationDescription(
                domain='', operation_type=op_type, version=version)
            if desc in _CONVERTER_REGISTRY:
                continue  # already registered (idempotent re-import)
            add_converter(operation_type=op_type, version=version)(converter)


_register()
