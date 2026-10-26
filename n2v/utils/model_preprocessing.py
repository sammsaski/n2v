"""
Model preprocessing utilities for neural network verification.

Provides BatchNorm fusion to simplify models before reachability analysis.
Fusing BatchNorm into preceding Conv2d/Linear layers eliminates BatchNorm
from the computation graph, which avoids the need for a dedicated BatchNorm
reachability implementation.

Also provides final-softmax stripping. Star reachability has a sound Softmax
rule (softmax_star, a box relaxation), but stripping is exact and
order-preserving: it leaves argmax/argmin unchanged and avoids the box
relaxation's correlation-dropping over-approximation. Valid only for specs over
logits/argmax, not softmax probabilities -- see strip_final_softmax.

Fusion formula (Conv2d + BatchNorm2d):
    scale = gamma / sqrt(running_var + eps)
    W_fused = scale.reshape(-1, 1, 1, 1) * W_conv
    b_fused = scale * (b_conv - running_mean) + beta

For Linear + BatchNorm1d, the scale reshape is (-1, 1). The fused parameters
are computed and kept in float64 (see _fold_batchnorm).
"""

import copy
import logging
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# onnx2torch converts an ONNX Softmax at opset <= 11 to OnnxSoftmaxV1V11, which
# is NOT an nn.Softmax subclass (opset >= 13 maps to nn.Softmax). Import it so
# detection/stripping recognizes that wrapper. Optional: without onnx2torch such
# a module can't appear, so the None fallback is sound (no new dep -- n2v.utils
# already imports onnx2torch via model_loader).
try:
    from onnx2torch.node_converters.activations import (
        OnnxSoftmaxV1V11 as _OnnxSoftmaxV1V11,
    )
except Exception:  # pragma: no cover - onnx2torch is optional here
    _OnnxSoftmaxV1V11 = None


def has_batchnorm(model: nn.Module) -> bool:
    """
    Check whether a model contains any BatchNorm layers.

    Args:
        model: PyTorch model to inspect.

    Returns:
        True if any module is an instance of BatchNorm1d, BatchNorm2d,
        or BatchNorm3d; False otherwise.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return True
    return False


def _is_softmax_module(module: nn.Module) -> bool:
    """
    Check whether a module is a plain (non-log) softmax activation.

    Recognizes nn.Softmax and onnx2torch's OnnxSoftmaxV1V11 wrapper (the
    non-nn.Softmax module an ONNX Softmax at opset <= 11 converts to). The
    log-softmax variants are excluded: nn.LogSoftmax is not an nn.Softmax
    subclass, and the wrapper sets is_log=True.
    """
    if isinstance(module, nn.Softmax):
        return True
    if _OnnxSoftmaxV1V11 is not None and isinstance(module, _OnnxSoftmaxV1V11):
        return not getattr(module, 'is_log', False)
    return False


def _is_softmax_call_node(node) -> bool:
    """
    Check whether an fx node is a non-module softmax call (functional or method).

    Matches a call_function whose target is named 'softmax' (e.g.
    torch.softmax, F.softmax) or a call_method named 'softmax' (e.g.
    ``x.softmax(-1)``); log_softmax is excluded by the exact name. Some ONNX
    importers emit softmax this way rather than as a module. In every matched
    form the softmax input is the node's first positional arg.
    """
    op = getattr(node, 'op', None)
    target = getattr(node, 'target', None)
    if op == 'call_function':
        return getattr(target, '__name__', None) == 'softmax'
    if op == 'call_method':
        return target == 'softmax'
    return False


def has_softmax(model: nn.Module) -> bool:
    """
    Check whether a model contains any (non-log) Softmax layers.

    Recognizes nn.Softmax and onnx2torch's OnnxSoftmaxV1V11 wrapper (emitted
    for ONNX Softmax at opset <= 11), plus a non-module softmax (call_function
    or call_method node) in a torch.fx.GraphModule. LogSoftmax variants are not
    counted.

    Args:
        model: PyTorch model to inspect.

    Returns:
        True if the model contains a softmax activation; False otherwise.
    """
    if any(_is_softmax_module(module) for module in model.modules()):
        return True
    # Functional / method softmax shows up as a call node, not a module.
    import torch.fx
    if isinstance(model, torch.fx.GraphModule):
        return any(_is_softmax_call_node(n) for n in model.graph.nodes)
    return False


def _bn_affine(bn: nn.Module):
    """(gamma, beta) of a BatchNorm layer; a layer built with affine=False
    has neither and applies gamma = 1, beta = 0."""
    gamma = bn.weight if bn.weight is not None else torch.ones_like(bn.running_mean)
    beta = bn.bias if bn.bias is not None else torch.zeros_like(bn.running_mean)
    return gamma, beta


def _fold_batchnorm(weight: torch.Tensor, bias, bn: nn.Module):
    """
    Fold an eval-mode BatchNorm into the weight and bias of the layer before
    it, in float64.

    ``gamma / sqrt(running_var + eps)`` is irrational in general, so the
    folded parameters are rounded in any precision. Rounded to the layer's
    float32 they moved the fused layer by about 6e-8 relative, and a Box,
    Star or ImageStar reach from a point missed the original network's real
    output by up to 1.5e-6. In float64 each fused weight is off by a few
    units of 1e-16 relative to itself. The fused bias is off by a few units
    of 1e-16 relative to ``|scale * (b - mean)| + |beta|``, the size of the
    terms folded into it, not to its own value: when those terms cancel,
    its relative error is large.

    This rounding is not bounded anywhere downstream. Box reach rounds
    outward only for the parameters it is given, so a Box through a fused
    layer is not a rigorous enclosure of the original network: from a zero
    point input, or from any Box for an output whose weight row is all
    zeros (a pruned neuron), it is the rounded bias as a single point, which
    cannot contain the irrational real value, and a later large weight
    scales that miss up in absolute terms. verify_specification can then
    return a false UNSAT under real-number semantics. The standalone
    BatchNorm Box reach (batchnorm_reach.batchnorm_box) bounds the rounding
    of its own fold, so a rigorous Box reach needs the BatchNorm left
    unfused.

    Args:
        weight: The Conv2d/Linear weight (output channels first).
        bias: The Conv2d/Linear bias, or None.
        bn: The batch normalization layer (must be in eval mode).

    Returns:
        (fused_weight, fused_bias) as float64 tensors on the weight's device.
    """
    def f64(t):
        return t.detach().to(dtype=torch.float64)

    gamma, beta = _bn_affine(bn)
    # Compute scale factor: gamma / sqrt(var + eps)
    scale = f64(gamma) / torch.sqrt(f64(bn.running_var) + bn.eps)

    # Fused weight: scale broadcast over the output-channel axis
    fused_weight = scale.reshape((-1,) + (1,) * (weight.dim() - 1)) * f64(weight)

    # Fused bias: scale * (b - mean) + beta
    mean = f64(bn.running_mean)
    shifted = f64(bias) - mean if bias is not None else -mean
    fused_bias = scale * shifted + f64(beta)
    return fused_weight, fused_bias


def _cast_input_to_weight_dtype(
        module: nn.Module, args: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
    """Forward pre-hook of a fused layer: cast its input (a Conv2d/Linear
    takes one tensor) to the dtype of its (float64) parameters.

    Typed so that torch.jit.script accepts the hook. The cast is skipped
    when the dtypes already match (e.g. after ``.float()`` of the fused
    model), so tracing or ONNX export then records no cast here.
    """
    x = args[0]
    if x.dtype != module.weight.dtype:
        x = x.to(module.weight.dtype)
    return (x,)


def _cast_output_to_network_dtype(
        module: nn.Module, args: Tuple[torch.Tensor],
        output: torch.Tensor) -> torch.Tensor:
    """Forward hook of a fused layer: cast the output back to the dtype of
    the BatchNorm it replaced, recorded as the dtype of the empty buffer
    ``network_dtype``. Typed and skipped on a dtype match, like
    _cast_input_to_weight_dtype."""
    if output.dtype != module.network_dtype.dtype:
        output = output.to(module.network_dtype.dtype)
    return output


def _with_float64_parameters(fused: nn.Module, fused_weight: torch.Tensor,
                             fused_bias: torch.Tensor,
                             bn: nn.Module) -> nn.Module:
    """
    Give a fused layer its float64 parameters, keeping it callable inside a
    network of the BatchNorm's dtype.

    The layer stays a plain Conv2d/Linear, so reachability reads the float64
    parameters directly. When the BatchNorm is not float64, two forward hooks
    let the torch forward (used for shape inference and by callers of
    fuse_batchnorm) take and return that dtype: the input is cast to float64
    and the output back. Reachability does not run the hooks; it treats
    float32/float64 casts as the identity. The dtype is kept as an empty,
    non-persistent buffer so that converting the whole fused model (e.g.
    ``.double()``) keeps the output cast consistent with the other layers.
    """
    fused.weight = nn.Parameter(fused_weight)
    fused.bias = nn.Parameter(fused_bias)
    dtype = bn.running_mean.dtype
    if dtype != torch.float64:
        fused.register_buffer(
            'network_dtype',
            torch.empty(0, dtype=dtype, device=fused_weight.device),
            persistent=False)
        fused.register_forward_pre_hook(_cast_input_to_weight_dtype)
        fused.register_forward_hook(_cast_output_to_network_dtype)
    return fused


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fuse a Conv2d layer with a following BatchNorm2d layer.

    Args:
        conv: The convolutional layer.
        bn: The batch normalization layer (must be in eval mode).

    Returns:
        A new Conv2d layer with fused float64 weights and bias.
    """
    # Conv weight shape: (out_channels, in_channels/groups, kH, kW)
    fused_weight, fused_bias = _fold_batchnorm(conv.weight, conv.bias, bn)

    # Create new Conv2d with same configuration but with bias
    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    )
    return _with_float64_parameters(fused_conv, fused_weight, fused_bias, bn)


def _fuse_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d) -> nn.Linear:
    """
    Fuse a Linear layer with a following BatchNorm1d layer.

    Args:
        linear: The linear layer.
        bn: The batch normalization layer (must be in eval mode).

    Returns:
        A new Linear layer with fused float64 weights and bias.
    """
    # Linear weight shape: (out_features, in_features)
    fused_weight, fused_bias = _fold_batchnorm(linear.weight, linear.bias, bn)

    # Create new Linear with same configuration but with bias
    fused_linear = nn.Linear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=True,
    )
    return _with_float64_parameters(fused_linear, fused_weight, fused_bias, bn)


def _computes_stock_function(module: nn.Module, base: type) -> bool:
    """True if ``module``'s class does not override ``base``'s forward (nor,
    for a conv, ``_conv_forward``). Fusion rebuilds the layer as a plain
    ``base`` from its parameters, which drops any overriding code (e.g. a
    weight-standardized conv)."""
    cls = type(module)
    return (cls.forward is base.forward
            and getattr(cls, '_conv_forward', None)
            is getattr(base, '_conv_forward', None))


def _is_fusable_pair(layer_a: nn.Module, layer_b: nn.Module) -> bool:
    """
    Check if two adjacent layers form a fusable pair.

    Returns True for Conv2d+BatchNorm2d or Linear+BatchNorm1d pairs whose
    classes compute the stock torch function. A BatchNorm without running
    statistics (track_running_stats=False) normalizes each batch by its own
    statistics even in eval mode, so it is not an affine layer and is never
    fused.
    """
    if (getattr(layer_b, 'running_mean', None) is None
            or getattr(layer_b, 'running_var', None) is None):
        return False
    for conv_type, bn_type in ((nn.Conv2d, nn.BatchNorm2d),
                               (nn.Linear, nn.BatchNorm1d)):
        if isinstance(layer_a, conv_type) and isinstance(layer_b, bn_type):
            return (_computes_stock_function(layer_a, conv_type)
                    and _computes_stock_function(layer_b, bn_type))
    return False


def _fuse_pair(layer_a: nn.Module, layer_b: nn.Module) -> nn.Module:
    """
    Fuse a fusable pair of layers into a single layer.

    Args:
        layer_a: Conv2d or Linear layer.
        layer_b: Corresponding BatchNorm layer.

    Returns:
        Fused layer.
    """
    if isinstance(layer_a, nn.Conv2d) and isinstance(layer_b, nn.BatchNorm2d):
        return _fuse_conv_bn(layer_a, layer_b)
    if isinstance(layer_a, nn.Linear) and isinstance(layer_b, nn.BatchNorm1d):
        return _fuse_linear_bn(layer_a, layer_b)
    raise ValueError(f"Cannot fuse {type(layer_a)} with {type(layer_b)}")


def _fuse_sequential(seq: nn.Sequential) -> nn.Sequential:
    """
    Fuse BatchNorm layers in a Sequential module.

    Walks through child modules pairwise, fusing Conv+BN and Linear+BN pairs.
    Replaces fused BN layers with nn.Identity().

    Args:
        seq: A Sequential module (already deep-copied).

    Returns:
        The same Sequential with fused layers (modified in-place on the copy).
    """
    children = list(seq.children())
    num_children = len(children)

    # First, recurse into any child Sequential modules (a subclass with its
    # own forward is traced, see _fuse_generic_module)
    for i, child in enumerate(children):
        if isinstance(child, nn.Sequential):
            children[i] = _fuse_generic_module(child)

    # Now fuse adjacent pairs at this level
    i = 0
    while i < num_children - 1:
        layer_a = children[i]
        layer_b = children[i + 1]

        if _is_fusable_pair(layer_a, layer_b):
            children[i] = _fuse_pair(layer_a, layer_b)
            children[i + 1] = nn.Identity()
            i += 2  # skip the Identity we just placed
        else:
            i += 1

    # Rebuild the Sequential with the updated children
    # Preserve the original keys (integer indices for plain Sequential)
    new_seq = nn.Sequential()
    for idx, child in enumerate(children):
        new_seq.add_module(str(idx), child)

    return new_seq


def _fuse_generic_module(module: nn.Module) -> nn.Module:
    """
    Fuse BatchNorm layers in an arbitrary module (not just Sequential).

    A Sequential uses _fuse_sequential (its children run in order). A
    torch.fx.GraphModule, or any other module after tracing, uses graph-based
    fusion; a module that cannot be traced is returned unfused. A Sequential
    subclass that overrides forward counts as "any other module":
    _fuse_sequential rebuilds a plain nn.Sequential, which would drop the
    overriding forward.

    Args:
        module: A module (already deep-copied).

    Returns:
        The module with fused layers (modified in-place on the copy).
    """
    # Handle Sequential directly
    if (isinstance(module, nn.Sequential)
            and type(module).forward is nn.Sequential.forward):
        return _fuse_sequential(module)

    # Handle torch.fx.GraphModule
    import torch.fx
    if isinstance(module, torch.fx.GraphModule):
        return _fuse_graph_module(module)

    # A generic module's children are not necessarily applied in the order
    # they are registered (a registered BN may normalize a different tensor,
    # or the conv's output may also feed another op), so pairing adjacent
    # children could fuse a BN into a conv it never follows -- the fused model
    # would then compute a different function. Trace it and fuse along the
    # graph instead. An untraceable module is left unfused: its BatchNorms
    # are reached as standalone layers.
    try:
        gm = torch.fx.symbolic_trace(module)
    except Exception as e:  # noqa: BLE001 - fusion is only an optimization
        logger.debug(f'BatchNorm fusion skipped: tracing failed ({e})')
        return module
    return _fuse_graph_module(gm)


def _fuse_graph_module(gm: 'torch.fx.GraphModule') -> 'torch.fx.GraphModule':
    """
    Fuse BatchNorm layers in a torch.fx.GraphModule by iterating over graph nodes.

    Finds sequential call_module pairs where the first is Conv2d/Linear and
    the second is the corresponding BatchNorm, fuses them, and recompiles. A
    pair is fused only when the BatchNorm is the conv/linear output's single
    user and each of the two modules is called once.

    Args:
        gm: A torch.fx.GraphModule (already deep-copied).

    Returns:
        The GraphModule with fused layers and recompiled graph.
    """

    graph = gm.graph
    nodes = list(graph.nodes)

    # Build a map from node name to the module it references
    modules = dict(gm.named_modules())

    for node in nodes:
        if node.op != 'call_module':
            continue

        # Check if this node's module is a BatchNorm
        mod = modules.get(node.target)
        if not isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            continue

        # Find the preceding node (the input to this BN)
        if len(node.args) == 0:
            continue
        prev_node = node.args[0]
        if not hasattr(prev_node, 'op') or prev_node.op != 'call_module':
            continue

        # The fused layer computes BN(conv), so the conv's output must feed
        # only this BN and the conv module must be called only here; any
        # other user would otherwise read BN(conv) in place of conv. The BN
        # module must be called only here too: fusion replaces it with an
        # identity, which would drop the normalization at its other calls.
        if len(prev_node.users) != 1:
            continue
        if any(sum(1 for n in graph.nodes if n.op == 'call_module'
                   and n.target == target) != 1
               for target in (prev_node.target, node.target)):
            continue

        prev_mod = modules.get(prev_node.target)
        if prev_mod is None:
            continue

        if _is_fusable_pair(prev_mod, mod):
            # Fuse the pair
            fused = _fuse_pair(prev_mod, mod)

            # Replace the Conv/Linear module in the GraphModule
            # Navigate the module hierarchy to set the attribute
            _set_module_by_name(gm, prev_node.target, fused)

            # Replace BN with Identity
            _set_module_by_name(gm, node.target, nn.Identity())
            # Keep the lookup current: a second BN on this conv (now fed by
            # the conv node) must fuse into the fused layer, not the original.
            modules[prev_node.target] = fused
            modules[node.target] = nn.Identity()

            # Redirect BN's output users to point to the Identity
            # (the Identity will just pass through the conv/linear output)
            node.replace_all_uses_with(prev_node)

            # Remove the BN node from the graph
            graph.erase_node(node)

    gm.recompile()
    return gm


def _set_module_by_name(model: nn.Module, target: str, new_module: nn.Module) -> None:
    """
    Set a module in a model by its dotted name path.

    For example, target="layer1.conv" sets model.layer1.conv = new_module.
    """
    parts = target.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def fuse_batchnorm(model: nn.Module) -> nn.Module:
    """
    Fuse BatchNorm layers into preceding Conv2d/Linear layers.

    Creates a deep copy of the model and fuses all Conv2d+BatchNorm2d and
    Linear+BatchNorm1d pairs. The BatchNorm layers are replaced with
    nn.Identity(). The original model is not modified.

    The fused model produces numerically identical outputs (within floating
    point tolerance) to the original model in eval mode.

    The fused layers keep float64 parameters (the fold is rounded at float64
    precision rather than float32's, see _fold_batchnorm) and compute in
    float64; they take and return the dtype of the BatchNorm they replace,
    so the fused model accepts the same inputs as the original. The fold is
    still rounded, so a Box reach of the fused model does not rigorously
    enclose the original network (see _fold_batchnorm).

    The float64 layers make the torch forward of the fused model about an
    order of magnitude slower on a GPU, and an ONNX export of it has Cast
    nodes around each fused layer. For torch use where speed
    matters more than the fold's precision (attacks, sampling, export to
    another tool), call ``.float()`` on the fused model: the fused layers
    then run in float32 like the rest of the network, with no casts, and
    the fold is rounded to float32.

    Linear+BatchNorm1d fusion assumes the Linear's output is 2-D (N,
    features). The input rank is not known here, so it is not checked: on a
    3-D (N, C, L) tensor BatchNorm1d normalizes axis 1 rather than the
    Linear's output features, and the fused model computes a different
    function. (n2v's Linear reach needs a set of dimension in_features, and
    a 3-D input with C > 1 has dimension C * in_features, so reach refuses
    such a model whether or not it is fused.)

    Args:
        model: PyTorch model to preprocess. Must be in eval mode or will
               be set to eval mode on the copy.

    Returns:
        A new model with BatchNorm layers fused into their preceding
        Conv2d/Linear layers. The model is in eval mode.

    Example:
        >>> model = nn.Sequential(
        ...     nn.Conv2d(3, 16, 3, padding=1),
        ...     nn.BatchNorm2d(16),
        ...     nn.ReLU(),
        ... )
        >>> model.eval()
        >>> fused = fuse_batchnorm(model)
        >>> # fused has no BatchNorm layers
        >>> assert not has_batchnorm(fused)
    """
    # Deep copy to avoid mutating original
    fused_model = copy.deepcopy(model)
    fused_model.eval()

    # Fuse BatchNorm layers
    fused_model = _fuse_generic_module(fused_model)
    fused_model.eval()

    return fused_model


def _strip_graph_module_softmax(gm: 'torch.fx.GraphModule') -> None:
    """Remove a trailing Softmax node from an fx GraphModule, in place.

    Rewires the output to the softmax's input, erases the node, then re-lints
    and recompiles. Handles module nodes (nn.Softmax or OnnxSoftmaxV1V11) and
    call_function / call_method softmax. No-op if the output isn't a softmax.
    """
    graph = gm.graph
    out_node = next(n for n in graph.nodes if n.op == 'output')
    last = out_node.args[0]
    if (getattr(last, 'op', None) == 'call_module'
            and _is_softmax_module(gm.get_submodule(last.target))):
        out_node.args = (last.args[0],)  # return the softmax's input instead
        graph.erase_node(last)
        # Replace the now-orphaned Softmax submodule with Identity so it no
        # longer shows up in model.modules().
        _set_module_by_name(gm, last.target, nn.Identity())
        graph.lint()
        gm.recompile()
    elif _is_softmax_call_node(last):
        # Functional / method softmax: its input is the first positional arg
        # (or the 'input' kwarg). Rewire the output to it and erase the node.
        softmax_input = last.args[0] if last.args else last.kwargs.get('input')
        if softmax_input is not None:
            out_node.args = (softmax_input,)
            graph.erase_node(last)
            graph.lint()
            gm.recompile()


def strip_final_softmax(model: nn.Module) -> nn.Module:
    """
    Remove a trailing Softmax layer from a model.

    Returns a deep copy with the final softmax dropped; the original is
    unchanged. Softmax is order-preserving, so this leaves argmax/argmin intact
    and exactly preserves the stripped model's logits.

    Precondition: the output spec must be over logits/argmax. Stripping is
    unsound for specs over softmax probabilities (e.g. "P(class) >= 0.9");
    verify those in-graph via softmax_star instead.

    Handles torch.fx.GraphModule (onnx2torch output) and nn.Sequential. The
    softmax is recognized as an nn.Softmax module, an OnnxSoftmaxV1V11 wrapper
    (ONNX opset <= 11), or a call_function / call_method node; LogSoftmax is
    left in place. Returns the copy unchanged if no trailing softmax is found.

    Args:
        model: PyTorch model to preprocess.

    Returns:
        A new model in eval mode with the trailing softmax removed.
    """
    import torch.fx

    stripped = copy.deepcopy(model)
    stripped.eval()

    if isinstance(stripped, torch.fx.GraphModule):
        _strip_graph_module_softmax(stripped)
    elif isinstance(stripped, nn.Sequential) and len(stripped) > 0 \
            and _is_softmax_module(stripped[-1]):
        stripped = nn.Sequential(*list(stripped)[:-1])

    stripped.eval()
    return stripped
