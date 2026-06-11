"""Custom fx Tracer that treats n2v wrapper modules as graph leaves.

n2v ships a family of "compound" wrapper modules under :mod:`n2v.nn.layers`
(e.g. :class:`SoftmaxAttention`, :class:`PatchEmbed`, :class:`CLSToken`,
:class:`DagAdd`) whose Python ``forward`` bodies use primitives -- ``torch.cat``,
``operator.add``, ``call_method 'transpose'``, ``matmul`` -- that
``_handle_graphmodule`` either does not yet handle or that the dispatcher's
multi-input path is designed to dispatch as a single ``call_module`` node.

Without leaf treatment, ``torch.fx.symbolic_trace`` descends INTO each
wrapper's forward, decomposing it into primitive nodes. The
single-call_module dispatcher then never sees these wrappers, the
multi-input dispatcher's per-wrapper isinstance chain is dead code, and
every primitive raises ``NotImplementedError`` after T0-1.

Treating each wrapper as a leaf module fixes this: ``fx.symbolic_trace``
emits a single ``call_module`` node per wrapper, the dispatcher's
isinstance chain or ``_handle_multi_input_op`` routes it to the correct
reach function, and primitive handlers are no longer needed for the
wrappers' internal arithmetic.

This is the canonical T1-7 fix in ``PR12_FIX_LIST.md`` and the keystone
for end-to-end ViT verification (the ViT integration test in
``tests/integration/test_minimal_vit.py`` cannot complete without it).
"""

from __future__ import annotations

from typing import Any, Tuple, Type

import torch
import torch.fx as fx
import torch.nn as nn


def _n2v_leaf_module_types() -> Tuple[Type[nn.Module], ...]:
    """Return the tuple of n2v wrapper classes that must be fx leaves.

    Lazy import to avoid an import cycle with ``n2v.nn.layers.__init__``.

    PR-1 audit I5: not every wrapper in ``__all__`` has a dispatcher
    branch. Treating one as a leaf without a helper causes ``__call__``
    to fall through ``_registry_lookup`` -> ``None`` ->
    ``NotImplementedError`` -- a coverage REGRESSION on previously
    decomposable models (pre-T1-7 fx happily descended into the
    forward body). We explicitly EXCLUDE wrappers whose forward is
    composed entirely of primitives the dispatcher can already handle,
    so fx can decompose them. This is the "whitelist" half of I5; the
    "implement helper" half is solved per-wrapper as needs arise.
    """
    # All wrappers from n2v.nn.layers are candidates -- they are designed
    # to be dispatched as a single node, not decomposed via fx.
    from n2v.nn import layers as _layers

    # Wrappers whose forward decomposes cleanly into primitives the
    # dispatcher already supports. Leaving these as fx leaves would
    # block ``__call__`` from ever invoking a helper. See I5.
    _EXCLUDED = {
        # ParallelResidual: y = x + a(x) + b(x). fx decomposes to two
        # operator.add calls (set + set), both already handled.
        "ParallelResidual",
    }

    leaf_types: list[Type[nn.Module]] = []
    # Public wrappers from n2v.nn.layers.__all__.
    for name in getattr(_layers, "__all__", []):
        if name in _EXCLUDED:
            continue
        obj = getattr(_layers, name, None)
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            leaf_types.append(obj)
    return tuple(leaf_types)


class N2VTracer(fx.Tracer):
    """fx Tracer that treats every n2v wrapper module as a graph leaf.

    The set of leaf types is materialised on first use, then cached on the
    class to keep tracing fast.
    """

    _leaf_types_cache: Tuple[Type[nn.Module], ...] | None = None

    @classmethod
    def _leaf_types(cls) -> Tuple[Type[nn.Module], ...]:
        if cls._leaf_types_cache is None:
            cls._leaf_types_cache = _n2v_leaf_module_types()
        return cls._leaf_types_cache

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        # Defer to base class for the standard nn.* leaf detection (which
        # already catches nn.Linear / nn.Conv2d / nn.ReLU / nn.LayerNorm /
        # etc.). Add n2v wrappers on top.
        if super().is_leaf_module(m, module_qualified_name):
            return True
        return isinstance(m, self._leaf_types())


def symbolic_trace(root: nn.Module, concrete_args: dict[str, Any] | None = None) -> fx.GraphModule:
    """fx.symbolic_trace replacement that uses :class:`N2VTracer`.

    Mirrors the signature of ``torch.fx.symbolic_trace``. Returns a
    :class:`torch.fx.GraphModule` whose graph treats every n2v wrapper as a
    leaf ``call_module`` node.
    """
    tracer = N2VTracer()
    graph = tracer.trace(root, concrete_args=concrete_args)
    name = root.__class__.__name__ if isinstance(root, nn.Module) else "GraphModule"
    return fx.GraphModule(tracer.root, graph, name)
