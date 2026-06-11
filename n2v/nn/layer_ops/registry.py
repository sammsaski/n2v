"""Declarative layer-reachability registry.

This module provides a decorator-based registration mechanism that
lets new layers declare their reachability handlers without needing
to edit ``dispatcher.py`` directly. The dispatcher consults the
registry as a fallback after its existing ``isinstance`` chains, so
this is purely additive and does not break any existing layer.

Usage::

    from n2v.nn.layer_ops.registry import register
    from n2v.sets import Star

    @register(MyLayer, Star)
    def my_layer_star(layer, input_sets, method='exact', **kwargs):
        ...

The registry is consulted in *registration order*; when multiple
handlers match an ``isinstance`` test, the *first* registered handler
wins. Register subclass handlers before their base classes if
precedence matters.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Tuple, Type

# Registry: list of (layer_cls, set_cls, handler) in registration order.
_REGISTRY: List[Tuple[Type, Type, Callable]] = []


def register(layer_cls: Type, set_cls: Type) -> Callable[[Callable], Callable]:
    """Register a reachability handler for a (layer_cls, set_cls) pair.

    Parameters
    ----------
    layer_cls
        The ``nn.Module`` (sub)class detected via ``isinstance``.
    set_cls
        The reachable set class (``Star``, ``Zono``, ``Box``, ``Hexatope``,
        ``Octatope`` and their Image variants).

    Returns
    -------
    Callable
        The decorator. The wrapped function must have the signature
        ``handler(layer, input_sets, method='exact', **kwargs)``.
    """

    def _wrap(fn: Callable) -> Callable:
        _REGISTRY.append((layer_cls, set_cls, fn))
        return fn

    return _wrap


def lookup(layer: Any, set_cls: Type) -> Optional[Callable]:
    """Look up a handler for the given ``layer`` instance and ``set_cls``.

    Returns the first registered handler whose ``layer_cls`` matches via
    ``isinstance`` and whose ``set_cls`` matches by ``issubclass``.
    Returns ``None`` if no handler is registered.
    """
    for layer_cls, registered_set, handler in _REGISTRY:
        if isinstance(layer, layer_cls) and issubclass(set_cls, registered_set):
            return handler
    return None


def registered_pairs() -> Iterable[Tuple[Type, Type]]:
    """Iterate over the registered ``(layer_cls, set_cls)`` pairs.

    Useful for introspection and for the coverage matrix in the docs.
    """
    for layer_cls, set_cls, _ in _REGISTRY:
        yield (layer_cls, set_cls)


def clear_registry() -> None:
    """Test helper: remove every registered handler."""
    _REGISTRY.clear()
