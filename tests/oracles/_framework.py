"""Reachability oracle framework.

The single load-bearing helper is :func:`assert_set_contains_pushforward`,
which checks that the *pushforward* of an input set through a concrete
PyTorch operation lies inside the reachable output set produced by an n2v
layer reachability routine.

Soundness intuition
-------------------
A reachability operation ``reach: (layer, set_in) -> set_out`` is sound iff
for every concrete input point ``x in set_in``, the value ``layer(x)`` is
contained in ``set_out``. The oracle Monte-Carlo-checks this by sampling
many points and verifying containment in (the union of) the produced
output sets.

This is a *necessary* but not *sufficient* condition for soundness, and is
intentionally cheaper than a formal proof. It catches dispatcher routing
bugs, basis-matrix transpose mistakes, and constraint-sign errors with
high probability.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Union

import numpy as np

from n2v.sets import Box, Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono

SetType = Union[Star, Zono, Box, ImageStar, ImageZono]


def sample_from_set(
    input_set: SetType,
    n_samples: int,
    seed: int = 0,
) -> np.ndarray:
    """Sample ``n_samples`` concrete points from ``input_set``.

    Sampling is SEEDED (fix-list T2-4): unseeded global-RNG sampling
    made oracle failures unreproducible and CI runs flaky at the
    containment boundary.

    Returns
    -------
    np.ndarray
        Array of shape ``(dim, k)`` where ``k <= n_samples``. For Box/Zono
        the helper always returns exactly ``n_samples`` columns; for Star
        it may return fewer if rejection sampling fails for the requested
        budget.
    """
    rng = np.random.default_rng(seed)

    if isinstance(input_set, (Star, ImageStar)):
        base = input_set if not isinstance(input_set, ImageStar) else input_set.to_star()
        # Star.sample uses the global RNG internally; seed it locally so
        # the draw is reproducible without disturbing other tests' RNG
        # state more than necessary.
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            return base.sample(n_samples)
        finally:
            np.random.set_state(state)

    if isinstance(input_set, (Zono, ImageZono)):
        base = input_set if not isinstance(input_set, ImageZono) else input_set.to_zono()
        # Sample alpha ~ U[-1, 1]^k and compute c + V @ alpha.
        n_gen = base.V.shape[1]
        alpha = rng.uniform(-1.0, 1.0, size=(n_gen, n_samples))
        return base.c + base.V @ alpha

    if isinstance(input_set, Box):
        lb = np.asarray(input_set.lb, dtype=np.float64).reshape(-1, 1)
        ub = np.asarray(input_set.ub, dtype=np.float64).reshape(-1, 1)
        return rng.uniform(lb, ub, size=(lb.shape[0], n_samples))

    raise TypeError(f"Unsupported set type for sampling: {type(input_set).__name__}")


def contains(output_set: SetType, point: np.ndarray, atol: float = 1e-5) -> bool:
    """Check whether ``point`` lies inside ``output_set`` (with tolerance)."""
    point = np.asarray(point, dtype=np.float64).reshape(-1, 1)

    if isinstance(output_set, ImageStar):
        return output_set.to_star().contains(point)
    if isinstance(output_set, ImageZono):
        return output_set.to_zono().to_star().contains(point)
    if isinstance(output_set, Star):
        return output_set.contains(point)
    if isinstance(output_set, Zono):
        return output_set.to_star().contains(point)
    if isinstance(output_set, Box):
        lb = output_set.lb.reshape(-1, 1)
        ub = output_set.ub.reshape(-1, 1)
        return bool(np.all(point >= lb - atol) and np.all(point <= ub + atol))

    raise TypeError(f"Unsupported set type for containment: {type(output_set).__name__}")


def _contains_in_any(output_sets: Sequence[SetType], point: np.ndarray, atol: float) -> bool:
    return any(contains(s, point, atol=atol) for s in output_sets)


def assert_set_contains_pushforward(
    layer_fn: Callable[[np.ndarray], np.ndarray],
    input_set: SetType,
    output_sets: Iterable[SetType],
    n_samples: int = 512,
    atol: float = 1e-5,
    min_required: int | None = None,
    seed: int = 0,
) -> None:
    """Sample ``n_samples`` points from ``input_set``, push them through
    ``layer_fn``, and assert every pushforward point lies inside at least
    one of ``output_sets``.

    Parameters
    ----------
    layer_fn
        Callable that maps a single concrete input point ``x`` (1D
        np.ndarray of length ``dim_in``) to a 1D np.ndarray of length
        ``dim_out``.
    input_set
        Input reachable set (Star/Zono/Box or their Image variants).
    output_sets
        Iterable of reachable output sets to test containment against.
    n_samples
        Target number of Monte-Carlo samples.
    atol
        Absolute tolerance for box-style containment checks.
    min_required
        Minimum number of successfully drawn sample points for the
        check to be meaningful. Defaults to ``max(32, n_samples // 8)``
        (deep-dive review: with no floor, a heavily-constrained Star
        whose rejection sampling yields 1-6 points silently "passed" a
        check that claims hundreds of MC samples -- a permissive-
        direction hole). Pass an explicit smaller value only when a
        tiny sample budget is genuinely intended.
    seed
        RNG seed for reproducible sampling (fix-list T2-4).
    """
    output_list: List[SetType] = list(output_sets)
    if not output_list:
        raise AssertionError("Oracle received no output sets to check against.")

    samples = sample_from_set(input_set, n_samples, seed=seed)
    if samples.size == 0:
        raise AssertionError("Sampling produced no points for the input set.")

    if min_required is None:
        min_required = max(32, n_samples // 8)

    k = samples.shape[1]
    if k < min_required:
        raise AssertionError(
            f"Oracle sampled only {k} points (< min_required={min_required}); "
            f"the containment check would be statistically vacuous. "
            f"Increase the sampling budget or pass an explicit "
            f"min_required if a small sample is intended."
        )

    misses = 0
    examples: list[np.ndarray] = []
    for i in range(k):
        x = samples[:, i]
        y = np.asarray(layer_fn(x), dtype=np.float64).reshape(-1)
        if not _contains_in_any(output_list, y, atol):
            misses += 1
            if len(examples) < 3:
                examples.append(y)

    if misses > 0:
        raise AssertionError(
            f"Pushforward containment failed for {misses}/{k} samples. "
            f"First miss(es): {examples!r}"
        )



