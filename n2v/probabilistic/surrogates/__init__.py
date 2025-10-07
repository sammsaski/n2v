"""
Surrogate methods for probabilistic verification.
"""

from n2v.probabilistic.surrogates.base import Surrogate
from n2v.probabilistic.surrogates.naive import NaiveSurrogate
from n2v.probabilistic.surrogates.clipping_block import (
    ClippingBlockSurrogate,
    BatchedClippingBlockSurrogate
)

__all__ = [
    'Surrogate',
    'NaiveSurrogate',
    'ClippingBlockSurrogate',
    'BatchedClippingBlockSurrogate'
]
