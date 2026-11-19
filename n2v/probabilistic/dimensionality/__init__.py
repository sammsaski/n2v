"""
Dimensionality reduction for high-dimensional outputs.
"""

from n2v.probabilistic.dimensionality.deflation_pca import DeflationPCA
from n2v.probabilistic.dimensionality.randomized_pca import RandomizedSVDPCA

__all__ = [
    "DeflationPCA",
    "RandomizedSVDPCA",
]
