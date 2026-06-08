"""
n2v.snn — SNN verification subpackage.

Provides F2FMLP model, latency encoding utilities, LP verification functions,
and the SNNVerifier training/verification class.
"""

from n2v.snn.model import F2FMLP
from n2v.snn.encoding import latency_from_values, encode_batch, spike_train_from_latencies
from n2v.snn.lp import (
    build_symbolic_relaxation_lp,
    build_symbolic_relaxation_lp_split,
    verify_symbolic_sample,
    make_bounds,
    feasible_latencies,
)
from n2v.snn.verifier import SNNVerifier, monte_carlo_outputs, bounds_cover_outputs

__all__ = [
    "F2FMLP",
    "latency_from_values",
    "encode_batch",
    "spike_train_from_latencies",
    "build_symbolic_relaxation_lp",
    "build_symbolic_relaxation_lp_split",
    "verify_symbolic_sample",
    "make_bounds",
    "feasible_latencies",
    "SNNVerifier",
    "monte_carlo_outputs",
    "bounds_cover_outputs",
]
