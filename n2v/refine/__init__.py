"""
Star-set branch-and-bound verifiers for fully connected ReLU networks.

``verify_refine`` splits ReLU activations over triangle-relaxed stars (exact
CPU checks, or NS-certified GPU bounds, prune and witnesses when CUDA is
available). ``verify_refine_input_crown`` bisects the input box with batched
CROWN bounds, for low-input-dimension networks.
"""

from n2v.refine.types import (LinearSpec, NeuronKey, NeuronMeta, Phase,
                              RefineResult, Status, Witness)
from n2v.refine.operations import split, split_wave, witness_phase
from n2v.refine.bab import verify_refine
from n2v.refine.input_split import verify_refine_input_crown
from n2v.refine.selectors import FaithfulSelector, LayerOrderSelector, Selector

__all__ = ["LinearSpec", "NeuronKey", "NeuronMeta", "Phase", "RefineResult",
           "Status", "Witness", "split", "split_wave", "witness_phase",
           "verify_refine", "verify_refine_input_crown", "Selector",
           "FaithfulSelector", "LayerOrderSelector"]
