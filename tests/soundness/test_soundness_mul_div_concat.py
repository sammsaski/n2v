"""A Sigmoid regression for a star that carries its value in a pinned
predicate variable (found by the L3 real-model oracle on mscn).

The Mul and Div cases this file held (the lsnc, mscn and ml4acopf failure
classes: a McCormick product that zero-padded the smaller predicate system
to the larger one) called ``_mul_sets`` and ``_div_sets`` directly and
checked a bounding box with a flat 1e-5 slack. They were removed on
2026-11-17 as duplicates: ``test_soundness_set_arith_cells.py`` reaches the
same code through the model entry with a containment proof or exact points
(``test_mul_of_two_branches_is_proved``,
``test_product_or_quotient_on_a_wide_box``,
``test_product_or_quotient_of_a_node_with_itself``,
``test_div_of_two_branches_is_proved_or_refused``). The Concat cases of the
same failure class (``_join_star_systems``) are in
``test_soundness_concat.py`` and ``test_soundness_set_arith_cells.py``.
"""

import numpy as np

from n2v.sets import Star
from n2v.nn.layer_ops import sigmoid_reach
from tests._oracles import rational
from tests._oracles.interval_proof import SIGMOID, chain, prove


class TestSigmoidConstantStar:
    """Regression: a star can carry its (constant) value in constrained
    predicate variables with a ZERO center column — e.g. the z vars of
    a McCormick product. The all-constant branch used to apply the
    function to the center column and returned sigmoid(0) = 0.5
    regardless of the true value (mscn, dev 2.3e-01)."""

    def test_value_in_predicate_not_center(self):
        """x = alpha with alpha pinned at 0.98 and a zero centre: the
        returned set must hold sigmoid(0.98). Catches the value read from
        the centre column (sigmoid(0) = 0.5) on whichever path the kernel
        takes; today the outward pad of the pinned variable's range sends
        this star through the relaxation path, not the all-constant branch,
        where only rows without a generator arrive and the centre is the
        value."""
        val = 0.98
        V = np.array([[0.0, 1.0]])  # x = 0 + 1 * alpha
        s = Star(V, np.zeros((0, 1)), np.zeros((0, 1)),
                 np.array([[val]]), np.array([[val]]))  # alpha == val
        out = sigmoid_reach.sigmoid_star_approx([s], lp_solver='linprog')[0]
        # One-sided: the returned set must hold sigmoid(val) for the stored
        # float val, by a containment proof on the set's own rows in
        # rationals (an outward mpmath enclosure of the value). The old
        # answer, sigmoid(0) = 0.5, is refuted; a wider sound set passes.
        result = prove(chain(SIGMOID), [val], [val], rational.star(out))
        assert result.proved, result
