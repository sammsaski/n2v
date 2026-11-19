"""Oracles the soundness and contract tests are checked against.

Nothing here calls an n2v kernel, solver wrapper or set method. The modules
import the set classes only to recognise them, read a computed set's stored
arrays (basis, constraint rows, predicate bounds) and decide the question
themselves, so a bug in n2v cannot agree with itself. ``rational.star`` reads
a set through ``membership.rows``, so the exact oracle accepts the same seven
set types.
The protocol the three oracles implement is
``.claude/research/34-API-Redesign/test-audit/PROTOCOL.md`` section 3:

``rational``
    Piecewise-linear operations: the exact image of a polytope as a union of
    vertex-described pieces in ``fractions.Fraction``, and exact membership of
    a rational point in a computed star (section 3.1).
``interval_proof``
    Smooth operations: a containment proof by interval branch and bound, with
    the function enclosed by mpmath interval arithmetic (section 3.2).
``membership``
    Point replay: is a float point a member of a computed set, by one LP on
    the set's own rows (section 3.3). It reads every set type n2v returns:
    Star, ImageStar, Zono, ImageZono, Box, Hexatope (difference rows, anchor
    pinned to 0) and Octatope (UTVPI rows), and stays reliable on predicate
    boxes as narrow as the solver tolerance by solving in scaled variables
    and checking the solver's point itself.

Each module has a self-test that plants a bug in a correct enclosure and
watches the oracle reject it; an oracle that accepts everything is the one
failure these tests cannot afford.
"""
