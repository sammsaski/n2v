"""A proof that a computed set contains the image of a box under a smooth
function, by interval branch and bound (PROTOCOL.md 3.2).

Sigmoid, Tanh, products and the like have no finite exact image, so the check
is a proof instead of a comparison. :func:`prove` covers the input box with
boxes. On each it encloses the function, and the box passes when one computed
set contains every vertex of the enclosure: a convex set that holds the
vertices holds the enclosure, and so the image of that part of the input. A
box that fails is cut in two.

    f = chain(linear(W, c), SIGMOID)                 # x -> sigmoid(W x + c)
    result = prove(f, lb, ub, rational.star(computed))
    assert result.proved, result

There are three outcomes.

- **proved**: every box passed. The computed sets contain the image.
- **refuted**: ``result.refuted`` is an input point whose value is certainly
  outside every computed set (the enclosure of the point's own value does not
  meet them). The computed sets are unsound, with a witness. Witnesses are
  looked for at the centre and the corners of every box that fails.
- **undecided**: boxes narrower than ``min_width`` still failed and no witness
  was found; they are in ``result.undecided``. This is a failure too. A sound
  set has positive thickness around the image except where a bound, a chord
  or a tangent meets it, and there the thickness is the kernel's outward
  rounding. A set with none touches the image. Where it touches along the end
  of an output range (the square of ``[1, 3]`` computed as exactly
  ``[1, 9]``) the proof still succeeds, through the interval enclosure
  below. Where a slanted face passes through one point of the graph (a
  relaxation line that is the tangent at the inflection point, or a chord
  with no rounding at all) no enclosure of any thickness fits, and the
  answer stays undecided. The same holds where a set ends exactly on the
  image and a slanted face passes the same point with less room than its
  slope times ``min_width``: the interval enclosure is a rectangle.

**Enclosures.** On a box with midpoint ``m`` every quantity is a
:class:`Form`, which encloses it twice, in rational numbers that hold for
every ``x`` of the box. The first is ``centre + slopes . (x - m) +
[low, high]``. A linear layer maps it exactly. A smooth scalar function ``g``
is applied by the mean value theorem, ``g(q) = g(q0) + g'(xi) (q - q0)``, with
``g`` and ``g'`` evaluated in mpmath interval arithmetic at :data:`PRECISION`
bits, rounded outward. Its remainder shrinks with the square of the box
width, so a tangent that touches the graph is reached in a few dozen boxes
instead of millions. The second is the interval ``span``, the quantity's
range by interval arithmetic. It is wider in the middle of a box and exact at
its ends, which the first never is. A box passes when one computed set holds
the vertices of the first enclosure, or of one in which some outputs are
replaced by their intervals. Membership is decided in rationals by
``rational.Membership``. Nothing here calls n2v.

The same function proves the first-stage check of the protocol, that the
relaxation polygon of one neuron contains the graph of the activation over
the neuron's range: ``prove(graph(chain(SIGMOID)), [l], [u], polygon)`` with
``polygon = rational.halfspaces(rows)``.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product

from mpmath import libmp
from mpmath.ctx_iv import MPIntervalContext

from tests._oracles.rational import (Membership, dot, feasible, matrix,
                                     meets_box, vector)

# Working precision of the interval arithmetic, in bits. Well above float64,
# so an enclosure is far thinner than one unit in the last place of a double.
PRECISION = 100

#: The interval context that functions are written in: ``IV.exp``, ``IV.sin``,
#: ``IV.cos``, ``IV.log``, ``IV.sqrt`` and arithmetic, all rounded outward.
#: It is private to this module; the global ``mpmath.iv`` is not touched.
IV = MPIntervalContext()
IV.prec = PRECISION

ZERO = Fraction(0)


# --------------------------------------------------------------------------- #
# Rationals to intervals and back                                               #
# --------------------------------------------------------------------------- #
def _interval(lo, hi):
    """An mpmath interval that contains ``[lo, hi]``."""
    low = IV.mpf(lo.numerator) / lo.denominator
    high = IV.mpf(hi.numerator) / hi.denominator
    return IV.mpf([low.a, high.b])


def _endpoints(x):
    """The two endpoints of an mpmath interval as exact rationals."""
    ends = []
    for raw in IV.mpf(x)._mpi_:
        if not raw[1] and raw[2]:           # mantissa 0, exponent set: inf, nan
            raise ArithmeticError(f"the enclosure {x} is not finite")
        ends.append(Fraction(*libmp.to_rational(raw)))
    return tuple(ends)


def _times(a, b):
    """The product of two rational intervals."""
    corners = [x * y for x in a for y in b]
    return min(corners), max(corners)


# --------------------------------------------------------------------------- #
# Forms and the stages that map them                                            #
# --------------------------------------------------------------------------- #
@dataclass
class Form:
    """One quantity on a box with midpoint ``m``, enclosed twice: it is
    ``centre + slopes . (x - m) + e`` for some ``e`` in ``[low, high]``, and
    it lies in the interval ``span``, for every ``x`` of the box."""
    centre: Fraction
    slopes: list
    low: Fraction = ZERO
    high: Fraction = ZERO
    span: tuple = None

    def range(self, radii):
        """The interval the linear part and its remainder allow."""
        reach = sum(abs(s) * r for s, r in zip(self.slopes, radii))
        return self.centre - reach + self.low, self.centre + reach + self.high

    def bounds(self, radii):
        """The interval both enclosures allow."""
        lo, hi = self.range(radii)
        return max(lo, self.span[0]), min(hi, self.span[1])


def _radii(box):
    return [(hi - lo) / 2 for lo, hi in box]


def _inputs(box):
    """The forms of the coordinates themselves."""
    n = len(box)
    return [Form((lo + hi) / 2, [Fraction(int(i == j)) for j in range(n)],
                 span=(lo, hi))
            for i, (lo, hi) in enumerate(box)]


def linear(W, c=None):
    """The stage ``q -> W q + c``. Exact: no remainder is added."""
    W = matrix(W)
    c = [ZERO] * len(W) if c is None else vector(c)

    def stage(box, forms):
        n = len(box)
        out = []
        for row, offset in zip(W, c):
            ends = [_times((w, w), (f.low, f.high)) for w, f in zip(row, forms)]
            spans = [_times((w, w), f.span) for w, f in zip(row, forms)]
            out.append(Form(
                offset + sum(w * f.centre for w, f in zip(row, forms)),
                [sum(w * f.slopes[i] for w, f in zip(row, forms))
                 for i in range(n)],
                sum(lo for lo, _ in ends), sum(hi for _, hi in ends),
                (offset + sum(lo for lo, _ in spans),
                 offset + sum(hi for _, hi in spans))))
        return out
    return stage


def smooth(g, dg):
    """The stage that applies a differentiable scalar function to every
    quantity. ``g`` and ``dg`` (its derivative) take and return an mpmath
    interval and are written with :data:`IV` and ordinary arithmetic; any such
    expression is a rigorous enclosure."""
    def stage(box, forms):
        radii = _radii(box)
        out = []
        for form in forms:
            taken = form.bounds(radii)
            # The expansion point q0 is the centre, which the quantity itself
            # need not take: the derivative is enclosed up to it.
            lo, hi = min(taken[0], form.centre), max(taken[1], form.centre)
            at_centre = _endpoints(g(_interval(form.centre, form.centre)))
            d_lo, d_hi = _endpoints(dg(_interval(lo, hi)))
            slope = (d_lo + d_hi) / 2
            # g(q) = g(q0) + slope (q - q0) + (g'(xi) - slope) (q - q0)
            rest = _times((d_lo - slope, d_hi - slope),
                          (lo - form.centre, hi - form.centre))
            carried = _times((slope, slope), (form.low, form.high))
            out.append(Form(
                at_centre[0], [slope * s for s in form.slopes],
                carried[0] + rest[0],
                at_centre[1] - at_centre[0] + carried[1] + rest[1],
                _endpoints(g(_interval(*taken)))))
        return out
    return stage


def function(f):
    """The stage of any function written in interval arithmetic: ``f`` takes a
    list of mpmath intervals (the ranges of the quantities) and returns a list
    of them. Rigorous for every such ``f``, with a remainder that shrinks only
    in proportion to the box width; enough for a computed set with a margin
    around the image."""
    def stage(box, forms):
        radii = _radii(box)
        values = f([_interval(*form.bounds(radii)) for form in forms])
        out = []
        for value in values:
            lo, hi = _endpoints(value)
            out.append(Form(lo, [ZERO] * len(box), ZERO, hi - lo, (lo, hi)))
        return out
    return stage


def _sigmoid(t):
    return 1 / (1 + IV.exp(-t))


def _tanh(t):
    return 1 - 2 / (IV.exp(2 * t) + 1)


def _falling_with_distance(at):
    """The interval function of a function that is even and decreasing in
    ``|t|``, from its values at single points: on an interval it ranges from
    its value at the point farthest from zero to its value at the nearest.
    The expression itself, evaluated on the interval, is rigorous too but
    much wider, because each occurrence of ``t`` then varies on its own."""
    def on(t):
        size = abs(IV.mpf(t))
        return IV.mpf([at(IV.mpf(size.b)).a, at(IV.mpf(size.a)).b])
    return on


SIGMOID = smooth(_sigmoid, _falling_with_distance(
    lambda t: _sigmoid(t) * (1 - _sigmoid(t))))
TANH = smooth(_tanh, _falling_with_distance(lambda t: 1 - _tanh(t) ** 2))
EXP = smooth(lambda t: IV.exp(t), lambda t: IV.exp(t))
SIN = smooth(lambda t: IV.sin(t), lambda t: IV.cos(t))
COS = smooth(lambda t: IV.cos(t), lambda t: -IV.sin(t))


def power(p):
    """The stage ``q -> q ** p`` for a whole number ``p`` of at least 2."""
    if p != int(p) or p < 2:
        raise ValueError(f"the exponent must be a whole number >= 2, not {p}")
    p = int(p)
    return smooth(lambda t: t ** p, lambda t: p * t ** (p - 1))


def chain(*stages):
    """The function that applies the stages in order to the input: an
    enclosure, a callable from a box to the forms of the outputs."""
    def enclose(box):
        forms = _inputs(box)
        for stage in stages:
            forms = stage(box, forms)
        return forms
    return enclose


def graph(enclose):
    """``x -> (x, f(x))``: the graph of the enclosure ``f``."""
    return lambda box: _inputs(box) + enclose(box)


def value(enclose, x):
    """A box that holds the function's value at the point ``x``, as a pair of
    rational vectors ``(low, high)``."""
    x = vector(x)
    ends = [f.bounds([ZERO] * len(x)) for f in enclose([(v, v) for v in x])]
    return [lo for lo, _ in ends], [hi for _, hi in ends]


# --------------------------------------------------------------------------- #
# The proof                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    """The outcome of :func:`prove`."""
    proved: bool
    refuted: list = None                 # an input point, when one was found
    undecided: list = field(default_factory=list)       # boxes left failing
    boxes: int = 0                       # how many boxes were examined

    def __str__(self):
        if self.proved:
            return f"proved with {self.boxes} boxes"
        if self.refuted is not None:
            point = ", ".join(f"{float(v):.17g}" for v in self.refuted)
            return f"refuted at the input ({point}) after {self.boxes} boxes"
        where = ", ".join(f"[{float(lo):.17g}, {float(hi):.17g}]"
                          for lo, hi in self.undecided[0])
        return (f"undecided: {len(self.undecided)} box(es) failed below the "
                f"minimum width, the first at {where}")


def _vertices(forms, radii):
    """Points whose convex hull holds every value the forms allow."""
    moves = [(-r, r) if r else (ZERO,) for r in radii]
    slack = [(f.low, f.high) if f.low != f.high else (f.low,) for f in forms]
    points = set()
    for move in product(*moves):
        base = [f.centre + dot(f.slopes, move) for f in forms]
        for extra in product(*slack):
            points.add(tuple(b + e for b, e in zip(base, extra)))
    return points


def _enclosures(forms, radii):
    """The vertex sets of the enclosures to try on one box. First every
    output by its linear part and remainder. Then, for the outputs whose
    interval is tighter than that at an end, each choice of them replaced by
    their interval: the image lies in every such mixture."""
    yield _vertices(forms, radii)
    flat = [ZERO] * len(radii)
    tighter = [k for k, f in enumerate(forms)
               if f.bounds(radii) != f.range(radii)]
    for choice in product((True, False), repeat=len(tighter)):
        if not any(choice):
            continue
        mixed = list(forms)
        for k, by_span in zip(tighter, choice):
            if by_span:
                lo, hi = forms[k].span
                mixed[k] = Form(lo, flat, ZERO, hi - lo, (lo, hi))
        yield _vertices(mixed, radii)


def _contained(holders, forms, radii):
    """Does one of the computed sets hold a whole enclosure of the box?"""
    return any(all(holder(y) for y in points)
               for points in _enclosures(forms, radii) for holder in holders)


def _certainly_outside(enclose, computed, holders, x):
    """Is the value at the point ``x`` outside every computed set?"""
    low, high = value(enclose, x)
    middle = [(lo + hi) / 2 for lo, hi in zip(low, high)]
    if any(holder(middle) for holder in holders):
        return False
    return not any(meets_box(s, low, high) for s in computed)


def _axis_to_cut(enclose, box):
    """The axis that, collapsed to its midpoint, leaves the smallest
    remainder: the one whose length the remainder comes from. Among equals the
    longest. An axis the function does not depend on is never cut."""
    best, best_size = None, None
    for i, (lo, hi) in enumerate(box):
        if lo == hi:
            continue
        mid = (lo + hi) / 2
        forms = enclose(box[:i] + [(mid, mid)] + box[i + 1:])
        size = (sum(f.high - f.low for f in forms), lo - hi)
        if best is None or size < best_size:
            best, best_size = i, size
    return best


def prove(enclose, lb, ub, *computed, domain=None,
          min_width=Fraction(1, 2 ** 40), max_boxes=20000):
    """Prove that the computed sets contain ``{f(x) : lb <= x <= ub}``.

    ``enclose`` is the function, built with :func:`chain`. Each of
    ``computed`` is a set in the form of ``rational.star``. ``domain``, a pair
    ``(A, b)``, restricts the input to the part of the box with ``A x <= b``:
    boxes that miss it are skipped, and a box that straddles its boundary is
    treated whole, so the computed set needs a margin there.

    A box that fails at a width of ``min_width`` or less on the axis it would
    be cut along is given up and reported in ``Result.undecided``. More than
    ``max_boxes`` boxes raises: the proof is too expensive as posed."""
    if not computed:
        raise ValueError("no computed set to compare with")
    lb, ub = vector(lb), vector(ub)
    if domain is not None:
        domain = (matrix(domain[0]), vector(domain[1]))
    n = len(lb)
    unit = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    box_rows = unit + [[-v for v in row] for row in unit]
    holders = [Membership(s) for s in computed]

    def in_domain(x):
        return domain is None or all(
            dot(row, x) <= rhs for row, rhs in zip(*domain))

    def meets_domain(box):
        if domain is None:
            return True
        rhs = [hi for _, hi in box] + [-lo for lo, _ in box]
        return feasible((), (), box_rows + domain[0], rhs + domain[1]) \
            is not None

    stack, undecided, examined = [list(zip(lb, ub))], [], 0
    while stack:
        box = stack.pop()
        if not meets_domain(box):
            continue
        examined += 1
        if examined > max_boxes:
            raise RuntimeError(
                f"no answer after {max_boxes} boxes; the last was {box}")
        if _contained(holders, enclose(box), _radii(box)):
            continue
        # A witness is looked for at the centre and at the corners. A set
        # that is too small by less than any box will be cut to is too small
        # at a corner of some box: at an end of a range, or along a face.
        corners = product(*[sorted(set(side)) for side in box])
        for point in [[(lo + hi) / 2 for lo, hi in box]] + list(corners):
            point = list(point)
            if in_domain(point) and \
                    _certainly_outside(enclose, computed, holders, point):
                return Result(False, point, [], examined)
        axis = _axis_to_cut(enclose, box)
        if axis is None or box[axis][1] - box[axis][0] <= min_width:
            undecided.append(box)
            continue
        lo, hi = box[axis]
        mid = (lo + hi) / 2
        stack.append(box[:axis] + [(mid, hi)] + box[axis + 1:])
        stack.append(box[:axis] + [(lo, mid)] + box[axis + 1:])
    return Result(not undecided, None, undecided, examined)
