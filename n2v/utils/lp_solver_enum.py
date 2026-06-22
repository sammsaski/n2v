"""
LP solver enum and resolver (Layer 4 in the LP solver architecture).

This is a leaf module with zero intra-repo imports. It defines:
- :class:`Backend` - enumeration of backend engines (SCIPY, HIGHSPY, CVXPY, SENTINEL)
- :class:`LPSolver` - a ``str``-mixin enum enumerating every supported LP solver
- :func:`resolve` - coerces ``LPSolver | str | None`` to an ``LPSolver`` member

Because :class:`LPSolver` subclasses both ``str`` and ``Enum``, comparisons like
``LPSolver.LINPROG == 'linprog'`` remain ``True``, preserving backward
compatibility with existing notebooks, tests, and callers that pass raw
strings.
"""

from enum import Enum
from typing import Optional, Union


class Backend(Enum):
    """Backend engine that actually executes an LP solve."""

    SCIPY = 'scipy'         # scipy.optimize.linprog
    HIGHSPY = 'highspy'     # direct highspy C++ API
    CVXPY = 'cvxpy'         # CVXPY with a pluggable conic/LP solver
    SENTINEL = 'sentinel'   # placeholder (e.g. DEFAULT) - resolves elsewhere


class LPSolver(str, Enum):
    """Enumeration of supported LP solvers.

    Subclassing ``str`` means ``LPSolver.LINPROG == 'linprog'`` is ``True`` and
    the enum can be passed to any API expecting the legacy string names.

    Members
    -------
    DEFAULT
        Sentinel meaning "resolve from global config".
    LINPROG, HIGHS, HIGHS_DS, HIGHS_IPM
        Routed through ``scipy.optimize.linprog`` (or direct highspy when
        eligible for the batch hot path).
    CLARABEL, SCS, ECOS, OSQP, GUROBI, MOSEK, CBC, GLPK, GLPK_MI,
    CPLEX, COPT, XPRESS, PIQP, PROXQP, SCIP, NAG, CUOPT, DAQP, SDPA
        Routed through CVXPY using the solver's upper-case name.
    """

    DEFAULT = 'default'

    # SciPy-linprog family
    LINPROG = 'linprog'
    HIGHS = 'highs'
    HIGHS_DS = 'highs-ds'
    HIGHS_IPM = 'highs-ipm'

    # CVXPY-routed solvers (canonical names match cvxpy.solvers.*)
    CLARABEL = 'CLARABEL'
    SCS = 'SCS'
    ECOS = 'ECOS'
    OSQP = 'OSQP'
    GUROBI = 'GUROBI'
    MOSEK = 'MOSEK'
    CBC = 'CBC'
    GLPK = 'GLPK'
    GLPK_MI = 'GLPK_MI'
    CPLEX = 'CPLEX'
    COPT = 'COPT'
    XPRESS = 'XPRESS'
    PIQP = 'PIQP'
    PROXQP = 'PROXQP'
    SCIP = 'SCIP'
    NAG = 'NAG'
    CUOPT = 'CUOPT'
    DAQP = 'DAQP'
    SDPA = 'SDPA'

    # ------------------------------------------------------------------
    # Taxonomy via properties / methods (keeps members flat)
    # ------------------------------------------------------------------

    @property
    def backend(self) -> Backend:
        """Return the :class:`Backend` responsible for executing this solver."""
        if self is LPSolver.DEFAULT:
            return Backend.SENTINEL
        if self in _SCIPY_MEMBERS:
            return Backend.SCIPY
        return Backend.CVXPY

    @property
    def scipy_method(self) -> Optional[str]:
        """Method string to pass to ``scipy.optimize.linprog`` (or ``None``).

        ``LINPROG`` maps to ``'highs'`` to preserve legacy behavior where
        ``'linprog'`` routed through SciPy's default HiGHS method.
        """
        if self is LPSolver.LINPROG:
            return 'highs'
        if self in _SCIPY_MEMBERS:
            return self.value
        return None

    @property
    def cvxpy_name(self) -> Optional[str]:
        """Name to pass to ``cvxpy.Problem.solve(solver=...)`` (or ``None``)."""
        if self.backend is Backend.CVXPY:
            return self.value
        return None

    def is_scipy(self) -> bool:
        """True iff this solver is routed through ``scipy.linprog``."""
        return self.backend is Backend.SCIPY

    def is_cvxpy(self) -> bool:
        """True iff this solver is routed through CVXPY."""
        return self.backend is Backend.CVXPY

    def is_highspy_batch_eligible(self) -> bool:
        """True iff this solver may use the direct highspy batch hot path.

        Only ``LINPROG`` and ``HIGHS`` are eligible; ``HIGHS_DS`` / ``HIGHS_IPM``
        are routed through scipy to preserve the user's method choice.
        """
        return self in _HIGHSPY_BATCH_MEMBERS

    def is_sentinel(self) -> bool:
        """True iff this member is a placeholder that must resolve elsewhere."""
        return self is LPSolver.DEFAULT


# Module-level sets used by the properties above. Defined after the class so
# they can reference enum members directly.
_SCIPY_MEMBERS = frozenset({
    LPSolver.LINPROG,
    LPSolver.HIGHS,
    LPSolver.HIGHS_DS,
    LPSolver.HIGHS_IPM,
})

_HIGHSPY_BATCH_MEMBERS = frozenset({
    LPSolver.LINPROG,
    LPSolver.HIGHS,
})


# ------------------------------------------------------------------
# Resolver
# ------------------------------------------------------------------

# Case-insensitive lookup built once at import time. Keys are lower-cased
# values; members are looked up by ``LPSolver(value)`` which respects the
# original case.
_NAME_LOOKUP = {member.value.lower(): member for member in LPSolver}


def resolve(
    value: Union['LPSolver', str, None],
    *,
    hex_oct_alias: bool = False,
    allow_sentinel: bool = True,
) -> 'LPSolver':
    """Coerce ``value`` to an :class:`LPSolver` member.

    Parameters
    ----------
    value:
        An existing :class:`LPSolver` member, a legacy string name (case-
        insensitive), or ``None``. ``None`` maps to :attr:`LPSolver.DEFAULT`.
    hex_oct_alias:
        When ``True``, the string ``'lp'`` is accepted as an alias for
        :attr:`LPSolver.DEFAULT`. This mirrors the Hex/Oct ``solver='lp'``
        kwarg surface. Scheduled for removal in #10.
    allow_sentinel:
        When ``False``, a resolved value of :attr:`LPSolver.DEFAULT` is further
        resolved by reading ``n2v.config.config.lp_solver`` so callers never
        see the sentinel.

    Returns
    -------
    LPSolver
        The canonical enum member.

    Raises
    ------
    ValueError
        If ``value`` is a string that doesn't match any known solver name.
    TypeError
        If ``value`` is neither ``None``, ``str``, nor :class:`LPSolver`.
    """
    if value is None:
        solver = LPSolver.DEFAULT
    elif isinstance(value, LPSolver):
        solver = value
    elif isinstance(value, str):
        key = value.lower()
        if hex_oct_alias and key == 'lp':
            solver = LPSolver.DEFAULT
        elif key in _NAME_LOOKUP:
            solver = _NAME_LOOKUP[key]
        else:
            valid = ', '.join(sorted(m.value for m in LPSolver))
            raise ValueError(
                f"Unknown LP solver: {value!r}. Valid names (case-insensitive): {valid}"
            )
    else:
        raise TypeError(
            f"lp_solver must be LPSolver | str | None, got {type(value).__name__}"
        )

    if not allow_sentinel and solver is LPSolver.DEFAULT:
        # Resolve sentinel against the global config. Imported lazily to
        # avoid a circular import at module load time.
        from n2v.config import config as _config
        resolved = _config.lp_solver
        # config.lp_solver is itself an LPSolver (after step 4); guard against
        # a stringy leftover during migration.
        if isinstance(resolved, LPSolver):
            solver = resolved
        else:
            solver = resolve(resolved, hex_oct_alias=False, allow_sentinel=True)
            if solver is LPSolver.DEFAULT:
                # Defensive: avoid infinite recursion if config somehow still
                # holds the sentinel.
                solver = LPSolver.LINPROG

    return solver


__all__ = ['Backend', 'LPSolver', 'resolve']
