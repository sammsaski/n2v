# Issue #9 — LP Solver Enum: Layered Architecture

Horizontal view of how an LP-solver selection flows from the user down to `n2v/utils/lpsolver.py`. Each layer is a file or group of files. The **coercion boundary** is where `str | None | LPSolver` becomes an `LPSolver` enum; everything below it speaks enum only.

```
+--------------------------------------------------------------------------------------------+
|  LAYER 0  -  USER / PUBLIC API                            (accepts str OR LPSolver)        |
|  notebooks, scripts, benchmarks/benchmark_time.py, tests                                   |
|  e.g.  star.get_ranges(lp_solver='highs')                                                  |
|        star.get_ranges(lp_solver=LPSolver.HIGHS)                                           |
+--------------------------------------------------------------------------------------------+
              |                                                             |
              v                                                             v
+---------------------------------------+        +-------------------------------------------+
|  LAYER 1  -  CONFIG                   |        |  LAYER 1  -  VNNCOMP HARNESS              |
|  n2v/config.py                        |        |  n2v/utils/vnncomp.py                     |
|  - stores _default_lp_solver (enum)   |        |  - lp_solver: LPSolver field              |
|  - setter: resolve(str|enum) -> enum  |        |  - to_dict() emits .value (str)           |
|  - get_config() emits .value (str)    |        +-------------------------------------------+
+---------------------------------------+                           |
              |                                                     |
              v                                                     v
+--------------------------------------------------------------------------------------------+
|  LAYER 2  -  SET CLASSES (public entry points, COERCE HERE)                                |
|  n2v/sets/star.py          : get_box, get_range, get_min, get_max, get_ranges,             |
|                              get_max_indexes, contains, is_empty_set                       |
|  n2v/sets/image_star.py    : get_ranges, get_range, get_range_flat, contains,              |
|                              is_empty_set                                                  |
|  n2v/sets/hexatope.py      : get_ranges (solver='lp'|'mcf'|None)  -> hex_oct_alias=True    |
|  n2v/sets/octatope.py      : get_ranges (solver='lp'|'mcf'|None)  -> hex_oct_alias=True    |
|                                                                                            |
|      solver = resolve(lp_solver)                        <-- COERCION BOUNDARY              |
|      # below this line: enum only                                                          |
+--------------------------------------------------------------------------------------------+
              |                                                     |
              v                                                     v
+---------------------------------------+        +-------------------------------------------+
|  LAYER 3  -  LAYER-OPS DISPATCHER     |        |  LAYER 3  -  REACH OPS (pass-through)     |
|  n2v/nn/layer_ops/dispatcher.py       |        |  n2v/nn/layer_ops/                        |
|  - pulls 'lp_solver' from kwargs      |        |    relu_reach, leakyrelu_reach,           |
|  - coerces ONCE via resolve()         |        |    sigmoid_reach, tanh_reach,             |
|  - forwards enum to set methods       |        |    sign_reach, maxpool2d_reach            |
+---------------------------------------+        |  - type hint: LPSolver | str              |
                                                 |  - forwards unchanged to Star/ImageStar   |
                                                 +-------------------------------------------+
              \                                                     /
               \                                                   /
                v                                                 v
+--------------------------------------------------------------------------------------------+
|  LAYER 4  -  ENUM + RESOLVER (the coercion layer itself)                                   |
|  n2v/utils/lp_solver_enum.py  (NEW)                                                        |
|                                                                                            |
|    class Backend(Enum):  SCIPY, HIGHSPY, CVXPY, SENTINEL                                   |
|                                                                                            |
|    class LPSolver(str, Enum):                                                              |
|        DEFAULT, LINPROG, HIGHS, HIGHS_DS, HIGHS_IPM,                                       |
|        CLARABEL, SCS, ECOS, OSQP, GUROBI, MOSEK, CBC, GLPK, GLPK_MI,                       |
|        CPLEX, COPT, XPRESS, PIQP, PROXQP, SCIP, NAG, CUOPT, DAQP, SDPA                     |
|        .backend  .scipy_method  .cvxpy_name                                                |
|        .is_scipy() .is_cvxpy() .is_highspy_batch_eligible() .is_sentinel()                 |
|                                                                                            |
|    def resolve(value, *, hex_oct_alias=False, allow_sentinel=True) -> LPSolver             |
+--------------------------------------------------------------------------------------------+
                                          |
                                          v
+--------------------------------------------------------------------------------------------+
|  LAYER 5  -  LP CORE                                                                       |
|  n2v/utils/lpsolver.py                                                                     |
|                                                                                            |
|    solve_lp(...)        -> coerce once, dispatch by solver.backend                         |
|    solve_lp_batch(...)  -> coerce once, dispatch by solver.backend                         |
|    check_feasibility(...)                                                                  |
|                                                                                            |
|    if solver.is_highspy_batch_eligible():  -> _solve_lp_batch_highspy                      |
|    elif solver.is_scipy():                 -> scipy.linprog(method=solver.scipy_method)    |
|    elif solver.is_cvxpy():                 -> cp.Problem(...).solve(solver=cvxpy_name)     |
|    elif solver is LPSolver.DEFAULT:        -> resolve(config.lp_solver)                    |
+--------------------------------------------------------------------------------------------+
              |                   |                        |
              v                   v                        v
+----------------------+   +---------------+   +--------------------------------+
|  LAYER 6  -  HIGHSPY |   |  scipy.optim  |   |  LAYER 6  -  CVXPY             |
|  highspy (direct)    |   |  .linprog     |   |  cvxpy -> GUROBI / MOSEK /     |
|  batch hot path      |   |  method=...   |   |  CLARABEL / SCS / ECOS / ...   |
+----------------------+   +---------------+   +--------------------------------+
```

## Reading the diagram

- **Arrows = data flow** (a solver selection travelling downward).
- The **coercion boundary** lives at the top of every Layer-2 public method and at the Layer-3 dispatcher. Nothing below Layer 2 ever sees a raw string.
- Layer 4 is a **leaf module** — it depends on nothing in the repo, so Layers 1, 2, 3, 5 can all import it without cycles.
- Layer 5 (`lpsolver.py`) still has a defensive `resolve()` at its own entry (cheap, idempotent) so direct callers from tests don't have to coerce.
- Hexatope / Octatope keep their `solver='lp'|'mcf'|None` kwarg surface; only the inner LP branch crosses into the enum world (`hex_oct_alias=True` maps `'lp'` -> `LPSolver.DEFAULT`). Full kwarg unification is deferred to issue #10.

## Invariants this design enforces

1. **One coercion per call path.** A value is coerced at Layer 1 (config) or Layer 2 (set methods) and then flows as an enum; Layers 3-5 never re-parse strings.
2. **No circular imports.** `lp_solver_enum.py` has zero intra-repo imports.
3. **Backward compatibility via `str` mixin.** `LPSolver.LINPROG == 'linprog'` is `True`, so existing notebook/test code that compares to raw strings still works.
4. **Backend-specific options stay on the enum.** `scipy_method`, `cvxpy_name`, `is_highspy_batch_eligible()` live as properties/methods, so Layer 5's dispatch is a clean `match solver.backend:` rather than re-interpreting strings.