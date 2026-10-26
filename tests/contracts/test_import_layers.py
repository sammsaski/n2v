"""Import layering contract (DESIGN section 3).

A module imports n2v code only from its own target package or from a lower
layer; underscore names cross a package boundary only into the numerics,
profiling and budget helpers; no cross-package import cycle exists at runtime;
and a lazy import never hides a cycle without being counted. Everything here
is decided by reading the source with ``ast``: nothing under ``n2v`` is
imported, so the test cannot pass because an import happened to succeed.

The tree is still the pre-redesign one, so every module is mapped to its
target package through ``MOVES`` before the layer rule is applied. Known
violations live in ``ALLOWLIST`` with the step of PLAN.md that removes them;
the test fails on any violation that is not listed and on any listed entry
that no longer matches, so the allowlist can only shrink. A planted package
in ``tmp_path`` checks that every import form is detected.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# Target layers (DESIGN section 3): a module may import its own package or any
# package of strictly lower rank. ``nn`` is the kept Ben Wooding / ViT lane,
# ranked between verification and refine so refine's planned re-export of
# nn/bab.py is layer-legal.
# --------------------------------------------------------------------------- #
RANK = {
    "errors": 0, "_profiling": 0,
    "config": 1,
    "_numerics": 2, "_budget": 2,
    "sets": 3,
    "specs": 4,
    "results": 5, "options": 5,
    "models": 6,
    "reachability": 7,
    "verification": 8,
    "nn": 8.5,
    "refine": 9, "probabilistic": 9,
    "top": 10, "_api": 10,
    "cli": 11,
}

# Underscore names may cross a package boundary only into these packages.
PRIVATE_OK_TARGETS = {"_numerics", "_profiling", "_budget"}

# Today's module (or package prefix) -> target package. The longest matching
# prefix wins. Package ``__init__`` modules are named by the package itself.
# After PLAN step 24 this table collapses to the identity and is deleted.
MOVES = {
    "n2v": "top",
    "n2v.config": "config",
    "n2v.profiling": "_profiling",
    "n2v.utils": "top",                       # the legacy facade; lazy forwards after step 24
    "n2v.utils.lp_solver_enum": "config",
    "n2v.utils.lpsolver": "_numerics",
    "n2v.utils.lpsolver_gpu": "_numerics",
    "n2v.utils.ns_certificate": "_numerics",
    "n2v.utils.load_vnnlib": "specs",
    "n2v.utils.vnnlib2": "specs",
    "n2v.utils.model_loader": "models",
    "n2v.utils.model_preprocessing": "models",
    "n2v.utils.quant_converters": "models",
    "n2v.utils.bounds_precomputation": "reachability",
    "n2v.utils.falsify": "verification",
    "n2v.utils.onnx_validate": "verification",
    "n2v.utils.verify_nonlinear": "verification",
    "n2v.utils.verify_specification": "verification",
    "n2v.utils.vnncomp": "_api",              # deleted in PLAN step 1
    "n2v.sets": "sets",
    "n2v.sets.halfspace": "specs",
    "n2v.sets.probabilistic_box": "probabilistic",
    "n2v.nn": "nn",                           # package init, bab.py, vit_crown.py, crown_reach.py
    "n2v.nn.neural_network": "models",
    "n2v.nn.reach": "reachability",
    "n2v.nn.layer_ops": "reachability",
    "n2v.nn.relational": "verification",
    "n2v.refine": "refine",
    "n2v.probabilistic": "probabilistic",
    "n2v.probabilistic.flow.star_viz": "sets",   # -> sets/plot.py
}

# Outside the package, only the private-name rule applies. ``examples/Research``
# is gitignored and stays out (DESIGN section 20 leaves it to the owner).
CONSUMER_DIRS = ["benchmarks", "examples/Submission/VNN_COMP2026"]

# Third-party modules that ``import n2v`` must not pull in eagerly (report only
# until PLAN step 24, DESIGN section 3 rule 7).
HEAVY_THIRD_PARTY = {"onnx2torch", "cvxpy", "plotly", "torchdiffeq", "onnxruntime"}

# Known violations: (rule, src module, dst module, kind) -> (reason, removed by).
# ``kind`` is "eager", "lazy", "conditional", "type" or "dynamic"; "*" matches
# any kind. Private-name entries use rule "private" and list the names.
ALLOWLIST = [
    # --- layer rule -------------------------------------------------------
    ("layer", "n2v.sets", "n2v.sets.halfspace", "eager",
     "HalfSpace re-exported from n2v.sets; a deprecated name until 0.3", "step 28"),
    ("layer", "n2v.sets", "n2v.sets.probabilistic_box", "eager",
     "ProbabilisticBox re-exported from n2v.sets; a deprecated name until 0.3", "step 28"),
    ("layer", "n2v.nn.neural_network", "n2v.nn.reach", "eager",
     "NeuralNetwork.reach forwards to reach_pytorch_model until 0.3; _function_node_to_module needs a home at L2 or below", "step 24"),
    ("layer", "n2v.nn.reach", "n2v.probabilistic.conformal_reach", "eager",
     "reach imports the probabilistic methods eagerly; the permanent exception is a lazy import", "step 16"),
    ("layer", "n2v.nn.reach", "n2v.probabilistic.flow.reach", "eager",
     "reach imports the probabilistic methods eagerly; the permanent exception is a lazy import", "step 16"),
    ("layer", "n2v.utils.verify_specification", "n2v.probabilistic.flow.sets", "eager",
     "the probabilistic branch of verify_specification moves to probabilistic/certification.py", "step 13"),
    ("layer", "n2v.utils.verify_specification", "n2v.probabilistic.flow.scenario_verify", "lazy",
     "the probabilistic branch of verify_specification moves to probabilistic/certification.py", "step 13"),
    ("layer", "n2v.utils.verify_specification", "n2v.probabilistic.flow.amls", "lazy",
     "the probabilistic branch of verify_specification moves to probabilistic/certification.py", "step 13"),
    ("layer", "n2v.utils.verify_specification", "n2v.probabilistic.flow.amls_bounded", "lazy",
     "the probabilistic branch of verify_specification moves to probabilistic/certification.py", "step 13"),
    ("layer", "n2v.utils.verify_specification", "n2v.probabilistic.flow.importance_sampling", "lazy",
     "the probabilistic branch of verify_specification moves to probabilistic/certification.py", "step 13"),
    ("layer", "n2v.utils.verify_specification", "n2v.probabilistic.flow.raw_mc_uniform", "lazy",
     "the probabilistic branch of verify_specification moves to probabilistic/certification.py", "step 13"),
    ("layer", "n2v.utils.verify_nonlinear", "n2v.nn", "lazy",
     "NeuralNetwork imported through the n2v.nn package init, which pulls reach; re-point to n2v.models", "step 24"),
    ("layer", "n2v.nn.relational", "n2v.nn", "eager",
     "NeuralNetwork imported through the n2v.nn package init, which pulls reach; re-point to n2v.models", "step 24"),
    # --- private names across packages -------------------------------------
    ("private", "n2v.nn.neural_network", "n2v.nn.reach", "eager",
     "_function_node_to_module (models -> reachability) needs a home at L2 or below", "step 24"),
    ("private", "n2v.utils.verify_specification", "n2v.nn.layer_ops.flatten_reach", "lazy",
     "_flatten_imagezono_chw: the ImageZono reorder needs a home at L1 or below before the sound branch moves to sets/_disjoint.py", "step 12"),
    ("private", "n2v.refine.reach_relaxed", "n2v.nn.layer_ops.avgpool2d_reach", "lazy",
     "_check_ceil_mode_divisor shared with the AvgPool kernel; becomes a public helper of the layer registry", "step 23"),
    ("private", "n2v.utils.verify_nonlinear", "n2v.utils.vnnlib2", "eager",
     "_COMPARE, the 2.0 parser's relational-operator table, shared with the nonlinear verifier; becomes part of specs/_expressions.py", "step 10"),
    ("private", "benchmarks/vit_vnncomp2023/reach.py", "n2v.nn.reach", "eager",
     "_add_sets used by the ViT benchmark script; replaced by the public set arithmetic of DESIGN section 8", "step 16"),
    ("private", "examples/Submission/VNN_COMP2026/vnncomp_runner.py", "n2v.utils.falsify", "eager",
     "_extract_halfspace_groups used by the runner; replaced by Spec.to_polytope()", "step 20"),
]

# Cross-package import cycles that exist today at runtime, each given by
# modules it must contain. Any other cross-package cycle fails.
ALLOWED_CYCLES = [
    ({"n2v.config", "n2v.utils", "n2v.sets", "n2v.sets.star"},
     "config -> utils/__init__ -> load_vnnlib -> sets/__init__ -> star -> config: latent (n2v/__init__ runs first), "
     "gone when utils/__init__ becomes lazy forwards and lp_solver_enum merges into config.py", "step 24"),
]

# Lazy imports that close a cycle, counted per target package of the importer
# and frozen: a new one must be a conscious decision recorded here.
LAZY_CYCLE_CLOSERS = {
    "config": 1,         # lp_solver_enum:211 -> config (closes config -> lp_solver_enum); gone when they merge
    "reachability": 3,   # bounds_precomputation -> nn.reach x3 (closes nn.reach -> bounds_precomputation)
    "sets": 12,          # star/box/zono/image_star/predicate_box converting into each other
}

# Edges that must be found for the scanner to be trusted on the real tree.
SENTINEL_EDGES = [
    ("n2v.config", "n2v.utils.lp_solver_enum", "eager"),
    ("n2v.nn.reach", "n2v.utils.bounds_precomputation", "eager"),
    ("n2v.utils.verify_specification", "n2v.nn.layer_ops.flatten_reach", "lazy"),
]


# --------------------------------------------------------------------------- #
# Scanner
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Edge:
    src: str          # importing module (dotted) or consumer path
    dst: str          # imported module (dotted, inside the scanned package)
    kind: str         # eager | lazy | conditional | type | dynamic
    line: int
    names: tuple = () # names imported with ``from dst import names``


@dataclass
class Scan:
    package: str
    modules: set = field(default_factory=set)        # dotted names of every module file
    packages: set = field(default_factory=set)       # dotted names that have an __init__
    edges: list = field(default_factory=list)
    third_party: dict = field(default_factory=dict)  # module -> set of top-level third-party names (eager)
    dynamic_nonliteral: list = field(default_factory=list)   # (module, line)
    forwards: set = field(default_factory=set)       # modules declaring _FORWARDS


def _module_name(root: Path, package: str, path: Path) -> str:
    rel = path.relative_to(root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _enclosing_kind(parents: list) -> str:
    for node in reversed(parents):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return "lazy"
    for node in parents:
        if isinstance(node, ast.If):
            test = node.test
            name = getattr(test, "id", None) or getattr(test, "attr", None)
            if name == "TYPE_CHECKING":
                return "type"
    for node in parents:
        if isinstance(node, (ast.If, ast.Try, ast.With)):
            return "conditional"
    return "eager"


def _walk_with_parents(tree):
    stack = [(tree, [])]
    while stack:
        node, parents = stack.pop()
        yield node, parents
        for child in ast.iter_child_nodes(node):
            stack.append((child, parents + [node]))


def _resolve_from(module: str, is_package: bool, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    base = module.split(".") if is_package else module.split(".")[:-1]
    up = node.level - 1
    if up > len(base):
        return None
    base = base[:len(base) - up] if up else base
    return ".".join(base + ([node.module] if node.module else []))


def scan_tree(root: Path, package: str, consumers: list[Path] = ()) -> Scan:
    """Read every ``.py`` under ``root`` (a package directory) and the consumer
    files, recording every import of ``package`` code with its kind."""
    scan = Scan(package=package)
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    for path in files:
        mod = _module_name(root, package, path)
        scan.modules.add(mod)
        if path.name == "__init__.py":
            scan.packages.add(mod)
    sources = [(_module_name(root, package, p), p, _module_name(root, package, p) in scan.packages)
               for p in files]
    for cpath in consumers:
        for p in sorted(Path(cpath).rglob("*.py")):
            if "__pycache__" not in p.parts:
                sources.append((str(p.relative_to(REPO)), p, False))

    for mod, path, is_pkg in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        third = set()
        for node, parents in _walk_with_parents(tree):
            if isinstance(node, ast.Assign) and not parents[1:]:
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "_FORWARDS":
                        scan.forwards.add(mod)
            if isinstance(node, ast.Import):
                kind = _enclosing_kind(parents)
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top == package:
                        scan.edges.append(Edge(mod, alias.name, kind, node.lineno))
                    elif kind in ("eager", "conditional"):
                        third.add(top)
            elif isinstance(node, ast.ImportFrom):
                kind = _enclosing_kind(parents)
                target = _resolve_from(mod, is_pkg, node)
                if target is None:
                    continue
                top = target.split(".")[0]
                if top != package:
                    if kind in ("eager", "conditional") and node.level == 0:
                        third.add(top)
                    continue
                plain = []
                for alias in node.names:
                    sub = f"{target}.{alias.name}"
                    if sub in scan.modules:
                        scan.edges.append(Edge(mod, sub, kind, node.lineno))
                    else:
                        plain.append(alias.name)
                if plain or not node.names:
                    scan.edges.append(Edge(mod, target, kind, node.lineno, tuple(plain)))
            elif isinstance(node, ast.Call):
                fn = node.func
                is_import_module = (isinstance(fn, ast.Attribute) and fn.attr == "import_module"
                                    and getattr(fn.value, "id", None) == "importlib")
                is_dunder = isinstance(fn, ast.Name) and fn.id == "__import__"
                if is_import_module or is_dunder:
                    arg = node.args[0] if node.args else None
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        if arg.value.split(".")[0] == package:
                            scan.edges.append(Edge(mod, arg.value, "dynamic", node.lineno))
                    else:
                        scan.dynamic_nonliteral.append((mod, node.lineno))
        if mod in scan.modules:
            scan.third_party[mod] = third
    return scan


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def home(module: str, moves: dict) -> str | None:
    """Target package of ``module`` (longest matching prefix in ``moves``)."""
    parts = module.split(".")
    for i in range(len(parts), 0, -1):
        key = ".".join(parts[:i])
        if key in moves:
            return moves[key]
    return None


def _is_private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


@dataclass
class Report:
    layer: list = field(default_factory=list)        # (src, dst, kind, line, src_pkg, dst_pkg)
    private: list = field(default_factory=list)      # (src, dst, kind, line, names)
    cycles: list = field(default_factory=list)       # sets of modules (cross-package SCCs)
    lazy_closers: dict = field(default_factory=dict) # package -> count
    lazy_closer_edges: list = field(default_factory=list)
    homeless: list = field(default_factory=list)
    dynamic_nonliteral: list = field(default_factory=list)


def analyse(scan: Scan, moves: dict, rank: dict) -> Report:
    rep = Report()
    for mod in sorted(scan.modules):
        if home(mod, moves) is None:
            rep.homeless.append(mod)
    for e in scan.edges:
        src_pkg = home(e.src, moves) if e.src in scan.modules else None
        dst_pkg = home(e.dst, moves)
        if dst_pkg is None:
            continue
        if src_pkg is not None and src_pkg != dst_pkg and rank[dst_pkg] >= rank[src_pkg]:
            rep.layer.append((e.src, e.dst, e.kind, e.line, src_pkg, dst_pkg))
        private_names = tuple(n for n in e.names if _is_private(n))
        private_module = any(_is_private(p) for p in e.dst.split("."))
        crosses = src_pkg != dst_pkg
        if crosses and dst_pkg not in PRIVATE_OK_TARGETS and (private_names or private_module):
            rep.private.append((e.src, e.dst, e.kind, e.line, private_names))
    rep.dynamic_nonliteral = [(m, l) for m, l in scan.dynamic_nonliteral if m not in scan.forwards]

    # Runtime graph: eager and conditional edges, plus the parent-package
    # __init__ edges that importing a submodule from outside its package
    # triggers (the root package is excluded: it always runs first).
    runtime = {m: set() for m in scan.modules}
    for e in scan.edges:
        if e.src not in scan.modules or e.kind not in ("eager", "conditional"):
            continue
        if e.dst in scan.modules:
            runtime[e.src].add(e.dst)
        parts = e.dst.split(".")
        for i in range(2, len(parts)):
            parent = ".".join(parts[:i])
            if parent in scan.packages and not (e.src == parent or e.src.startswith(parent + ".")):
                runtime[e.src].add(parent)
    for scc in _sccs(runtime):
        if len(scc) > 1 and len({home(m, moves) for m in scc}) > 1:
            rep.cycles.append(set(scc))

    # Lazy imports that close a cycle: the target reaches the importer over
    # explicit edges of every kind (never over package-init edges).
    explicit = {m: set() for m in scan.modules}
    for e in scan.edges:
        if e.src in scan.modules and e.dst in scan.modules:
            explicit[e.src].add(e.dst)
    reach_cache = {}

    def reaches(a, b):
        if a not in reach_cache:
            seen, stack = set(), [a]
            while stack:
                x = stack.pop()
                for y in explicit.get(x, ()):
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            reach_cache[a] = seen
        return b in reach_cache[a]

    for e in scan.edges:
        if e.kind == "lazy" and e.src in scan.modules and e.dst in scan.modules and reaches(e.dst, e.src):
            pkg = home(e.src, moves)
            rep.lazy_closers[pkg] = rep.lazy_closers.get(pkg, 0) + 1
            rep.lazy_closer_edges.append((e.src, e.dst, e.line))
    return rep


def _sccs(graph: dict) -> list:
    """Tarjan's strongly connected components (iterative)."""
    index, low, on_stack, stack, out = {}, {}, set(), [], []
    counter = [0]
    for start in graph:
        if start in index:
            continue
        work = [(start, iter(graph[start]))]
        index[start] = low[start] = counter[0]; counter[0] += 1
        stack.append(start); on_stack.add(start)
        while work:
            node, it = work[-1]
            advanced = False
            for nxt in it:
                if nxt not in graph:
                    continue
                if nxt not in index:
                    index[nxt] = low[nxt] = counter[0]; counter[0] += 1
                    stack.append(nxt); on_stack.add(nxt)
                    work.append((nxt, iter(graph[nxt])))
                    advanced = True
                    break
                if nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                comp = []
                while True:
                    m = stack.pop(); on_stack.discard(m); comp.append(m)
                    if m == node:
                        break
                out.append(comp)
    return out


def eager_closure(scan: Scan, start: str) -> set:
    """Modules reached from ``start`` over eager and conditional edges, with
    the package-init edges every import triggers."""
    runtime = {m: set() for m in scan.modules}
    for e in scan.edges:
        if e.src in scan.modules and e.kind in ("eager", "conditional"):
            if e.dst in scan.modules:
                runtime[e.src].add(e.dst)
            parts = e.dst.split(".")
            for i in range(1, len(parts)):
                parent = ".".join(parts[:i])
                if parent in scan.packages:
                    runtime[e.src].add(parent)
    seen, stack = {start}, [start]
    while stack:
        x = stack.pop()
        for y in runtime.get(x, ()):
            if y not in seen:
                seen.add(y); stack.append(y)
    return seen


def _match(entry, rule, src, dst, kind):
    e_rule, e_src, e_dst, e_kind = entry[:4]
    return e_rule == rule and e_src == src and e_dst == dst and e_kind in ("*", kind)


def _fmt(rows):
    return "\n".join("  " + " ".join(str(x) for x in r) for r in rows)


# --------------------------------------------------------------------------- #
# The real tree
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real():
    scan = scan_tree(REPO / "n2v", "n2v", [REPO / d for d in CONSUMER_DIRS])
    return scan, analyse(scan, MOVES, RANK)


def test_scanner_finds_the_sentinel_edges(real):
    scan, _ = real
    assert len(scan.modules) >= 80, "far fewer modules than today's tree: is the scan root right?"
    found = {(e.src, e.dst, e.kind) for e in scan.edges}
    missing = [s for s in SENTINEL_EDGES if s not in found]
    assert not missing, f"sentinel edges not found, the scanner is not reading the tree: {missing}"


def test_every_module_has_a_target_package(real):
    scan, rep = real
    assert not rep.homeless, f"modules with no entry in MOVES: {rep.homeless}"
    stale = [k for k in MOVES if k != "n2v" and k not in scan.modules and k not in scan.packages]
    assert not stale, f"MOVES names modules that no longer exist: {stale}"


def test_layer_rule(real):
    _, rep = real
    unlisted = [(s, d, k, l, f"{sp}({RANK[sp]}) -> {dp}({RANK[dp]})") for s, d, k, l, sp, dp in rep.layer
                if not any(_match(a, "layer", s, d, k) for a in ALLOWLIST)]
    assert not unlisted, (
        "imports from a higher or sibling layer that are not on the allowlist "
        "(src dst kind line layers):\n" + _fmt(unlisted))


def test_private_names_stay_inside_their_package(real):
    _, rep = real
    unlisted = [(s, d, k, l, n) for s, d, k, l, n in rep.private
                if not any(_match(a, "private", s, d, k) for a in ALLOWLIST)]
    assert not unlisted, (
        "underscore names imported across a package boundary and not on the "
        "allowlist (src dst kind line names):\n" + _fmt(unlisted))


def test_no_dynamic_import_without_a_declared_forward(real):
    _, rep = real
    assert not rep.dynamic_nonliteral, (
        "importlib/__import__ with a computed name outside a module that declares "
        f"_FORWARDS: {rep.dynamic_nonliteral}")


def test_runtime_import_cycles_are_the_known_ones(real):
    _, rep = real
    unexplained = [sorted(c) for c in rep.cycles
                   if not any(req <= c for req, _, _ in ALLOWED_CYCLES)]
    assert not unexplained, (
        "cross-package import cycles at runtime (eager imports plus the package "
        "__init__ files they trigger) that are not on ALLOWED_CYCLES:\n" + _fmt(unexplained))
    stale = [req for req, _, _ in ALLOWED_CYCLES if not any(req <= c for c in rep.cycles)]
    assert not stale, f"ALLOWED_CYCLES entries that match no cycle any more (remove them): {stale}"


def test_lazy_cycle_closers_are_frozen(real):
    _, rep = real
    assert rep.lazy_closers == LAZY_CYCLE_CLOSERS, (
        "lazy imports that close an import cycle changed; a new one is a design "
        "decision to record in LAZY_CYCLE_CLOSERS, a removed one is progress to record too.\n"
        f"  found: {dict(sorted(rep.lazy_closers.items()))}\n  edges:\n" + _fmt(sorted(rep.lazy_closer_edges)))


def test_allowlist_is_not_stale(real):
    scan, rep = real
    seen = {("layer", s, d, k) for s, d, k, *_ in rep.layer} | {("private", s, d, k) for s, d, k, *_ in rep.private}
    stale = [a[:4] for a in ALLOWLIST
             if not any(a[0] == r and a[1] == s and a[2] == d and a[3] in ("*", k) for r, s, d, k in seen)]
    assert not stale, f"ALLOWLIST entries that match nothing any more (remove them): {stale}"


def test_heavy_third_party_imports_report(real):
    """Report only until PLAN step 24: which heavy third-party modules does
    ``import n2v`` pull in eagerly, and through which module."""
    scan, _ = real
    closure = eager_closure(scan, "n2v")
    carriers = {}
    for mod in sorted(closure):
        for name in scan.third_party.get(mod, ()) & HEAVY_THIRD_PARTY:
            carriers.setdefault(name, []).append(mod)
    print("\nheavy third-party modules in the eager closure of `import n2v` (report only):")
    for name in sorted(HEAVY_THIRD_PARTY):
        print(f"  {name}: {carriers.get(name, 'not imported eagerly')}")


# --------------------------------------------------------------------------- #
# Self-check: a planted package with one case per import form
# --------------------------------------------------------------------------- #
_FAKE_MOVES = {"fake": "top", "fake.low": "sets", "fake.mid": "specs", "fake.high": "refine"}


def _plant(tmp_path: Path) -> Path:
    root = tmp_path / "fake"
    files = {
        "__init__.py": "",
        "low/__init__.py": "",
        "low/a.py": (
            "from typing import TYPE_CHECKING\n"
            "import importlib\n"
            "from fake.high.b import _secret\n"          # eager, upward, private
            "from ..mid import m\n"                      # level-2 relative, upward (sets -> specs)
            "importlib.import_module('fake.high.b')\n"   # dynamic literal, upward
            "if TYPE_CHECKING:\n"
            "    from fake.high import b\n"              # type-only, upward, resolves to fake.high.b
            "def f():\n"
            "    from fake.high.b import thing\n"        # lazy, upward
            "    return thing\n"),
        "low/d.py": "import importlib\ndef g(name):\n    return importlib.import_module(name)\n",
        "mid/__init__.py": "",
        "mid/m.py": "from fake.low import a\n",          # downward through the package: legal
        "high/__init__.py": "from fake.high.c import C\n",
        "high/b.py": "thing = 1\n_secret = 2\n",
        "high/c.py": "from fake.low.a import f\nC = f\n",  # closes a cycle only through high/__init__
    }
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def test_self_check_on_a_planted_package(tmp_path):
    root = _plant(tmp_path)
    scan = scan_tree(root, "fake")
    rep = analyse(scan, _FAKE_MOVES, RANK)

    kinds = {(s, d, k) for s, d, k, *_ in rep.layer}
    assert ("fake.low.a", "fake.high.b", "eager") in kinds, "eager upward import missed"
    assert ("fake.low.a", "fake.high.b", "lazy") in kinds, "lazy upward import missed"
    assert ("fake.low.a", "fake.high.b", "type") in kinds, "type-only upward import missed (and `from pkg import submodule` resolution)"
    assert ("fake.low.a", "fake.high.b", "dynamic") in kinds, "importlib.import_module with a literal missed"
    assert ("fake.low.a", "fake.mid.m", "eager") in kinds, "level-2 relative import not resolved"
    assert not any(s == "fake.mid.m" for s, *_ in rep.layer), "a downward import was reported as a violation"

    assert any(s == "fake.low.a" and d == "fake.high.b" and "_secret" in n for s, d, k, l, n in rep.private), \
        "private cross-package name missed"
    assert rep.dynamic_nonliteral == [("fake.low.d", 3)], "computed import name not reported"

    assert any({"fake.low.a", "fake.high", "fake.high.c"} <= c for c in rep.cycles), \
        "the cycle that exists only through high/__init__ was not found"

    # the ratchet: an allowlist entry that matches nothing is stale
    seen = {("layer", s, d, k) for s, d, k, *_ in rep.layer}
    bogus = ("layer", "fake.low.a", "fake.high.zzz", "eager")
    assert not any(bogus[1:] == (s, d, k) for _, s, d, k in seen), "self-check setup"
