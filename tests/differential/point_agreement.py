"""Point-agreement machinery: n2v's parsed representation vs naive_eval.

For a spec file and a concrete point (x, y):
  - naive side: values keyed by the variable spellings in the file,
    evaluated by tests/differential/naive_eval.py directly on the text;
  - our side: the parsed pairs structure, evaluated by ~15 lines below.

The atom->flat-index mapping here is written independently of the parser
(simple regexes over the declarations + row-major ravel), so an indexing
bug in the parser cannot hide. Point GENERATION may consult the parsed
structure for coverage (hitting regions, boundaries) — generation bias
cannot fake agreement, since both verdicts are computed independently.
"""

import re

import numpy as np

from tests.differential import naive_eval


# ---------------------------------------------------------------------------
# independent atom -> (space, flat index) mapping
# ---------------------------------------------------------------------------

def _shapes_from_text(text):
    """Declared tensors in declaration order: list of (name, kind, shape)."""
    out = []
    for kind, name, shape_s in re.findall(
            r"\(declare-(input|output)\s+(\S+)\s+\S+\s*\[([^\]]*)\]", text):
        shape = tuple(int(s) for s in shape_s.replace(",", " ").split())
        out.append((name, "in" if kind == "input" else "out", shape))
    return out


def atom_index_map(text, atoms=None):
    """Map every variable atom spelling -> ('in'|'out', joint flat index).

    Handles 1.0 (``X_3`` -> index 3) and 2.0 (declared tensor names with
    row-major tensor indexing, concatenated per kind in declaration order).
    """
    mapping = {}
    if atoms is None:
        atoms = naive_eval.variable_atoms(text)
    decls = _shapes_from_text(text)
    if not decls:  # legacy 1.0: indices live in the names
        for atom in atoms:
            kind = "in" if atom.startswith("X") else "out"
            mapping[atom] = (kind, int(atom.split("_")[1]))
        return mapping

    offsets = {}
    sizes = {"in": 0, "out": 0}
    for name, kind, shape in decls:
        size = int(np.prod(shape)) if shape else 1
        offsets[name] = (kind, sizes[kind], shape)
        sizes[kind] += size

    for atom in atoms:
        m = re.match(r"^(\S+?)\[([0-9,\s]*)\]$", atom)
        if m:
            name, idx_s = m.group(1), m.group(2)
            kind, off, shape = offsets[name]
            idx = tuple(int(s) for s in idx_s.replace(",", " ").split())
            flat = 0
            for d, i in zip(shape, idx):
                flat = flat * d + i
            mapping[atom] = (kind, off + flat)
        else:  # bare name = rank-0/scalar tensor
            kind, off, _ = offsets[atom]
            mapping[atom] = (kind, off)
    return mapping


# ---------------------------------------------------------------------------
# our-side verdict from the parsed pairs
# ---------------------------------------------------------------------------

def pairs_satisfied(pairs, x, y, tol=0.0):
    """Does (x, y) witness a violation per the parsed representation?"""
    for pair in pairs:
        lb = np.asarray(pair["lb"], dtype=np.float64).flatten()
        ub = np.asarray(pair["ub"], dtype=np.float64).flatten()
        if not (np.all(x >= lb - tol) and np.all(x <= ub + tol)):
            continue
        ok = True
        for group in pair["prop"]:
            hg = group["Hg"]
            hs_list = hg if isinstance(hg, list) else [hg]
            if not any(np.all(np.asarray(h.G) @ y
                              <= np.asarray(h.g).flatten() + tol)
                       for h in hs_list):
                ok = False
                break
        if ok:
            return True
    return False


def relational_satisfied(result, x, y, tol=0.0):
    """Verdict for a relational (multi-network) parse: joint box AND
    coupling AND every output group (OR within a group)."""
    lb = np.asarray(result["lb"], dtype=np.float64).flatten()
    ub = np.asarray(result["ub"], dtype=np.float64).flatten()
    if not (np.all(x >= lb - tol) and np.all(x <= ub + tol)):
        return False
    C = result["input_coupling"]
    if C is not None and not np.all(
            np.asarray(C.G) @ x <= np.asarray(C.g).flatten() + tol):
        return False
    for group in result["prop"]:
        hg = group["Hg"]
        hs_list = hg if isinstance(hg, list) else [hg]
        if not any(np.all(np.asarray(h.G) @ y
                          <= np.asarray(h.g).flatten() + tol)
                   for h in hs_list):
            return False
    return True


def check_relational_agreement(text, result, n_points=120, seed=0):
    """Two-sided check for relational specs. Sampling enforces the
    equality couplings on half the points (coverage of the satisfied
    side); verdicts on every point remain independently computed."""
    spec = naive_eval.NaiveSpec(text)
    amap = atom_index_map(text, atoms=spec.variable_atoms())
    in_dim = len(np.asarray(result["lb"]).flatten())
    out_dim = max((i for k, i in amap.values() if k == "out"),
                  default=-1) + 1
    for group in result["prop"]:
        hg = group["Hg"]
        for h in (hg if isinstance(hg, list) else [hg]):
            out_dim = max(out_dim, int(np.asarray(h.G).shape[1]))
    lb = np.asarray(result["lb"], dtype=np.float64).flatten()
    ub = np.asarray(result["ub"], dtype=np.float64).flatten()
    base_lb = np.where(np.isfinite(lb), lb, -1.0)
    base_ub = np.where(np.isfinite(ub), ub, 1.0)
    half = len(lb) // 2
    rng = np.random.default_rng(seed)
    disagreements = []
    for _ in range(n_points):
        x = base_lb + (base_ub - base_lb) * rng.random(in_dim)
        if rng.random() < 0.5 and 2 * half == len(lb):
            x[half:] = x[:half]      # enforce the f==g couplings
        if rng.random() < 0.3:
            d = int(rng.integers(in_dim))
            x[d] += rng.normal() * 0.5   # perturb: exercise violations
        y = rng.normal(size=out_dim) * 2.0
        if rng.random() < 0.5 and 2 * (out_dim // 2) == out_dim:
            y[out_dim // 2:] = y[:out_dim // 2] + rng.normal(
                size=out_dim // 2) * 0.1  # near-equal outputs
        values = {atom: (x[i] if kind == "in" else y[i])
                  for atom, (kind, i) in amap.items()}
        naive = spec.satisfied(values)
        ours = relational_satisfied(result, x, y)
        if naive != ours:
            disagreements.append((x.tolist(), y.tolist(), naive, ours))
    return n_points, disagreements


# ---------------------------------------------------------------------------
# point generation (coverage-oriented; verdicts stay independent)
# ---------------------------------------------------------------------------

def sample_points(pairs, in_dim, out_dim, n, rng):
    """Mixed sample: inside each region / perturbed outside / fully random."""
    points = []
    boxes = [(np.asarray(p["lb"], dtype=np.float64).flatten(),
              np.asarray(p["ub"], dtype=np.float64).flatten())
             for p in pairs]
    while len(points) < n:
        kind = rng.integers(3)
        lb, ub = boxes[int(rng.integers(len(boxes)))]
        if kind == 0:    # inside the region (equality dims land exactly)
            x = lb + (ub - lb) * rng.random(in_dim)
        elif kind == 1:  # perturbed outside
            x = lb + (ub - lb) * rng.random(in_dim)
            d = int(rng.integers(in_dim))
            x[d] = ub[d] + abs(rng.normal()) + 1e-3 if rng.random() < 0.5 \
                else lb[d] - abs(rng.normal()) - 1e-3
        else:            # fully random near the box scale
            span = np.maximum(np.abs(lb), np.abs(ub)) + 1.0
            x = rng.normal(size=in_dim) * span
        y = rng.normal(size=out_dim) * 10.0
        points.append((x, y))
    return points


def check_file_agreement(text, pairs, n_points=120, seed=0):
    """Run the two-sided check. Returns (n_checked, disagreements)."""
    spec = naive_eval.NaiveSpec(text)  # single parse of the text
    amap = atom_index_map(text, atoms=spec.variable_atoms())
    in_dim = max((i for k, i in amap.values() if k == "in"), default=-1) + 1
    out_dim = max((i for k, i in amap.values() if k == "out"), default=-1) + 1
    in_dim = max(in_dim, max(len(np.asarray(p["lb"]).flatten())
                             for p in pairs))
    # asserts may reference only a subset of outputs; the halfspace width
    # is authoritative for the y vector length
    for p in pairs:
        for group in p["prop"]:
            hg = group["Hg"]
            for h in (hg if isinstance(hg, list) else [hg]):
                out_dim = max(out_dim, int(np.asarray(h.G).shape[1]))
    rng = np.random.default_rng(seed)
    disagreements = []
    for x, y in sample_points(pairs, in_dim, out_dim, n_points, rng):
        values = {atom: (x[i] if kind == "in" else y[i])
                  for atom, (kind, i) in amap.items()}
        naive = spec.satisfied(values)
        ours = pairs_satisfied(pairs, x, y)
        if naive != ours:
            disagreements.append((x.tolist(), y.tolist(), naive, ours))
    return n_points, disagreements
