"""
Load and parse VNN-LIB property files.

This module provides functionality to parse VNN-LIB format property files
and convert them into a structured format for verification.

Supported formats:
    Input specifications:
        a) Multiple assert statements with individual bounds (e.g., prop_1-4)
        b) Single assert with (or (and ...) (and ...)) for multiple input regions (e.g., prop_6)
        c) Combined input/output in same assert (both X_ and Y_ in one or statement)

    Output specifications:
        a) Series of assertions with no "and", "or" conditions  ---->  property['prop']: (1x1) HalfSpace
        b) One assertion with "and" statements                  ---->  property['prop']: (1x1) HalfSpace
        c) One assertion with "or" of "and" statements          ---->  property['prop']: (Nx1) list of HalfSpace

    Constraints must be linear, only >= or <= are supported.

    If the vnnlib does not meet these conditions, this function may not work.
"""

import numpy as np
from typing import Dict, List, Tuple
from n2v.sets.halfspace import HalfSpace


def load_vnnlib(property_file: str) -> Dict:
    """
    Load and parse a VNN-LIB property file (auto-detects 1.0 vs 2.0).

    VNN-COMP 2026 migrated to the VNNLIB 2.0 format (``(declare-network ...)``
    with tensor-indexed variables ``X[0,0,0,0]``). This dispatcher detects
    the format from the file contents and routes accordingly:

      - **1.0** (legacy ``(declare-const X_0 Real)`` / ``X_0`` syntax) ->
        :func:`_load_vnnlib_v1` (unchanged).
      - **2.0** (``vnnlib-version`` / ``declare-network``) ->
        :func:`n2v.utils.vnnlib2.load_vnnlib_v2`.

    Both produce the same structure. As a guard against the historical
    failure mode where an unrecognized spec parsed to an *empty* property
    (silently "verifying nothing"), this function raises if the result has
    no input bounds or no output property.

    Args:
        property_file: Path to the VNN-LIB file describing property to verify

    Returns:
        property: Dictionary with following fields:
            - 'lb': input lower bound vector (numpy array or list of arrays)
            - 'ub': input upper bound vector (numpy array or list of arrays)
            - 'prop': collection of HalfSpaces describing output specification to verify
    """
    with open(property_file, 'r') as f:
        text = f.read()

    if 'vnnlib-version' in text or 'declare-network' in text:
        from n2v.utils.vnnlib2 import load_vnnlib_v2
        result = load_vnnlib_v2(property_file, text)
    else:
        result = _load_vnnlib_v1(property_file)

    if result.get('format') == 'nonlinear':
        # Spec outside the linear fragment: the resolved assertion ASTs
        # are the authoritative representation ('prop' is None; the box
        # is best-effort). Validated in the nonlinear loader.
        return result
    _assert_nonempty(result, property_file)
    if result.get('format') == 'relational':
        # Multi-network spec: joint box + coupling + joint output prop.
        # The pairs contract (independent input regions) does not apply;
        # bound validation happened in the relational lowering (a dim may
        # be unbounded in the box if constrained via coupling).
        return result
    _build_pairs(result)
    _validate_pairs(result, property_file)
    return result


def _build_pairs(result: Dict) -> None:
    """Normalize every parse to ``result['pairs']``: a list of
    ``{'lb', 'ub', 'prop'}`` dicts, one per input region.

    Semantics: a counterexample exists iff for SOME pair, an input in
    [lb, ub] produces an output satisfying that pair's prop. For specs
    where regions and output constraints are independent, every pair
    shares the same prop; for combined input/output specs (``paired``),
    region i carries its own prop entry i.
    """
    if result.get('pairs'):
        return  # parser produced pairs directly
    lb, ub, prop = result['lb'], result['ub'], result['prop']
    if isinstance(lb, list):
        if result.get('paired'):
            pairs = [{'lb': l, 'ub': u, 'prop': [p]}
                     for l, u, p in zip(lb, ub, prop)]
        else:
            pairs = [{'lb': l, 'ub': u, 'prop': prop} for l, u in zip(lb, ub)]
    else:
        pairs = [{'lb': lb, 'ub': ub, 'prop': prop}]
    result['pairs'] = pairs


def _validate_pairs(result: Dict, property_file: str) -> None:
    """Reject impossible or unbounded regions loudly.

    The historical failure mode was a mis-parse producing lb > ub (or
    fabricated bounds) that either crashed far downstream or silently
    altered the verified property.
    """
    for i, pair in enumerate(result['pairs']):
        lb = np.asarray(pair['lb'], dtype=np.float64).flatten()
        ub = np.asarray(pair['ub'], dtype=np.float64).flatten()
        if not (np.isfinite(lb).all() and np.isfinite(ub).all()):
            raise ValueError(
                f"Parsed VNN-LIB spec leaves input dimensions unbounded "
                f"in region {i}: {property_file}"
            )
        if (lb > ub).any():
            bad = int(np.argmax(lb > ub))
            raise ValueError(
                f"Parsed VNN-LIB spec has lb > ub in region {i} "
                f"(dim {bad}: {lb[bad]} > {ub[bad]}): {property_file}"
            )
        if not pair['prop']:
            raise ValueError(
                f"Parsed VNN-LIB spec has no output property for region "
                f"{i}: {property_file}"
            )


def _assert_nonempty(result: Dict, property_file: str) -> None:
    """Raise if a parsed spec has no input bounds or no output property.

    Guards against silently 'verifying nothing' when a spec format is not
    recognized by the parser.
    """
    lb = result.get('lb')
    prop = result.get('prop')
    empty_lb = lb is None or (isinstance(lb, list) and len(lb) == 0)
    empty_prop = prop is None or (isinstance(prop, list) and len(prop) == 0)
    if empty_lb or empty_prop:
        raise ValueError(
            f"Parsed VNN-LIB spec is empty (lb or prop missing): "
            f"{property_file}. The spec format may be unsupported."
        )


def _load_vnnlib_v1(property_file: str) -> Dict:
    """
    Load and parse a legacy (1.0) VNN-LIB property file.

    Args:
        property_file: Path to the VNN-LIB file describing property to verify

    Returns:
        property: Dictionary with following fields:
            - 'lb': input lower bound vector (numpy array or list of arrays)
            - 'ub': input upper bound vector (numpy array or list of arrays)
            - 'prop': collection of HalfSpaces describing output specification to verify
    """
    with open(property_file, 'r') as f:
        lines = f.readlines()

    # State machine variables
    phase = "start"  # Four phases: start, DeclareInput, DeclareOutput, DefineInput, DefineOutput
    dim = 0
    output_dim = 0
    lb_template = None
    ub_template = None
    lb_input = None
    ub_input = None
    property_dict = {
        'lb': None,
        'ub': None,
        'prop': []
    }

    line_idx = 0
    while line_idx < len(lines):
        tline = lines[line_idx].strip()

        # Skip empty lines and comments
        if not tline or tline.startswith(';'):
            line_idx += 1
            continue

        # Merge multi-line statements
        lines_consumed = 1  # Default: single line
        if tline.count('(') != tline.count(')'):
            tline, lines_consumed = _merge_lines(lines, line_idx)

        # Process based on current phase
        if phase == "DeclareInput":
            if "declare-const" in tline and "X_" in tline:
                dim += 1
            elif "declare-const" in tline and "Y_" in tline:
                # Unconstrained until asserted; the pairs validity guard
                # rejects any dimension left unbounded (previously these
                # defaulted to 0, silently fabricating bounds).
                lb_template = np.full(dim, -np.inf, dtype=np.float64)
                ub_template = np.full(dim, np.inf, dtype=np.float64)
                dim = 0
                phase = "DeclareOutput"
                continue  # Redo this line in correct phase

        elif phase == "DeclareOutput":
            if "declare-const" in tline and "Y_" in tline:
                dim += 1
            elif "assert" in tline:
                output_dim = dim
                dim = 1
                phase = "DefineInput"
                lb_input = lb_template.copy()
                ub_input = ub_template.copy()
                continue  # Redo this line in correct phase

        elif phase == "DefineInput":
            # Four options:
            # 1) One input set -> multiple assert statements (2 per dimension)
            # 2) Multiple input sets -> or statement with only X_ constraints,
            #    followed by separate output assertion(s)
            # 3) Multiple input and output sets -> or statement with both X_ and Y_
            # 4) Transition to output phase when assertion has no X_

            if "assert" in tline and "or" in tline and "X_" in tline:
                if "Y_" in tline:  # Option 3: combined input/output in same assertion
                    lb_array, ub_array, prop_array = _process_combined_input_output(
                        tline, lb_input, ub_input, output_dim
                    )
                    property_dict['lb'] = lb_array
                    property_dict['ub'] = ub_array
                    property_dict['prop'] = prop_array
                    # Region i is paired with prop entry i (per-disjunct).
                    property_dict['paired'] = True
                    # Done processing - this format has everything in one assertion
                else:  # Option 2: multiple input regions, separate output assertion
                    lb_array, ub_array = _process_multiple_inputs(tline, lb_input, ub_input)
                    property_dict['lb'] = lb_array
                    property_dict['ub'] = ub_array
                    # Transition to DefineOutput phase for the next assertion
                    phase = "DefineOutput"
                    property_dict['prop'] = []

            elif ">" in tline or "<" in tline:
                if "X_" in tline:  # Option 1: single input set with separate assertions
                    lb_input, ub_input = _process_input_constraint(tline, lb_input, ub_input)
                else:  # Option 4: Move to output phase (assertion without X_)
                    phase = "DefineOutput"
                    property_dict['lb'] = lb_input
                    property_dict['ub'] = ub_input
                    property_dict['prop'] = []
                    continue  # Redo this line in correct phase

        elif phase == "DefineOutput":
            if "assert" in tline:
                if ">=" in tline or "<=" in tline or ">" in tline or "<" in tline:
                    ast = _process_assertion(tline, output_dim)
                else:
                    raise ValueError(f"Property not supported yet for assertion: {tline}")

                # Add assertion to property
                if not property_dict['prop']:
                    property_dict['prop'] = [ast]
                else:
                    last_ast = property_dict['prop'][-1]
                    if isinstance(last_ast.get('Hg'), list) and len(last_ast['Hg']) > 1:
                        # Previous ast was an "or"
                        property_dict['prop'].append(ast)
                    else:
                        # Concatenate with previous assertion (AND)
                        last_ast['Hg'] = HalfSpace(
                            np.vstack([last_ast['Hg'].G, ast['Hg'].G]),
                            np.vstack([last_ast['Hg'].g, ast['Hg'].g])
                        )
                        property_dict['prop'][-1] = last_ast
        else:
            # Initializing (no phase)
            if "declare-const" in tline and "X_" in tline:
                phase = "DeclareInput"
                dim = 0
                continue  # Redo this line in correct phase

        line_idx += lines_consumed

    return property_dict


# ============================================================================
# Helper Functions - General
# ============================================================================

def _merge_lines(lines: List[str], start_idx: int) -> Tuple[str, int]:
    """Combine multiple lines into a single one if they belong to a single statement.

    Linear time: accumulates parts and tracks the paren balance
    incrementally (the previous version re-concatenated and re-counted the
    whole string per line — quadratic on multi-MB single-assert specs).
    """
    first = lines[start_idx].strip()
    parts = [first]
    balance = first.count('(') - first.count(')')
    lines_consumed = 1

    while balance != 0 and (start_idx + lines_consumed) < len(lines):
        next_line = lines[start_idx + lines_consumed].strip()
        parts.append(next_line)
        balance += next_line.count('(') - next_line.count(')')
        lines_consumed += 1

    return " ".join(parts), lines_consumed


# ============================================================================
# Helper Functions - OUTPUT
# ============================================================================

def _process_assertion(tline: str, dim: int) -> Dict:
    """Process output assertion."""
    ast = {
        'dim': dim,
        'Hg': None,
        'H': None,
        'g': None
    }

    tline = tline.strip()
    # Remove "(assert" from tline
    tline = tline[7:].strip()

    while tline:
        if len(tline) == 1 and tline == ')':
            break

        tline = tline.strip()

        # Process linear constraint
        if tline.startswith('(<=') or tline.startswith('(>=') or tline.startswith('(<') or tline.startswith('(>'):
            ast, length = _process_constraint(tline, ast)
            tline = tline[length+1:]

            if ast['Hg'] is None:
                ast['Hg'] = HalfSpace(ast['H'].reshape(1, -1), np.array([[ast['g']]], dtype=np.float64))
            else:
                # Concatenate (similar to AND statement)
                ast['Hg'] = HalfSpace(
                    np.vstack([ast['Hg'].G, ast['H'].reshape(1, -1)]),
                    np.vstack([ast['Hg'].g, [[ast['g']]]])
                )

        elif tline.startswith('(or'):
            tline = tline[3:].strip()
            ast, tline = _process_or(tline, ast)

        elif tline.startswith('(and'):
            raise ValueError('Property not supported for now. Not allowed -> assertion starting with AND and no OR.')

        elif tline.startswith(')'):
            tline = tline[1:].strip()
        else:
            raise ValueError(f"Not sure what is happening, but you should not be here: {tline}")

    return ast


def _process_constraint(tline: str, ast: Dict) -> Tuple[Dict, int]:
    """Process a single output constraint."""
    # Find the closing parenthesis
    length = 0
    while length < len(tline) and tline[length] != ')':
        length += 1

    const = tline[:length+1]
    parts = const.split()

    op = parts[0][1:]  # Remove leading '('
    var1 = parts[1]
    var2 = parts[2].rstrip(')')

    # Extract index from variable
    idx1 = int(var1.split('_')[1])

    # Initialize constraint
    H = np.zeros(ast['dim'], dtype=np.float64)
    g = 0.0

    if '<=' in op or '<' in op:
        H[idx1] = 1
        if 'Y' in var2:
            idx2 = int(var2.split('_')[1])
            H[idx2] = -1
        else:
            g = float(var2)
    else:  # >= or >
        H[idx1] = -1
        if 'Y' in var2:
            idx2 = int(var2.split('_')[1])
            H[idx2] = 1
        else:
            g = -float(var2)

    ast['H'] = H
    ast['g'] = g

    return ast, length


def _process_and(tline: str, ast: Dict) -> Tuple[Dict, str]:
    """Process output 'and' statement."""
    temp_ast = {
        'dim': ast['dim'],
        'Hg': None
    }

    params = 1  # From the removed '(and'

    while tline and params > 0:
        tline = tline.strip()

        if tline.startswith('(<=') or tline.startswith('(>='):
            temp_ast, length = _process_constraint(tline, temp_ast)
            tline = tline[length+1:]

            if temp_ast['Hg'] is None:
                temp_ast['Hg'] = HalfSpace(
                    temp_ast['H'].reshape(1, -1),
                    np.array([[temp_ast['g']]], dtype=np.float64)
                )
            else:
                temp_ast['Hg'] = HalfSpace(
                    np.vstack([temp_ast['Hg'].G, temp_ast['H'].reshape(1, -1)]),
                    np.vstack([temp_ast['Hg'].g, [[temp_ast['g']]]])
                )

        elif tline.startswith('(or'):
            raise ValueError("Currently we do not support an OR statement within an AND statement.")

        elif tline.startswith('(and'):
            raise ValueError("Currently we do not support an AND statement within an AND statement.")

        elif tline.startswith(')'):
            params -= 1
            tline = tline[1:].strip()
        else:
            raise ValueError(
                "We may be doing something wrong while processing the AND statement "
                "or the property is currently not supported."
            )

    return temp_ast, tline


def _process_or(tline: str, ast: Dict) -> Tuple[Dict, str]:
    """Process output 'or' statement."""
    temp_ast = {
        'dim': ast['dim'],
        'Hg': []
    }

    pars = 1

    while tline and pars > 0:
        tline = tline.strip()

        if tline.startswith('(<=') or tline.startswith('(>='):
            or_ast, length = _process_constraint(tline, temp_ast)
            tline = tline[length+1:]

            halfspace = HalfSpace(
                or_ast['H'].reshape(1, -1),
                np.array([[or_ast['g']]], dtype=np.float64)
            )

            if not temp_ast['Hg']:
                temp_ast['Hg'] = [halfspace]
            else:
                temp_ast['Hg'].append(halfspace)

        elif tline.startswith('(or'):
            raise ValueError("Currently we do not support an OR statement within an OR statement.")

        elif tline.startswith('(and'):
            tline = tline[4:].strip()
            pars += 1
            or_ast, tline = _process_and(tline, temp_ast)

            if not temp_ast['Hg']:
                temp_ast['Hg'] = [or_ast['Hg']]
            else:
                temp_ast['Hg'].append(or_ast['Hg'])

        elif tline.startswith(')'):
            pars -= 1
            tline = tline[1:].strip()
        else:
            raise ValueError("Something went wrong while processing the OR statement.")

    ast = temp_ast
    return ast, tline


# ============================================================================
# Helper Functions - INPUT
# ============================================================================

def _process_input_constraint(tline: str, lb_input: np.ndarray, ub_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Process input constraint."""
    # Extract the constraint part
    parts = tline.split('(')
    constraint_part = parts[-1] if parts else tline

    tokens = constraint_part.split()

    # Extract variable index
    var_idx = int(tokens[1].split('_')[1])

    # Extract value
    value_str = tokens[2].rstrip(')')
    value = float(value_str)

    # Determine if it's a lower or upper bound
    if '>=' in tokens[0]:
        lb_input[var_idx] = value
    else:  # '<='
        ub_input[var_idx] = value

    return lb_input, ub_input


def _read_sexpr(text: str):
    """Parse one balanced s-expression string into nested lists/atoms.

    Linear time — replaces the char-walking/string-slicing extraction that
    was quadratic on multi-MB single-assert specs and silently dropped
    every constraint after the first in each and-block.
    """
    tokens = text.replace('(', ' ( ').replace(')', ' ) ').split()
    pos = 0

    def read():
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("unexpected end of s-expression")
        tok = tokens[pos]
        pos += 1
        if tok == '(':
            lst = []
            while pos < len(tokens) and tokens[pos] != ')':
                lst.append(read())
            pos += 1  # consume ')'
            return lst
        if tok == ')':
            raise ValueError("unexpected ')'")
        return tok

    return read()


def _var_index(atom) -> Tuple[str, int]:
    """``X_3`` -> ('X', 3); ``Y_0`` -> ('Y', 0); else (None, -1)."""
    if isinstance(atom, str) and len(atom) > 2 and atom[1] == '_' \
            and atom[0] in ('X', 'Y'):
        try:
            return atom[0], int(atom[2:])
        except ValueError:
            pass
    return None, -1


def _apply_block_atom(atom, lb: np.ndarray, ub: np.ndarray,
                      H_rows: List, g_vals: List, output_dim: int) -> None:
    """Apply one comparison ``(op var value)`` from an and-block.

    X constraints tighten the region bounds; Y constraints append a
    ``row . y <= g`` halfspace row (Y-vs-const or Y-vs-Y).
    """
    if not (isinstance(atom, list) and len(atom) == 3):
        raise ValueError(f"unsupported constraint in or/and block: {atom!r}")
    op, a, b = atom
    kind_a, idx_a = _var_index(a)
    if kind_a is None:
        raise ValueError(f"expected variable on constraint LHS: {atom!r}")

    if kind_a == 'X':
        value = float(b)
        if op in ('<=', '<'):
            ub[idx_a] = min(ub[idx_a], value)
        elif op in ('>=', '>'):
            lb[idx_a] = max(lb[idx_a], value)
        else:
            raise ValueError(f"unsupported operator on input: {op!r}")
        return

    # Y constraint -> halfspace row
    row = np.zeros(output_dim, dtype=np.float64)
    kind_b, idx_b = _var_index(b)
    if op in ('<=', '<'):
        row[idx_a] = 1.0
        if kind_b == 'Y':
            row[idx_b] -= 1.0
            gval = 0.0
        else:
            gval = float(b)
    elif op in ('>=', '>'):
        row[idx_a] = -1.0
        if kind_b == 'Y':
            row[idx_b] += 1.0
            gval = 0.0
        else:
            gval = -float(b)
    else:
        raise ValueError(f"unsupported operator on output: {op!r}")
    H_rows.append(row)
    g_vals.append(gval)


def _trivially_true_halfspace(output_dim: int) -> HalfSpace:
    """``0 . y <= 0`` — always satisfied. Used for disjuncts with no
    output constraint ('any output violates' for that input region)."""
    return HalfSpace(np.zeros((1, output_dim), dtype=np.float64),
                     np.zeros((1, 1), dtype=np.float64))


def _process_combined_input_output(tline: str, lb_input: np.ndarray, ub_input: np.ndarray,
                                   output_dim: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
    """Process an assertion combining input and output constraints:
    ``(assert (or (and X-bounds... Y-constraints...) ...))``.

    Each disjunct becomes one (region, prop) pair: region i is paired
    with prop entry i.
    """
    tree = _read_sexpr(tline)
    if not (isinstance(tree, list) and tree and tree[0] == 'assert'):
        raise ValueError(f"expected an assert statement, got: {tline[:80]!r}")
    formula = tree[1]
    if isinstance(formula, list) and formula and formula[0] == 'or':
        blocks = formula[1:]
    else:
        blocks = [formula]

    lb_array, ub_array, prop_array = [], [], []
    for block in blocks:
        if isinstance(block, list) and block and block[0] == 'and':
            atoms = block[1:]
        else:
            atoms = [block]

        lb_temp = lb_input.copy()
        ub_temp = ub_input.copy()
        H_rows: List = []
        g_vals: List = []
        for atom in atoms:
            _apply_block_atom(atom, lb_temp, ub_temp, H_rows, g_vals, output_dim)

        if H_rows:
            Hg = HalfSpace(np.array(H_rows, dtype=np.float64),
                           np.array(g_vals, dtype=np.float64).reshape(-1, 1))
        else:
            Hg = _trivially_true_halfspace(output_dim)

        prop_array.append({
            'Hg': Hg,
            'H': Hg.G,
            'g': Hg.g,
        })
        lb_array.append(lb_temp)
        ub_array.append(ub_temp)

    return lb_array, ub_array, prop_array


def _process_multiple_inputs(tline: str, lb_input: np.ndarray, ub_input: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Process an input-only disjunction:
    ``(assert (or (and X-bounds...) ...))`` -> list of regions."""
    tree = _read_sexpr(tline)
    if not (isinstance(tree, list) and tree and tree[0] == 'assert'):
        raise ValueError(f"expected an assert statement, got: {tline[:80]!r}")
    formula = tree[1]
    if isinstance(formula, list) and formula and formula[0] == 'or':
        blocks = formula[1:]
    else:
        blocks = [formula]

    lb_array, ub_array = [], []
    for block in blocks:
        if isinstance(block, list) and block and block[0] == 'and':
            atoms = block[1:]
        else:
            atoms = [block]
        lb_temp = lb_input.copy()
        ub_temp = ub_input.copy()
        for atom in atoms:
            _apply_block_atom(atom, lb_temp, ub_temp, [], [], 0)
        lb_array.append(lb_temp)
        ub_array.append(ub_temp)

    return lb_array, ub_array
