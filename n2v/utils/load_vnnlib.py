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
    Load and parse a VNN-LIB property file.

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
                lb_template = np.zeros(dim, dtype=np.float32)
                ub_template = np.zeros(dim, dtype=np.float32)
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
                    elif "quadrotor2d_state" in property_file:
                        # Quick fix for lsnc_relu
                        last_ast['Hg'] = HalfSpace(
                            np.vstack([last_ast['Hg'].G, ast['Hg'].G]),
                            np.vstack([last_ast['Hg'].g, ast['Hg'].g])
                        )
                        property_dict['prop'][-1] = last_ast
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
    """Combine multiple lines into a single one if they belong to a single statement."""
    tline = lines[start_idx].strip()
    lines_consumed = 1

    while tline.count('(') != tline.count(')') and (start_idx + lines_consumed) < len(lines):
        next_line = lines[start_idx + lines_consumed].strip()
        tline = tline + " " + next_line
        lines_consumed += 1

    return tline, lines_consumed


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
                ast['Hg'] = HalfSpace(ast['H'].reshape(1, -1), np.array([[ast['g']]], dtype=np.float32))
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
    H = np.zeros(ast['dim'], dtype=np.float32)
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
                    np.array([[temp_ast['g']]], dtype=np.float32)
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
            raise ValueError("We may be doing something wrong while processing the AND statement or the property is currently not supported.")

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
                np.array([[or_ast['g']]], dtype=np.float32)
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


def _process_combined_input_output(tline: str, lb_input: np.ndarray, ub_input: np.ndarray,
                                   output_dim: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
    """Process input assertion with combined input and output (or statement)."""
    # Extract all (and ...) blocks
    and_blocks = []
    depth = 0
    current_block = ""
    in_and = False

    for i, char in enumerate(tline):
        if char == '(' and i + 3 < len(tline) and tline[i:i+4] == '(and':
            in_and = True
            depth = 0
            current_block = ""

        if in_and:
            current_block += char
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    and_blocks.append(current_block)
                    in_and = False

    lb_array = []
    ub_array = []
    prop_array = []

    for block in and_blocks:
        lb_temp = lb_input.copy()
        ub_temp = ub_input.copy()
        H_list = []
        g_list = []

        # Extract constraints from this block
        constraints = []
        depth = 0
        current_constraint = ""

        for char in block:
            if char == '(':
                depth += 1
                current_constraint += char
            elif char == ')':
                depth -= 1
                current_constraint += char
                if depth == 1:  # Complete constraint at depth 1
                    if current_constraint.strip() and current_constraint.count('(') > 1:
                        constraints.append(current_constraint.strip())
                    current_constraint = ""
            else:
                current_constraint += char

        # Process each constraint
        for constraint in constraints:
            if 'X' in constraint:
                lb_temp, ub_temp = _process_input_constraint(constraint, lb_temp, ub_temp)
            else:
                H, g = _process_output_combo_constraint(constraint, H_list, g_list, output_dim)
                H_list = H
                g_list = g

        # Create HalfSpace
        if H_list:
            Hg = HalfSpace(np.array(H_list, dtype=np.float32), np.array(g_list, dtype=np.float32).reshape(-1, 1))
        else:
            Hg = None

        prop = {
            'Hg': Hg,
            'H': np.array(H_list, dtype=np.float32) if H_list else None,
            'g': np.array(g_list, dtype=np.float32) if g_list else None
        }

        lb_array.append(lb_temp)
        ub_array.append(ub_temp)
        prop_array.append(prop)

    return lb_array, ub_array, prop_array


def _process_output_combo_constraint(tline: str, H: List, g: List,
                                     output_dim: int) -> Tuple[List, List]:
    """Process output constraint in combined input/output format."""
    parts = tline.split('(')
    constraint_part = parts[-1] if parts else tline

    tokens = constraint_part.split()

    Hvec = np.zeros(output_dim, dtype=np.float32)

    # Extract variable index
    var_idx = int(tokens[1].split('_')[1])

    # Extract value
    value_str = tokens[2].rstrip(')')
    value = float(value_str)

    # Determine constraint type
    if '>=' in tokens[0] or '>' in tokens[0]:
        Hvec[var_idx] = -1
        gval = -value
    else:  # '<=' or '<'
        Hvec[var_idx] = 1
        gval = value

    H.append(Hvec)
    g.append(gval)

    return H, g


def _process_multiple_inputs(tline: str, lb_input: np.ndarray, ub_input: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Process input assertion with multiple input sets (or statement)."""
    tline = tline[11:].strip()  # Remove '(assert (or'

    pars = 1
    lb_array = []
    ub_array = []
    arr_count = -1

    lb_temp = lb_input.copy()
    ub_temp = ub_input.copy()

    while tline and pars > 0:
        tline = tline.strip()

        if tline.startswith('(<=') or tline.startswith('(>='):
            # Find end of constraint
            end_idx = tline.find(')')
            constraint = tline[:end_idx+1]
            lb_temp, ub_temp = _process_input_constraint(constraint, lb_temp, ub_temp)
            tline = tline[end_idx+1:].strip()

        elif tline.startswith('(or'):
            raise ValueError("Currently we do not support an OR statement within an OR statement.")

        elif tline.startswith('(and'):
            arr_count += 1
            lb_temp = lb_input.copy()
            ub_temp = ub_input.copy()
            tline = tline[4:].strip()
            pars += 1

        elif tline.startswith(')'):
            pars -= 1
            tline = tline[1:].strip()
            if arr_count >= 0 and pars == 1:
                lb_array.append(lb_temp)
                ub_array.append(ub_temp)
        else:
            raise ValueError("Something went wrong while processing vnnlib property with multiple inputs.")

    return lb_array, ub_array
