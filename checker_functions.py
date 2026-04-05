import re
import os
from hyperon import *
from sexpdata import loads, Symbol

from built_in import *

# TODO: handle keyword collision, e.g. `expr_format_check("(: Empty (-> Concept Type))")` will fail

def expr_format_check(expr):
    try:
        parsed = loads(expr)
        # Expected format: (: <name> <body...>) where body has at least one element
        if isinstance(parsed, list) and len(parsed) >= 3 and parsed[0] == Symbol(':'):
            return (True, None)
        return (False, None)
    except Exception as e:
        print(f"Got an exception in expr_format_check for '{expr}': {e}")
        return (False, e)

def type_def_check(expr):
    expr = re.sub(r'\n\s*', ' ', expr)
    match = re.search(r'\(: .+ \(-> (.*)\)\)', expr)
    if not match:
        return False
    return True

def stmt_format_check(expr):
    try:
        parsed = loads(expr)
        # Expected format: (: <prf> <main> (STV <s> <c>))
        # parsed list: [Symbol(':'), <prf>, <main>, [Symbol('STV'), <s>, <c>]]
        if isinstance(parsed, list) and len(parsed) == 4 and parsed[0] == Symbol(':'):
            stv_part = parsed[3]
            if isinstance(stv_part, list) and len(stv_part) == 3 and stv_part[0] == Symbol('STV'):
                 return (True, None)
        return (False, None)
    except Exception as e:
        print(f"Got an exception in stmt_format_check for '{expr}': {e}")
        return (False, e)

def query_format_check_1(expr):
    try:
        parsed = loads(expr)
        # Expected format: (: <x> <y> <tv>)
        # parsed list: [Symbol(':'), <x>, <y>, <tv>]
        if isinstance(parsed, list) and len(parsed) == 4 and parsed[0] == Symbol(':'):
            return (True, None)
        return (False, None)
    except Exception as e:
        print(f"Got an exception in query_format_check_1 for '{expr}': {e}")
        return (False, e)

def query_format_check_2(expr):
    try:
        expr = re.sub(r'\n\s*', ' ', expr)
        match = re.search(r'\(: \$.+ \(.+\) \$.+\)', expr)
        if not match:
            return False
    except Exception as e:
        print(f"Got an exception in query_format_check_2 for '{expr}': {e}")
        return False
    return True

def metta_type_check(type_defs, stmt):
    temp_metta = MeTTa()
    try:
        for type_def in type_defs:
            type_def_atom = temp_metta.parse_all(type_def)[0]
            temp_metta.space().add_atom(type_def_atom)

        # try to type-check in MeTTa based on the given type definitions and see if we'll get an error
        rtn1 = temp_metta.run(f"!{stmt}")[0][0]
        rtn2 = temp_metta.run(f"!(car-atom {rtn1})")[0][0]
        if rtn2.get_name() == "Error":
            return (False, None)
        return (True, None)
    except Exception as e:
        print(f"Got an exception in metta_type_check for '{stmt}': {e}")
        return (False, e)

def unused_preds_check(type_defs, stmts):
    # Premises/Conclusions are structural wrappers used inside Implication, not user-defined predicates
    implication_keywords = ["Premises", "Conclusions"]
    preds_used = list(set(sum([re.findall(r'\((.+?) ', re.sub(r'\n\s*', ' ', expr)) for expr in stmts], [])))
    preds_defined = list(set([re.search(r'\(: (.+?) \(-> ', re.sub(r'\n\s*', ' ', type_def)).group(1) for type_def in type_defs]))
    filtered_preds_used = [item for item in preds_used if item not in (built_in_ops + special_symbols + implication_keywords) and not item.startswith('$')]
    filtered_preds_defined = [item for item in preds_defined if item not in (built_in_ops + special_symbols + implication_keywords)]
    preds_defined_not_used = [item for item in filtered_preds_defined if item not in filtered_preds_used]
    if preds_defined_not_used:
        return (False, preds_defined_not_used)
    else:
        return (True, [])

def undefined_preds_check(type_defs, stmts):
    implication_keywords = ["Premises", "Conclusions"]
    preds_used = list(set(sum([re.findall(r'\((.+?) ', re.sub(r'\n\s*', ' ', expr)) for expr in stmts], [])))
    preds_defined = list(set([re.search(r'\(: (.+?) \(-> ', re.sub(r'\n\s*', ' ', type_def)).group(1) for type_def in type_defs]))
    filtered_preds_used = [item for item in preds_used if item not in (built_in_ops + special_symbols + implication_keywords) and not item.startswith('$')]
    filtered_preds_defined = [item for item in preds_defined if item not in (built_in_ops + special_symbols + implication_keywords)]
    preds_used_not_defined = [item for item in filtered_preds_used if item not in filtered_preds_defined]
    if preds_used_not_defined:
        return (False, preds_used_not_defined)
    else:
        return (True, [])

def connectivity_check(stmts):
    def extract_elements(sexp):
        """
        Extract elements that are not predicates, also ignore:
        - strings
        - numbers
        - proof_names
        - variables
        """
        if sexp[0] == Symbol(":"):
            # ignore proof_names
            return extract_elements(sexp[1:])

        ele_lst = []
        # ignore predicates
        for ele in sexp[1:]:
            if isinstance(ele, list):
                ele_lst += extract_elements(ele)
            # ignore strings, numbers, etc that are not parsed as Symbols
            elif isinstance(ele, Symbol):
                # ignoring variables, assuming expressions should not be connected via a variable with the same name globally
                if not str(ele).startswith("$"):
                    ele_lst.append(str(ele))
        return ele_lst

    stmt_sexprs = [loads(stmt) for stmt in stmts]
    stmt_ele_lst = [extract_elements(sexpr) for sexpr in stmt_sexprs]

    # there could be exprs has no elements extracted, like an Implication rule with only predicates and variables, they can be excluded from connectivity check
    filtered_stmt_ele_lst = list(filter(lambda x: len(x) > 0, stmt_ele_lst))
    # print(f"Extracted elements (filtered): {filtered_stmt_ele_lst}")

    if len(filtered_stmt_ele_lst) <= 1:
        return True

    connected = {0}
    while True:
        new_connections = set()
        for i in connected:
            for j, other_list in enumerate(filtered_stmt_ele_lst):
                if j not in connected and set(filtered_stmt_ele_lst[i]) & set(other_list):
                    new_connections.add(j)
        if not new_connections:
            break
        connected.update(new_connections)

    return True if len(connected) == len(filtered_stmt_ele_lst) else False
