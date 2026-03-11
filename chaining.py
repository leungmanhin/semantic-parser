import multiprocessing
import os
import re
import time
import queue
from pettachainer.pettachainer import PeTTaChainer


def _parse_sexp(tokens):
    token = tokens.pop(0)
    if token == '(':
        lst = []
        while tokens[0] != ')':
            lst.append(_parse_sexp(tokens))
        tokens.pop(0)
        return lst
    else:
        return token


def _sexp_to_string(node):
    if isinstance(node, list):
        return f"({' '.join(_sexp_to_string(n) for n in node)})"
    return node


def flatten_connectives(expr: str) -> str:
    def flatten(node):
        if not isinstance(node, list) or not node:
            return node
        flattened_children = [flatten(child) for child in node[1:]]
        head = node[0]
        if head not in ('And', 'Or'):
            return [head] + flattened_children
        merged_children = []
        for child in flattened_children:
            if isinstance(child, list) and child and child[0] == head:
                merged_children.extend(child[1:])
            else:
                merged_children.append(child)
        return [head] + merged_children

    tokens = re.findall(r'\(|\)|[^\s()]+', expr)
    tree = _parse_sexp(tokens)
    flat_tree = flatten(tree)
    return _sexp_to_string(flat_tree)


def equivalence_to_implications(expr: str) -> tuple[str, str]:
    tokens = re.findall(r'\(|\)|[^\s()]+', expr)
    tree = _parse_sexp(tokens)
    # Expected: [':', prf_name, ['Equivalence', expr1, expr2], stv]
    prf_name = tree[1]
    expr1 = tree[2][1]
    expr2 = tree[2][2]
    stv = tree[3]
    fwd = _sexp_to_string([':', f'{prf_name}_fwd', ['Implication', expr1, expr2], stv])
    bwd = _sexp_to_string([':', f'{prf_name}_bwd', ['Implication', expr2, expr1], stv])
    return fwd, bwd


def _expand_equivalences(kb: list[str]) -> list[str]:
    result = []
    for expr in kb:
        tokens = re.findall(r'\(|\)|[^\s()]+', expr)
        try:
            tree = _parse_sexp(tokens)
            if (isinstance(tree, list) and len(tree) == 4
                    and isinstance(tree[2], list) and tree[2]
                    and tree[2][0] == 'Equivalence'):
                result.extend(equivalence_to_implications(expr))
                continue
        except Exception:
            pass
        result.append(expr)
    return result


def build_kb_handler(kb):
    handler = PeTTaChainer()
    kb = [flatten_connectives(x) for x in kb]
    kb = _expand_equivalences(kb)
    for x in kb:
        print(f"... adding to space: {x}")
        handler.add_atom(x.replace("'", ""))
    return handler


def _main_chaining(kb, query, result_queue, handler, max_depth):
    # print(f"Chaining (handler = {handler}):\n```\nkb = {kb}\nquery = {query}\n```")

    # post-process to work more efficiently
    kb = [flatten_connectives(x) for x in kb]
    kb = _expand_equivalences(kb)
    query = flatten_connectives(query)
    print(f"Chaining (post-processed):\n```\nkb = {kb}\nquery = {query}\n```")

    # may impact efficiency significantly
    # kb += additional_rules

    if handler is None:
        handler = PeTTaChainer()

    try:
        for x in kb:
            print(f"... adding to space: {x}")
            handler.add_atom(x.replace("'", ""))
    except Exception as e:
        print(f"\n!!! EXCEPTION: {e}\n")

    depth = 0
    result = []
    start_time = time.time()
    print(f"... chaining for: {query}")
    try:
        while ((not result) and (depth < max_depth)):
            depth += 1
            print(f"... chaining with depth = {depth}")
            result = handler.query(query, depth=depth)
    except Exception as e:
        print(f"\n!!! EXCEPTION: {e}\n")

    end_time = time.time()
    print(f"Chaining result: {result}\n(Time used: {end_time - start_time} seconds)\n")
    result_queue.put(result)


def chaining(kb, query, handler=None, timeout=30, max_depth=10):
    result = None
    chaining_return_queue = multiprocessing.Queue()

    chaining_process = multiprocessing.Process(
        target = _main_chaining,
        args = (kb, query, chaining_return_queue, handler, max_depth)
    )

    chaining_process.start()

    try:
        result = chaining_return_queue.get(timeout=timeout)
    except queue.Empty:
        pass

    chaining_process.join(timeout=1)

    if chaining_process.is_alive():
        print(f"... chaining_process is taking too long (>= {timeout} seconds), terminating")
        chaining_process.terminate()
        chaining_process.join()
        print("... chaining_process terminated")

    return result
