import re
import networkx as nx
from collections import defaultdict

from built_in import *

def get_pairwise_variations(stmt_sets, alpha_renaming=True):
    """
    Computes the Maximum Common Subgraph (MCS) among all input graphs (Global MCS),
    the variations of each graph relative to Global MCS (Global Variations),
    and performs a pairwise analysis to find the Pairwise MCS and variations for each pair.

    Args:
        stmt_sets: List of lists of PLN statements.
        alpha_renaming: Whether to allow variable renaming during MCS matching.
                        If True, variables in the unique parts (variations) of each graph are
                        also normalized/renamed to match the variable names used in the MCS.

    Returns:
        pairwise_analysis: List of dicts, each containing:
            - graph_x_idx, graph_y_idx
            - pairwise_mcs: MCS specific to these two graphs
            - variation_x: X's variation relative to Pairwise MCS
            - variation_y: Y's variation relative to Pairwise MCS
    """

    # --- Helper Functions ---

    def simple_sexp_parser(s):
        """
        A simple recursive S-expression parser that handles strings with spaces.
        Returns a nested list structure.
        """
        tokens = re.findall(r'\(|\)|\"[^\"]*\"|[^\s()]+', s)
        stack = [[]]
        for token in tokens:
            if token == '(':
                stack.append([])
            elif token == ')':
                if len(stack) > 1:
                    finished_list = stack.pop()
                    stack[-1].append(finished_list)
            else:
                stack[-1].append(token)
        return stack[0][0] if stack[0] else []

    def flatten_core(sexp):
        """
        Extracts relations from a parsed S-expression.
        """
        if not sexp:
            return []
        if len(sexp) >= 3 and sexp[0] == ':':
            return flatten_core(sexp[2])
        if isinstance(sexp, list) and len(sexp) > 0 and sexp[0] == 'And':
            relations = []
            for child in sexp[1:]:
                relations.extend(flatten_core(child))
            return relations
        return [sexp]

    def is_constant(token):
        if not isinstance(token, str):
            return False
        if token.startswith('"') or token[0].isdigit():
            return True
        if token[0].isupper() and token != 'sentence_creation_time':
            return True
        if token == 'sentence_creation_time':
            return True
        return False

    def parse_stmts_to_rels(stmts):
        rels = []
        for s in stmts:
            parsed = simple_sexp_parser(s)
            rels.extend(flatten_core(parsed))
        return rels

    def deep_match(t1, t2, mapping, alpha_renaming=True):
        if isinstance(t1, list) and isinstance(t2, list):
            if len(t1) != len(t2):
                return False
            for c1, c2 in zip(t1, t2):
                if not deep_match(c1, c2, mapping, alpha_renaming):
                    return False
            return True
        elif isinstance(t1, list) or isinstance(t2, list):
            return False
        else:
            if not alpha_renaming:
                return t1 == t2
            else:
                if is_constant(t1) or is_constant(t2):
                    return t1 == t2
                else:
                    if t1 in mapping:
                        return mapping[t1] == t2
                    else:
                        if t2 in mapping.values():
                            return False
                        mapping[t1] = t2
                        return True

    def get_max_common_subgraph(stmt_sets, alpha_renaming=True):
        if not stmt_sets:
            return []

        def pairwise_mcs(rels1, rels2):
            matches = []
            for i, r1 in enumerate(rels1):
                for j, r2 in enumerate(rels2):
                    if not isinstance(r1, list) or not isinstance(r2, list): continue
                    if len(r1) != len(r2): continue

                    compatible = True
                    mapping_candidates = {}

                    if not deep_match(r1, r2, mapping_candidates, alpha_renaming):
                        compatible = False

                    if compatible:
                        matches.append((i, j, mapping_candidates))

            G = nx.Graph()
            for idx, (i, j, mapping) in enumerate(matches):
                G.add_node(idx, r1_idx=i, r2_idx=j, mapping=mapping)

            node_indices = list(G.nodes())
            for idx_u in range(len(node_indices)):
                for idx_v in range(idx_u + 1, len(node_indices)):
                    u = node_indices[idx_u]
                    v = node_indices[idx_v]
                    node_u = G.nodes[u]
                    node_v = G.nodes[v]

                    if node_u['r1_idx'] == node_v['r1_idx'] or node_u['r2_idx'] == node_v['r2_idx']:
                        continue

                    map_u = node_u['mapping']
                    map_v = node_v['mapping']
                    consistent = True

                    common_vars = set(map_u.keys()) & set(map_v.keys())
                    for var in common_vars:
                        if map_u[var] != map_v[var]:
                            consistent = False; break
                    if not consistent: continue

                    inv_map_u = {val: key for key, val in map_u.items()}
                    inv_map_v = {val: key for key, val in map_v.items()}
                    common_ranges = set(inv_map_u.keys()) & set(inv_map_v.keys())
                    for val in common_ranges:
                        if inv_map_u[val] != inv_map_v[val]:
                            consistent = False; break
                    if not consistent: continue

                    G.add_edge(u, v)

            if G.number_of_nodes() == 0:
                return []

            clique = nx.algorithms.clique.max_weight_clique(G, weight=None)
            clique_nodes = clique[0]
            result_relations = [rels1[G.nodes[node_idx]['r1_idx']] for node_idx in clique_nodes]
            return result_relations

        # Main Loop
        current_rels = parse_stmts_to_rels(stmt_sets[0])

        for i in range(1, len(stmt_sets)):
            next_rels = parse_stmts_to_rels(stmt_sets[i])
            current_rels = pairwise_mcs(current_rels, next_rels)
            if not current_rels:
                break

        return current_rels

    def get_graph_variations(stmt_sets, alpha_renaming=True):
        # 1. Compute MCS
        mcs_rels = get_max_common_subgraph(stmt_sets, alpha_renaming)
        variations = []

        # 2. Compare each graph against MCS
        for i, stmts in enumerate(stmt_sets):
            rels_i = parse_stmts_to_rels(stmts)

            # --- Reuse pairwise matching logic to embed MCS into Graph i ---
            matches = []
            for idx_m, r_m in enumerate(mcs_rels):
                for idx_g, r_g in enumerate(rels_i):
                    if len(r_m) != len(r_g): continue

                    compatible = True
                    mapping_candidates = {}

                    if not deep_match(r_m, r_g, mapping_candidates, alpha_renaming):
                        compatible = False

                    if compatible:
                        matches.append((idx_m, idx_g, mapping_candidates))

            G = nx.Graph()
            for idx, (idx_m, idx_g, mapping) in enumerate(matches):
                G.add_node(idx, r1_idx=idx_m, r2_idx=idx_g, mapping=mapping)

            # Build consistency edges
            node_indices = list(G.nodes())
            for u_i in range(len(node_indices)):
                for v_i in range(u_i + 1, len(node_indices)):
                    u = node_indices[u_i]
                    v = node_indices[v_i]
                    node_u = G.nodes[u]
                    node_v = G.nodes[v]

                    if node_u['r1_idx'] == node_v['r1_idx'] or node_u['r2_idx'] == node_v['r2_idx']:
                        continue

                    map_u = node_u['mapping']
                    map_v = node_v['mapping']
                    consistent = True

                    common_vars = set(map_u.keys()) & set(map_v.keys())
                    for var in common_vars:
                        if map_u[var] != map_v[var]:
                            consistent = False; break
                    if not consistent: continue

                    inv_map_u = {val: key for key, val in map_u.items()}
                    inv_map_v = {val: key for key, val in map_v.items()}
                    common_ranges = set(inv_map_u.keys()) & set(inv_map_v.keys())
                    for val in common_ranges:
                        if inv_map_u[val] != inv_map_v[val]:
                            consistent = False; break
                    if not consistent: continue

                    G.add_edge(u, v)

            if G.number_of_nodes() == 0:
                clique_nodes = []
            else:
                clique = nx.algorithms.clique.max_weight_clique(G, weight=None)
                clique_nodes = clique[0]

            combined_mapping = {}
            covered_indices_g = set()

            for node_idx in clique_nodes:
                node_data = G.nodes[node_idx]
                combined_mapping.update(node_data['mapping'])
                covered_indices_g.add(node_data['r2_idx'])

            remainder = [r for idx, r in enumerate(rels_i) if idx not in covered_indices_g]

            variations.append({
                "graph_idx": i,
                "mapping_from_mcs": combined_mapping,
                "remainder": remainder
            })

        return mcs_rels, variations

    # --- 1. Global Analysis ---
    # Compute Global MCS and variations relative to it
    global_mcs, raw_global_vars = get_graph_variations(stmt_sets, alpha_renaming)

    # Normalize Global Variations
    global_variations = []
    for v in raw_global_vars:
        idx = v['graph_idx']
        remainder = v['remainder']

        # Map variables in the remainder back to MCS variables for consistency
        mapping = v['mapping_from_mcs']
        inv_mapping = {val: key for key, val in mapping.items()}
        def norm(sexp):
            if isinstance(sexp, list):
                return [norm(x) for x in sexp]
            elif sexp in inv_mapping:
                return inv_mapping[sexp]
            else:
                return sexp

        global_variations.append({
            "graph_idx": idx,
            "variation": [norm(r) for r in remainder]
        })

    # --- 2. Pairwise Analysis ---
    pairwise_analysis = []
    indices = range(len(stmt_sets))

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            idx1 = indices[i]
            idx2 = indices[j]

            # Re-run graph variations analysis on just this pair to find Pairwise MCS
            pair_sets = [stmt_sets[idx1], stmt_sets[idx2]]
            pair_mcs, pair_vars_raw = get_graph_variations(pair_sets, alpha_renaming)

            # Process variations for this pair (Normalize to Pairwise MCS)
            processed_pair_vars = {}
            for v in pair_vars_raw:
                # v['graph_idx'] will be 0 or 1 (relative to pair_sets)
                # Map back to real indices idx1, idx2
                real_idx = idx1 if v['graph_idx'] == 0 else idx2
                remainder = v['remainder']

                mapping = v['mapping_from_mcs']
                inv_mapping = {val: key for key, val in mapping.items()}
                def norm_pair(sexp):
                    if isinstance(sexp, list):
                        return [norm_pair(x) for x in sexp]
                    elif sexp in inv_mapping:
                        return inv_mapping[sexp]
                    else:
                        return sexp
                processed_pair_vars[real_idx] = [norm_pair(r) for r in remainder]

            pairwise_analysis.append({
                "graph_x_idx": idx1,
                "graph_y_idx": idx2,
                "pairwise_mcs": pair_mcs,
                "variation_x": processed_pair_vars[idx1],
                "variation_y": processed_pair_vars[idx2]
            })

    print(f"Global MCS Size: {len(global_mcs)} relations")
    for expr in global_mcs:
        print(expr)

    print("\nGlobal Variations (Relative to Global MCS)")
    for v in global_variations:
        print(f"Graph {v['graph_idx']}: {v['variation']}")

    return pairwise_analysis

def extract_predicates_with_arity(expressions):
    """
    Parses PLN expressions and extracts predicates with their arities from the core expression.
    """
    def simple_sexp_parser(s):
        tokens = re.findall(r'\(|\)|\"[^\"]*\"|[^\s()]+', s)
        stack = [[]]
        for token in tokens:
            if token == '(':
                stack.append([])
            elif token == ')':
                if len(stack) > 1:
                    finished_list = stack.pop()
                    stack[-1].append(finished_list)
            else:
                stack[-1].append(token)
        return stack[0][0] if stack[0] else []

    predicates = []

    def traverse(node):
        if isinstance(node, list) and len(node) > 0:
            head = node[0]
            # Assuming head is the predicate
            if isinstance(head, str):
                # We record (head, arity)
                # Arity is length of the list minus 1 (the head itself)
                predicates.append((head, len(node) - 1))

            # Recursively traverse arguments
            for child in node[1:]:
                traverse(child)

    for expr in expressions:
        parsed = simple_sexp_parser(expr)
        # Expected format: (: prf_name core_expr stv)
        # parsed list: [':', 'prf_name', core_expr, stv]
        if len(parsed) >= 3 and parsed[0] == ':':
            core_expr = parsed[2]
            traverse(core_expr)

    # Filter out duplicates and the built-in operators/predicates
    predicates = [p for p in list(set(predicates)) if p[0] not in built_in_ops]

    return predicates
