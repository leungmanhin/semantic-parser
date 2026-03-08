from graphs import get_pairwise_variations
from llm import to_openrouter
from prompts import add_bridging_rules_system_prompt, create_bridging_rules_prompt, BridgingRules

def generate_equivalence_expr(pred1, pred2, arity, stv_strength):
    """
    Generates an Equivalence expression string for two predicates with a given arity.
    Example: pred1="Eat", pred2="Consume", arity=2
    Returns: "(: eat_consume_eq (Equivalence (Eat $var_0 $var_1) (Consume $var_0 $var_1)) (STV 0.95 0.95))"
    """
    variables = [f"$var_{i}" for i in range(arity)]
    vars_str = " ".join(variables)
    rule_name = f"{pred1.lower()}_{pred2.lower()}_eq"
    return f"(: {rule_name} (Equivalence ({pred1} {vars_str}) ({pred2} {vars_str})) (STV {stv_strength:.3f} 0.95))"

def generate_bridging_rules(all_stmts):
    pairwise_analysis = get_pairwise_variations(all_stmts)

    bridging_rules = []

    for pair in pairwise_analysis:
        print(f"\nPair: Graph {pair['graph_x_idx']} vs Graph {pair['graph_y_idx']}")
        print(f"  Pairwise MCS Size: {len(pair['pairwise_mcs'])} relations")
        print(f"  Pairwise MCS: {pair['pairwise_mcs']}")
        print(f"  Variation X (rel. to Pairwise): {pair['variation_x']}")
        print(f"  Variation Y (rel. to Pairwise): {pair['variation_y']}")

        if not (pair['variation_x'] or pair['variation_y']):
            continue

        chat_history = [{
            "role": "system",
            "content": add_bridging_rules_system_prompt
        }]

        llm_outputs = to_openrouter(create_bridging_rules_prompt(pair["pairwise_mcs"], pair["variation_x"], pair["variation_y"]), history=chat_history, output_format=BridgingRules)

        print(f"Bridging rules generated: {llm_outputs['bridging_rules']}")
        bridging_rules += llm_outputs['bridging_rules']

    # TODO: filter out duplicates and useless expressions (e.g. type defs) more robustly
    return list(set(bridging_rules))
