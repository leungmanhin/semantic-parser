special_symbols = [
    ":",
    "->"
]

built_in_ops = [
    "And",
    "Or",
    "Not",
    "Implication",
    "Equivalence",
    "Similarity",
    "STV",
    "LikelierThan",
    "TemporalBefore",
    "TemporalAfter",
    "TemporalContained",
    "TemporalOverlap",
    "IsA",
    "HasAttribute",
    "Has",
]

built_in_type_defs = [
    "(: And (-> Type Type Type))",
    "(: Or (-> Type Type Type))",
    "(: Not (-> Type Type))",
    "(: Implication (-> Type Type Type))",
    "(: Equivalence (-> Type Type Type))",
    "(: Similarity (-> Concept Concept Type))",
    "(: STV (-> Number Number TV))",
    "(: LikelierThan (-> Type Type Type))",
    "(: TemporalBefore (-> Concept Concept Type))",
    "(: TemporalAfter (-> Concept Concept Type))",
    "(: TemporalContained (-> Concept Concept Type))",
    "(: TemporalOverlap (-> Concept Concept Type))",
    "(: IsA (-> Concept Concept Type))",
    "(: HasAttribute (-> Concept Concept Type))",
    "(: Has (-> Concept Concept Type))",
]

pad_str = "\n- "
built_in_ops_str = pad_str + pad_str.join(built_in_ops)
built_in_type_defs_str = pad_str + pad_str.join(built_in_type_defs)

additional_rules = [
    "(: similarity_1 (Implication (And ($pred $x) (Similarity $x $y)) ($pred $y)) (STV 0.9 0.9))",
    "(: similarity_2l (Implication (And ($pred $x $y) (Similarity $x $z)) ($pred $z $y)) (STV 0.9 0.9))",
    "(: similarity_2r (Implication (And ($pred $x $y) (Similarity $y $z)) ($pred $x $z)) (STV 0.9 0.9))",
    # "(: equivalence_to_implication (Implication (Equivalence $x $y) (And (Implication $x $y) (Implication $y $x))) (STV 0.9 0.9))",
]
