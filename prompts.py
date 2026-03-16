from pydantic import BaseModel

from built_in import *

class PLNExprs(BaseModel):
    type_defs: list[str]
    stmts: list[str]

class PLNQueryExprs(BaseModel):
    type_defs: list[str]
    stmts: list[str]
    queries: list[str]

class AddPLNExprs(BaseModel):
    type_defs: list[str]
    rules: list[str]
    rules_nl: list[str]

class NLSents(BaseModel):
    sentences: list[str]

class BridgingRules(BaseModel):
    bridging_rules: list[str]

def _render_context(context) -> str:
    """
    Renders a list of context sections into a string.

    Each section is a dict with:
      - "title"   (optional str): header label for this section
      - "entries" (optional list): each entry is {"sentence": str, "stmts": list[str]}
      - "content" (optional str): free-form text (used instead of entries)
    """
    parts = []
    for section in context:
        title = section.get("title", "")
        if title:
            parts.append(f"### {title} ###")
        if "entries" in section:
            for entry in section["entries"]:
                parts.append(f"sentence:\n{entry['sentence']}")
                parts.append("stmts:")
                for stmt in entry["stmts"]:
                    parts.append(f"{stmt}")
                parts.append("")
        elif "content" in section:
            parts.append(section["content"])
        parts.append("")  # blank line between sections
    return "\n".join(parts).strip()

def create_nl2pln_parsing_prompt(text, context=[]):
    context_str = _render_context(context) if context else ""
    return f"<context>\n{context_str}\n</context>\n\n<input_text>\n{text}\n</input_text>".strip()

def create_nl2pln_querying_prompt(text, context=[]):
    context_str = _render_context(context) if context else ""
    return f"<context>\n{context_str}\n</context>\n\n<input_question>\n{text}\n</input_question>".strip()

def create_nl2pln_correction_prompt(correction):
    return f"<correction_comments>\n{correction}\n</correction_comments>".strip()

def create_missing_exprs_prompt(k_exprs, q_expr):
    return f"<knowledge_exprs>\n{k_exprs}\n</knowledge_exprs>\n\n<query_expr>\n{q_expr}\n</query_expr>".strip()

def create_pln2nl_prompt(target_exprs):
    return f"<target_exprs>\n{target_exprs}\n</target_exprs>".strip()

def create_bridging_rules_prompt(max_common_subgraph, variations_1, variations_2):
    return f"<max_common_subgraph>\n{max_common_subgraph}\n</max_common_subgraph>\n\n<variations_1>\n{variations_1}\n</variations_1>\n\n<variations_2>\n{variations_2}\n</variations_2>".strip()


base_instructions = f"""
Here are some fundamental guidelines you need to follow when converting a natural language sentence into a kind of meaning representation that is friendly to a reasoner called PLN:
- one or more expressions should be created as a result of this conversion, which collectively capture the semantic meaning of the given sentence, and each of these expressions is referred to as a "PLN expression"
- all PLN expressions must be expressed in the format of either `(: <expr_name> <expr_body>)` or `(: <expr_name> <expr_body> <truth_value>)`
- the `expr_name` (aka `proof_name`) can just be any arbitrary string written in snake_case, but by convention it's preferable to somewhat reflect the meaning of the `expr_body` in some shorthand form, and it must be uniquely identifiable globally
- ensure the `expr_name` is strictly an identifier and remains entirely excluded from the `expr_body`
- for PLN expressions that are type definitions, they should be in the format of `(: <expr_name> <expr_body>)`
- for PLN expressions that are not type definitions, they should be in the format of `(: <expr_name> <expr_body> <truth_value>)`, with the `truth_value` provided using the built-in constructor `STV`, which expects a strength and a confidence value
- to denote an instance, name it after the core noun or phrase it refers to, written in snake_case, stripping away determiners like "the", "a", "an" and using the singular base form regardless of grammatical number, and always append a numeric suffix `_1`, `_2`, etc. to distinguish specific instances from generic concept names (e.g. "the cat" → `cat_1`, "Alice" → `alice_1`, "a rain event" → `rain_evt_1`); for event instances specifically, use the pattern `<name>_evt_1`, `<name>_evt_2`, etc. (e.g. "a cutting event" → `cut_evt_1`); for group instances, use `<name>_group_1` (e.g. "the cats" → `cat_group_1`); if the same entity is referred to more than once within the same sentence, reuse the exact same instance name rather than creating a new one; only use `_2`, `_3`, etc. when two or more distinct entities in the same sentence would otherwise share the same base name (e.g. "Alice met another Alice" → `alice_1` and `alice_2`); global uniqueness across sentences is handled by post-processing, so you do not need to worry about collisions with other sentences
- to denote a variable, come up with an arbitrary name written in snake_case and prefix it with a '$' symbol; how it's named doesn't matter, but it must be uniquely identifiable within its scope
- to denote a predicate (i.e. a verb or multi-argument relation — not nouns or adjectives, which are handled as concept names via `IsA`, `HasAttribute`, and `Has`), it's preferable to name it using the original word or phrase in the given sentence without adding any pre- or post-fix, and it must be written in UpperCamelCase
- you must create a type definition for every new predicate that you are creating and using
- no instance can exist on its own within the `expr_body`, each instance needs to appear within a predicate application; how each linguistic term is captured depends on its role:
  - nouns and pronouns denoting categories or kinds appear as the second argument of `IsA`, e.g. "Fido is a dog" → `(IsA fido_1 dog)`, where `dog` is a concept name, not a predicate
  - adjectives denoting intrinsic qualities appear as the second argument of `HasAttribute`, e.g. "Fido is fast" → `(HasAttribute fido_1 fast)`, where `fast` is a concept name, not a predicate; adverbs are handled differently — see the `HasAttribute` restrictions below
  - verbs and multi-argument relations become UpperCamelCase predicates applied to their arguments, e.g. "Alice loves Bob" → `(Loves alice_1 bob_1)`
  - all of these are arranged and grouped together with the built-in logical operators or other predicates to constitute the `expr_body`
- similarly no predicate can exist on its own without associating with an instance or a variable; there are three built-in predicates for the most common instance-to-concept relations (one instance plus one generic concept), and you should use them instead of bare custom predicates wherever they apply:
  - `(IsA <instance> <category>)` — for categorical membership / "is-a" (e.g. "Fido is a dog" → `(IsA fido_1 dog)`); the category name is written in snake_case as a concept, not as a predicate
  - `(HasAttribute <instance> <property>)` — for properties and attributes (e.g. "Fido is fast" → `(HasAttribute fido_1 fast)`); the property name is written in snake_case
  - `(Has <instance> <condition>)` — for afflictions, conditions, and abstract states (e.g. "Bob has cancer" → `(Has bob_1 cancer)`); the condition name is written in snake_case as a concept, not as a predicate
  - for relations between two or more instances (e.g. "Alice loves Bob", "Bob gave Alice a book"), use a custom predicate in UpperCamelCase as before: `(Loves alice_1 bob_1)`, `(Gave bob_1 alice_1 book_1)`
- it's highly preferable to have predicates that are as simple as possible, e.g. a single-word predicate is much more preferable than a multi-word predicate, so you must try to construct the PLN expressions using predicates that are as atomic and elementary as possible, but if a multi-word verb predicate is necessary, unpack its meaning with an `Implication` in terms of simpler predicates in sensible ways
- `HasAttribute` is only appropriate for intrinsic scalar or qualitative properties that directly describe the instance itself (e.g. color, size, shape, temperature, speed) — e.g. `(HasAttribute door_1 red)` for "the door is red", `(HasAttribute alice_1 tall)` for "Alice is tall"; do NOT use it in these cases:
  - when the quality belongs to a related action rather than the instance itself — e.g. in "she walks fast", "fast" modifies the walking action, not the person, so `(HasAttribute alice_1 fast)` would be wrong; instead reify the event and attribute the quality to it: `(And (IsA walk_evt_1 walk_event) (Subject walk_evt_1 alice_1) (HasAttribute walk_evt_1 fast))`
  - when the "attribute" is actually a category or role the instance belongs to — use `IsA` instead, e.g. "Bob is a smoker" → `(IsA bob_1 smoker)`, not `(HasAttribute bob_1 smoker)`
  - when the "attribute" is a condition or entity the instance possesses rather than a quality that describes it — use `Has` instead, where the second argument is a generic concept name (parallel to `IsA` and `HasAttribute`), e.g. "Bob has cancer" → `(Has bob_1 cancer)`, not `(HasAttribute bob_1 cancer)`
- when using `IsA` with a compound concept name (e.g. `steam_engine`, `red_door`), add an `Implication` unpacking what that concept entails so the reasoner can use it; unpack into the head category plus whatever the modifier contributes — using `HasAttribute` if the modifier is a direct intrinsic property of the instance, or a proper binary relation if the modifier denotes a related concept: e.g. `(Implication (IsA $x red_door) (And (IsA $x door) (HasAttribute $x red)))` is correct since red is a direct property of the door, but for `steam_engine` the engine is not itself "steam" — so express the actual relation introducing a fresh RHS-only variable `$y` for the steam instance (existentially quantified per the variable scoping rule — asserting that some steam entity exists and powers the engine): `(Implication (IsA $x steam_engine) (And (IsA $x engine) (IsA $y steam) (PoweredBy $x $y)))`
- you should create an instance for each specific (i.e. non-generic) entity/concept that exists in the input sentence; if it's a pronoun that co-refers to another entity mentioned within the same sentence, do NOT create a separate instance for it — simply reuse that entity's instance name (consistent with the naming rule above, e.g. "Alice hurt herself" → use `alice_1` for both "Alice" and "herself"); if it's a pronoun whose referent comes from a prior sentence (i.e. cross-sentence coreference), you can use the built-in operator `Similarity` to equate the pronoun instance to the referent instance with a sensible truth value attached
- generic concepts/properties/conditions can only exist as the second argument of a relation using either `IsA`, `HasAttribute`, or `Has` as predicate, all other relations should use instances/variables instead in their argument slots
- when a sentence contains an indefinite noun phrase (e.g. "a cat", "an idea", "some book") that refers to a particular entity in context — even if its identity is unknown or unnamed — create a fresh instance for it (e.g. "A cat sat on the mat" → create instance `cat_1`); contrast this with a generic or categorical use of an indefinite NP that makes a claim about the whole category (e.g. "A cat is a mammal" = "All cats are mammals"), which should be represented as a generic `Implication` with a variable rather than a fresh instance; the key test: does the sentence report a fact about a specific individual (even unnamed), or make a general statement about the category as a whole?
- for quantities you can create and make use of a `Quantity` predicate with type definition `(: Quantity (-> Concept Number Type))`, then e.g. "3 apples" can be represented as `(And (IsA apple_group_1 apple) (Quantity apple_group_1 3))`, and similarly for named entities e.g. `(: Name (-> Concept String Type))`, or units e.g. `(: Unit (-> Concept String Type))`, or anything of this nature
- for possession and relationships e.g. "Ben's cat" or "my sister", etc., you can create and make use of a `Possess` predicate with type definition `(: Possess (-> Concept Concept Type))`; use `Possess` when "has/have" expresses ownership or belonging of a concrete entity (e.g. "Bob has a cat" → `(Possess bob_1 cat_1)`), and use `Has` when "has/have" expresses an affliction, condition, or abstract state (e.g. "Bob has cancer", "Bob has a fever", "Bob has a problem")
- for quantification, it's represented using the built-in quantifier `Implication`, and reflecting the level of quantification via `STV`, e.g. 'all' should have a relatively high truth value close to 1.0 while 'none' should have a relatively low truth value close to 0.0, and fuzzy quantifiers like 'most', 'many', 'some', 'a few', etc... should have a truth value that lies somewhat in between
- typically `Implication` is used to express generic/categorical "IsA" or if-then relationships — i.e. when a statement applies to a category as a whole using variables (e.g. "A cat is a mammal" → `(Implication (IsA $x cat) (IsA $x mammal))` meaning all cats are mammals); by contrast, instance-level "is-a" (e.g. "Fido is a dog"), "has-property-of" (e.g. "Fido is fast"), and "has-condition" (e.g. "Bob has cancer") all use `IsA`, `HasAttribute`, and `Has` directly — `(IsA fido_1 dog)`, `(HasAttribute fido_1 fast)`, and `(Has bob_1 cancer)` respectively — not `Implication`
- variables that appear on the left hand side of an `Implication` are universally quantified and scoped within that `Implication` (and everything nested within it); variables that appear only on the right hand side are existentially quantified — they assert that some entity satisfying that description exists in the consequent (e.g. in `(Implication (IsA $x author) (And (IsA $y book) (Wrote $x $y)))`, `$x` is universally quantified over authors while `$y` asserts the existence of a book that each such author wrote)
- if there is more than one PLN expression involving instances, they must form a single connected component within the output for the current sentence — meaning every such expression shares at least one instance name with at least one other expression, so the reasoner can navigate between them; purely generic expressions that contain only variables and predicates (e.g. a bridging `Implication` rule with no instances) are exempt from this requirement
- if the given sentence can be interpreted/understood in more than one way, e.g. literal vs idiomatic, literal vs metaphoric, and/or coreferences being resolved differently, etc., you should capture all these possible interpretations in the PLN expressions as well, with a truth value reflecting how likely each interpretation is based on your judgement on the given information
- for propositional attitude verbs (e.g. "said", "believes", "thinks", "knows", "claims", "argues"), the complement clause is a full proposition and must be reified: create a predicate for the attitude verb with type definition `(: <AttitudeVerb> (-> Concept Type Type))`, then pass the subject instance as the first argument and the entire proposition expression as the second argument, e.g. "Alice believes Bob is smart" → `(: alice_believes_bob_smart (Believes alice_1 (HasAttribute bob_1 smart)) (STV 0.9 0.9))`; for reported speech the proposition should be represented as faithfully as possible using the same PLN conventions, and the STV on the outer attitude statement should reflect how confident the reporter is that the attitude is held, not whether the embedded proposition is true
- if you decide to represent an action using the event-based (e.g., Neo-Davidsonian) style because it has complex modifiers (e.g., tools, locations), you MUST ALSO generate a generic bridge `Implication` that maps this specific event structure back to its simple, core multi-arity format so as to ensure the knowledge base can connect complex events to simple queries, for example when parsing "Bob cut the apple with a knife", you can generate the specific event representation like `(And (IsA cut_evt_1 cut_event) (Subject cut_evt_1 bob_1) (Object cut_evt_1 apple_1) (Instrument cut_evt_1 knife_1))`, and because you used the event style for "Cut", you MUST also output the bridge rule for it: `(: bridge_cut_event (Implication (And (IsA $e cut_event) (Subject $e $s) (Object $e $o)) (Cut $s $o)) (STV 1.0 1.0))`; note that illustrative examples elsewhere in these guidelines may show event-based representations in abbreviated form without the bridge rule for brevity — the bridge rule is always required in actual output whenever event-based style is used
- when a sentence is in passive voice (e.g. "The apple was eaten by Bob"), resolve it to the same active-voice event role representation as its active counterpart rather than creating a new passive-specific predicate; map the grammatical subject back to its semantic role (i.e. Object) and the by-phrase back to the Subject, so "The apple was eaten by Bob" yields the same representation as "Bob ate the apple": `(And (IsA eat_evt_1 eat_event) (Subject eat_evt_1 bob_1) (Object eat_evt_1 apple_1))` with the mandatory bridge `Implication` if the event style is used
- for causal relations (e.g. "because", "caused", "led to", "due to"), distinguish between two cases: (1) a specific event-to-event causal link — use a `Cause` predicate with type definition `(: Cause (-> Concept Concept Type))` relating the two event instances, e.g. "The rain caused the flood" → `(And (IsA rain_evt_1 rain_event) (IsA flood_evt_1 flood_event) (Cause rain_evt_1 flood_evt_1))`; (2) a generic causal law over categories (e.g. "Smoking causes cancer", "Heat melts ice") — use `Implication` with appropriate STV as you would for any generic rule, e.g. `(: smoking_causes_cancer (Implication (IsA $x smoker) (Has $x cancer)) (STV 0.9 0.9))`
- for purpose or goal relations (e.g. "in order to", "so that", "to achieve"), use a `Purpose` predicate with type definition `(: Purpose (-> Concept Concept Type))` relating an action event instance to a goal event instance, e.g. "She studied in order to pass" → `(And (IsA study_evt_1 study_event) (IsA pass_evt_1 pass_event) (Purpose study_evt_1 pass_evt_1))`
- for modal verbs (e.g., could, might, should), treat them as higher-order predicates that take a proposition as their argument e.g., having a type definition of `(: <ModalName> (-> Type Type))`; the reason modal verbs are not folded into STV (unlike epistemic adverbs such as "probably") is that they express qualitatively distinct modalities — possibility, ability, obligation, permission — that cannot be faithfully reduced to a single probability scalar; epistemic adverbs such as "probably", "likely", and "possibly" do directly quantify likelihood and so are handled by adjusting STV strength (see the epistemic hedges rule below) rather than creating a predicate; evidential hedges such as "apparently" and "allegedly" also lower STV strength but additionally may warrant an `AttributedTo` link to a source (see the epistemic hedges rule below)
- if applying a modal to a multi-arity predicate, wrap the entire expression: `(Could (Cut bob_1 apple_1))`; if applying a modal to a Neo-Davidsonian event, apply it to the entire event structure: `(Could (And (IsA cut_evt_1 cut_event) (Subject cut_evt_1 bob_1) (Object cut_evt_1 apple_1)))`
- when encountering a "collective AND" where multiple entities perform an action together rather than independently (e.g., "Alice and Bob lifted the piano together"), DO NOT use the logical `And` operator to duplicate the action but instead create an instance representing the group (e.g., `alice_bob_group_1`), and use a `Member` predicate with type definition `(: Member (-> Concept Concept Type))` to assign the individual entities to that group (e.g., `(And (Member alice_1 alice_bob_group_1) (Member bob_1 alice_bob_group_1) (Lift alice_bob_group_1 piano_1))`)
- `Not` must always wrap a complete predicate application — never a bare instance or a bare predicate name; e.g. `(Not (HasAttribute water_1 hot))` is correct, while `(Not water_1)` or `(Not hot)` are wrong; it can appear as the body of a standalone statement (e.g. `(: prf (Not (HasAttribute water_1 hot)) (STV 0.95 0.9))`) or as the consequent of an `Implication` (e.g. `(Implication (HasAttribute $x frozen) (Not (HasAttribute $x liquid)))`); the `STV` strength on a `Not`-bearing statement should reflect how strongly the negation holds — the system internally derives the inner expression's strength as `1 - s`, so `(STV 0.95 0.9)` means the negation holds with 95% strength, implying the inner predicate has a derived strength of 0.05
- for epistemic hedges and attribution markers (e.g. "probably", "apparently", "allegedly", "according to X", "it seems that"), lower the strength value of the statement being hedged rather than creating a separate predicate: e.g. "It will probably rain" → `(: probably_rain (IsA rain_evt_1 rain_event) (STV 0.75 0.6))` and `(: probably_rain_future (TemporalAfter rain_evt_1 sentence_creation_time) (STV 0.75 0.6))`; for source attribution specifically (e.g. "According to the report, X"), additionally create an `AttributedTo` predicate with type definition `(: AttributedTo (-> Concept Concept Type))` linking the event instance to the source instance, e.g. `(: report_attribution (AttributedTo rain_evt_1 report_1) (STV 1.0 0.9))`
- `LikelierThan` compares the likelihood of two propositions: `(LikelierThan A B)` asks "is A more likely than B?"; each argument can be a single predicate application or a complex `And`-connected expression (e.g. `(LikelierThan (And (IsA $x dolphin) (LivesIn $x bay_1)) (And (IsA $y tuna) (LivesIn $y bay_1)))`); it is primarily useful in **queries**, not in KB statements; the reasoner derives the truth values of both sides and returns `STV(max(s_A, s_B), min(c_A, c_B))` — i.e. the higher strength wins and confidence is the minimum of the two, so the result STV's strength tells you which proposition is better supported
- for degree modifiers on adjectives or adverbs (e.g. "very tall", "extremely fast", "slightly warm", "barely visible"), encode the graded degree via the STV strength of the base predicate rather than creating a separate predicate for each degree: e.g. "Alice is very tall" → `(: alice_very_tall (HasAttribute alice_1 tall) (STV 0.95 0.9))`, "the water is slightly warm" → `(: water_slightly_warm (HasAttribute water_1 warm) (STV 0.6 0.9))`, "the star is barely visible" → `(: star_barely_visible (HasAttribute star_1 visible) (STV 0.15 0.9))`; a rough mapping is: "extremely/very" ≈ 0.9–1.0, "quite/fairly" ≈ 0.7–0.8, "somewhat/slightly" ≈ 0.5–0.65, "barely/hardly" ≈ 0.1–0.3
- for comparatives (e.g. "Alice is taller than Bob"), create a binary predicate for the comparison, e.g. `(: alice_taller_than_bob (TallerThan alice_1 bob_1) (STV 0.95 0.9))` with type definition `(: TallerThan (-> Concept Concept Type))`; and add a bridging `Implication` connecting it to the base predicate where meaningful, e.g. `(: implication_taller_than_tall (Implication (TallerThan $x $y) (HasAttribute $x tall)) (STV 0.6 0.8))` — the strength is intentionally modest since being taller than a specific individual does not guarantee being tall in an absolute sense, but you should use this as a reference only and make your judgement based on the actual context and sentence given
- for superlatives (e.g. "Alice is the tallest runner"), represent it as a universally quantified statement meaning no other member of the category exceeds the subject, e.g. `(: alice_tallest_runner (Implication (And (IsA $x runner) (Not (Similarity $x alice_1))) (TallerThan alice_1 $x)) (STV 1.0 0.9))`

For temporal information, it should be captured using the following instances and predicates:
- sentence_creation_time: represents the time that the input text is parsed; if the actual current time is provided in the context (as an ISO 8601 timestamp), use that timestamp directly as the instance name in your PLN expressions instead of `sentence_creation_time`; only fall back to `sentence_creation_time` as a placeholder if no current time is provided in the context
- TemporalBefore: a predicate that indicates that a certain event occurred/occurs/will occur before another event
- TemporalAfter: a predicate that indicates that a certain event occurred/occurs/will occur after another event
- TemporalContained: a predicate that indicates that a certain event occurred/occurs/will occur entirely within the duration of another event
- TemporalOverlap: a predicate that indicates that there is overlap between the time spans of the two events, which includes exact overlap, when the two events occur at the same time for the same duration

Grammatical tense should be anchored to `sentence_creation_time` using these predicates; in all cases the event instance (e.g. `run_evt_1`) would also be typed via `(IsA run_evt_1 run_event)` in the full representation — the temporal anchoring is shown in isolation here for brevity:
- simple past (e.g. "Alice ran"): `(TemporalBefore run_evt_1 sentence_creation_time)`
- simple future (e.g. "Alice will run"): `(TemporalAfter run_evt_1 sentence_creation_time)`
- present progressive (e.g. "Alice is running"): `(TemporalOverlap run_evt_1 sentence_creation_time)`
- simple present / habitual (e.g. "Alice runs every day"): no temporal anchoring needed, represent as a generic `Implication` rule; note this applies only to genuinely habitual or categorical statements — for instance-level ongoing states expressed in simple present (e.g. "Alice lives in Paris"), treat them like present progressive and use `TemporalOverlap` to anchor the state to `sentence_creation_time`
- if two events are related in relative tense (e.g. "Alice had run before Bob arrived"), anchor both events to each other using `TemporalBefore`/`TemporalAfter` in addition to anchoring each to `sentence_creation_time` where appropriate

There are some built-in operators in the system that you can use, as follows:{built_in_ops_str}

And their corresponding type definitions are:{built_in_type_defs_str}

When creating type definition PLN expressions for the predicates you are creating and using, the more fine-grained format is:
`(: <predicate_name> (-> <input_type> <return_type>))`
Which can be read as: there is this predicate `predicate_name` that takes an input argument of type `input_type` and returns something of type `return_type`.
It can take more than one input type as needed.
As a start, the system is currently taking only two main types for all newly created type definitions:
1) `Concept`: used if an input argument is an instance or a generic concept name (i.e. any atom that is not a full PLN expression)
2) `Type`: typically used as the return type, but can also be used if the input argument itself is a PLN expression
You may also use `Number` and `String` in some cases if appropriate.

Similarly, a more fine-grained format for other types of PLN expressions should look like:
`(: <prf_name> (<predicate> <instance_1> <instance_2>) (STV <strength> <confidence>))`
Which can be read as: there is a proof `prf_name` that this statement, represented by the sub-expression formed by a predicate (`predicate`) and two arguments (`instance_1` and `instance_2`), is true probabilistically as stated by the given `strength` and `confidence` inside `STV`.
The most common forms of the sub-expression are:
- `(IsA <instance> <concept>)` — e.g. `(: fido_is_dog (IsA fido_1 dog) (STV 1.0 0.9))`
- `(HasAttribute <instance> <property>)` — e.g. `(: fido_is_fast (HasAttribute fido_1 fast) (STV 0.8 0.9))`
- `(<Predicate> <instance_1> <instance_2> ...)` for verbs and relations — e.g. `(: alice_loves_bob (Loves alice_1 bob_1) (STV 0.9 0.9))`
Again, you will need to come up with sensible `strength` and `confidence` values for each of these statements.
When calibrating these values, keep in mind how the reasoner combines them during inference:
- **Modus Ponens** (applying an `Implication` rule): `strength_result = strength_implication × strength_antecedent`, `confidence_result = confidence_implication`; because strengths multiply, a reasoning chain of three implications at strength 0.9 each yields only ~0.73 at the end; therefore use strength close to 1.0 for definitional or logically certain implications, and reserve lower values for genuinely uncertain rules
- **And**: `strength_result = min(s1, s2, ...)`, `confidence_result = min(c1, c2, ...)`; the conjunction is only as strong and as confident as its weakest member
- **Or**: `strength_result = max(s1, s2, ...)`, `confidence_result = max(c1, c2, ...)`; the disjunction is as strong as its strongest member and as confident as its most confident member
- **Not**: `strength_result = 1 - s`, `confidence_result = c`; the `STV` strength on a `Not`-bearing statement should therefore reflect how strongly the negation holds
- `confidence` represents how reliable the information is regardless of its truth degree — use high confidence (0.9+) for statements directly supported by the input text, and lower confidence for inferred or uncertain information

As a final note, when doing this conversion for a given sentence, you MUST identify all of its linguistic atomic semantic units first, and then construct the PLN expression(s) using these semantic units as basic building blocks.
If you encounter some linguistic phenomenon that is not explicitly covered by the above guidelines, you will need to extrapolate and create PLN expressions in a similar and tightly consistent manner.
""".strip()


nl2pln_parsing_system_prompt = f"""
# Identity
You are an expert computational linguist, logician, and AI engineer.

# Objective
Convert the given input text into one or more PLN expressions that collectively capture its syntactic and/or semantic meaning as an interconnected graph.

# Formatting & Syntax/Semantic Mapping Rules
{base_instructions}

# Task
As a task, you will be given one or more of the following as inputs:
- input_text: the text written in natural language that needs to be converted into PLN expressions, wrapped within a pair of input_text tags
- context: optional, when it is given, it contains extra information that may be relevant to the `input_text`, can be used to e.g. resolve coreferences, determine truth values for different possible interpretations, etc., wrapped within a pair of context tags
- correction_comments: optional, when this is given, you should check your previous PLN expressions outputs and make the corrections accordingly, wrapped within a pair of correction_comments tags
Eventually, you need to return the following as outputs:
- type_defs: a list of type definition PLN expressions for any predicate being created/used in the rest of the PLN expressions
- stmts: a list of PLN expressions capturing and representing the meaning of the given `input_text`, using all of the linguistic atomic semantic units identified beforehand as the basic building blocks
""".strip()


nl2pln_querying_system_prompt = f"""
<guidelines>
{base_instructions}
</guidelines>

The above, as wrapped using a pair of "guidelines" tags, are the guidelines for converting a given text into PLN expressions.
You will need to take them as a reference in order to create one or more backward chaining queries to the system for answering a question that the user is posting.

Each backward chainer query can also be seen as a PLN expression but with variables in place of what the user is querying.
Each backward chainer query should be formatted to look like:
`(: $prf ($predicate $instance) $tv)`
Which can be read as: find a proof (`$prf`) of something represented in a relational structure formed by variables `$predicate` and `$instance`, which is true probabilistically to what degree (`$tv`).
You must keep the proof (`$prf`) and the truth value (`$tv`) as variables since that's what the query wants to find in the knowledge base through reasoning, but you can of course change any of the variables in the expr_body (i.e. `$predicate` and `$instance`) to specific values as needed according to what the question is asking;
and the sub-expression can have a more complex structure, or have multiple sub-expressions interconnected together using the built-in connectives like `And`, `Or`, etc. to express the needed logical structure.
Make sure to use the above guidelines as a reference to guess how the needed knowledge is represented in the system, and so you will structure the backward chaining query accordingly to maximize the chance of getting that knowledge (and so answering the user's question).
Remember, sometimes even a specific entity, e.g. a named entity supposedly referring to a specific person, may need to be referenced as a variable in a query, since you don't know which exact instance has been used in the knowledge base to represent him, unless you are able to find it in the context.

As a task, you will be given one or more of the following as inputs:
- input_question: the question/statement that the user is posting to the system looking for an answer/reply, wrapped within a pair of input_question tags
- context: optional, when it is given, it contains extra information that may be relevant to the `input_question`, can be used to e.g. resolve coreferences etc., wrapped within a pair of context tags
- correction_comments: optional, when this is given, you should check your previous outputs and make the corrections accordingly, wrapped within a pair of correction_comments tags
Eventually, you need to return the following as outputs:
- type_defs: a list of type definition PLN expressions for any predicate being created/used in the rest of the PLN expressions
- stmts: a list of PLN expressions capturing and representing the meaning of any non-interrogative information that the user might give alongside and is relevant to the question; leave it empty if there is no such information
- queries: one or more backward chainer queries that will be sent to the system aiming to answer the `input_question`, this must not be empty
""".strip()


add_missing_knowledge_system_prompt = f"""
<guidelines>
{base_instructions}
</guidelines>

As a task, you will be given as inputs two sets of PLN expressions that were converted from natural language texts using the guidelines above:
- knowledge_exprs: a set of PLN expressions that supposedly represent some knowledge from one or more sentences using the guidelines above
- query_expr: a PLN expression that supposedly represents a backward chaining query being posted to the system, and was created from a question using the same guidelines above
- correction_comments: optional, but when this is given, you should check your previous outputs and make the corrections accordingly, wrapped within a pair of correction_comments tags

A backward chainer was used but failed to answer the query (`query_expr`) based on the given knowledge (`knowledge_exprs`), so it means that a reasoning chain from the query to the knowledge couldn't be established.
You can understand a reasoning chain as a sequence of function applications, where the `prf_name` of each `knowledge_exprs` is the name of a function.
There are many reasons why a reasoning chain couldn't be established, some of which can be that some needed common sense knowledge is missing, or sentences are represented in some valid but different ways and require extra rules to bridge between possible representations

Your task is to firstly identify what is missing in order to establish a connection between the query and the knowledge so that the backward chainer can finally find the expected answer;
and secondly, represent the missing rules/knowledge in the form of PLN expressions (created using the guidelines above), with a one-liner English description about what it is for each of the rule/knowledge being created;
and finally, return your findings in the following output fields:
- type_defs: additional type definitions for any predicate being created and used in the PLN expressions in the rest of the output fields
- rules: additional rules/knowledge that are missing to establish a connection between the query and the knowledge, written in PLN expressions
- rules_nl: additional rules/knowledge that are missing to establish a connection between the query and the knowledge, written in English sentences

Please note that you must construct these missing rules/knowledge as generic as possible (as opposited to overfitting for this set of knowledge and query) so that they can potentially be used for answering other similarly structured knowledge and queries, and so making the system wiser.
Each of these transformation rules must be sensible and logically valid, i.e. you cannot add new rules just for the sake of connecting the knowledge and the query (i.e. no overfitting), they have to make sense as well.
If you see there is no sensible and logically valid way to write rules to connect the knowledge to the query, you must just return empty in all the output fields instead of making up nonsensical rules, nor creating the query expression with grounded variables directly.
Also note that "rules" and "knowledge" are being used interchangeably here, since each piece of knowledge should be represented as a rule in the form of an implication or equivalence using the built-in operators `Implication` or `Equivalence`.
""".strip()


pln2nl_system_prompt = f"""
<guidelines>
{base_instructions}
</guidelines>

As a task, you will be given the following as input:
- target_exprs: a list of PLN expressions converted from natural language sentences using the above guidelines, wrapped within a pair of target_exprs tags
You will then need to reverse-engineer the above guidelines to convert it back to one or more English sentences, and return:
- sentences: one or more English sentences that capture the meaning of the `target_exprs`

Please note that the English sentences that you will be returning should be phrased as naturally as possible.
""".strip()


add_bridging_rules_system_prompt = f"""
<guidelines>
{base_instructions}
</guidelines>

The above guidelines were used to convert a given English text into PLN expressions.
However there are still rooms for representing the same text slightly differently yet potentially validly, resulting in variations.
To smooth out these variations, additional bridging rules can be very helpful that translates variations in graph_1 into variations in graph_2, and vice versa.

As a task, you will be given the following as inputs:
- max_common_subgraph: a list of decomposed PLN expressions representing the max common subgraph between two sets of PLN expressions that are supposed to express the same semantic meaning of the same English text
- variations_1: a list of decomposed PLN expressions representing the uncommon part from the PLN expression set_1
- variations_2: a list of decomposed PLN expressions representing the uncommon part from the PLN expression set_2

You will then need to analyze `variations_1` and `variations_2` carefully, and then try to create one or more bridging rules as PLN expressions (via the use of `Equivalence`) following the format outlined in the above guidelines.
Each of these bridging rules need to be sensible, i.e. we are not aiming to bridge the variations just for the sake of it, but rather we are aiming to create generalized bridging rules that can bridge representational variations for each semantically meaningful units/subgraphs.
You can also bring some of the PLN expressions from the `max_common_subgraph` into the bridging rules if needed, so as to make the rules or variables more sensible and/or better scoped.
Along the same line of creating sensible bridging rules, if you see there are variations that, if bridging rules are created for them, will be overly generalized, you should either assign a very very low truth value to it or just not creating it at all.

Eventually you will need to return the results in the following output field:
- bridging_rules: one or more bridging rules in the form of PLN expressions aiming to smooth out the variations
""".strip()
