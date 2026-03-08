from pydantic import BaseModel
from textwrap import dedent

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

def create_nl2pln_parsing_prompt(text, context=[]):
    context_str = ""
    if context:
        for c in context:
            context_str += f"# {c['sentence']}\n{c['stmts']}\n\n"
    return dedent(f"""
        <context>
        {context_str.strip()}
        </context>

        <input_text>
        {text}
        </input_text>
        """).strip()

def create_nl2pln_querying_prompt(text, context=[]):
    context_str = ""
    if context:
        for c in context:
            context_str += f"# {c['sentence']}\n{c['stmts']}\n\n"
    return dedent(f"""
        <context>
        {context_str.strip()}
        </context>

        <input_question>
        {text}
        </input_question>
        """).strip()

def create_nl2pln_correction_prompt(correction):
    return dedent(f"""
        <correction_comments>
        {correction}
        </correction_comments>
        """).strip()

def create_missing_exprs_prompt(k_exprs, q_expr):
    return dedent(f"""
        <knowledge_exprs>
        {k_exprs}
        </knowledge_exprs>

        <query_expr>
        {q_expr}
        </query_expr>
        """).strip()

def create_pln2nl_prompt(target_exprs):
    return dedent(f"""
        <target_exprs>
        {target_exprs}
        </target_exprs>
        """).strip()

def create_bridging_rules_prompt(max_common_subgraph, variations_1, variations_2):
    return dedent(f"""
        <max_common_subgraph>
        {max_common_subgraph}
        </max_common_subgraph>

        <variations_1>
        {variations_1}
        </variations_1>

        <variations_2>
        {variations_2}
        </variations_2>
        """).strip()


base_instructions = f"""
Here are some fundamental guidelines you need to follow when converting a natural language sentence into a kind of meaning representation that is friendly to a reasoner called PLN:
- one or more expressions should be created as a result of this conversion, which collectively capture the syntactic and/or semantic meaning of the given sentence, and each of these expressions is referred to as a "PLN expression"
- all PLN expressions must be expressed in the format of either `(: <expr_name> <expr_body>)` or `(: <expr_name> <expr_body> <truth_value>)`
- the `expr_name` (aka `proof_name`) can just be any arbitrary string written in snake_case, but by convention it's preferable to somewhat reflect the meaning of the `expr_body` in some shorthand form, and it must be uniquely identifiable globally
- ensure the `expr_name` is strictly an identifier and remains entirely excluded from the `expr_body`
- for PLN expressions that are type definitions, they should be in the format of `(: <expr_name> <expr_body>)`
- for PLN expressions that are not type definitions, they should be in the format of `(: <expr_name> <expr_body> <truth_value>)`, with the `truth_value` provided using the built-in constructor `STV`, which expects a strength and a confidence value
- to denote an instance, come up with an arbitrary name for it which should be written in snake_case; how it's named doesn't matter, but it must be uniquely identifiable globally, and by convention it's preferable to have a name that is identical or similar to the word or phrase it is referring to in the given sentence
- to denote a variable, come up with an arbitrary name written in snake_case and prefix it with a '$' symbol; how it's named doesn't matter, but it must be uniquely identifiable within its scope
- to denote a predicate, it's preferable to name it using the original word or phrase in the given sentence without adding any pre- or post-fix, and it must be written in UpperCamelCase
- an instance can be understood as the existence of a unique or specific (i.e. non-generic) entity/concept in the given sentence, which will be globally scoped in the knowledge base and so can be seen and used by other PLN expressions in the same knowledge base
- no instance can exist on its own within the `expr_body`, each instance needs to be associated with a predicate, and so in this representation essentially every linguistic term (e.g. nouns, pronouns, verbs, adjectives, adverbs, or even prepositions etc.) can be considered as an instance of a predicate, arranged and grouped together with the built-in logical operators or other predicates, constituting the `expr_body`
- similarly no predicate can exist on its own without associating with an instance or a variable, so it should look like: `(<predicate> <instance or variable>)`, which can be understood as "this instance or variable has a property/attribute denoted by the `predicate`"
- it's highly preferable to have predicates that are as simple as possible, e.g. a single-word predicate is much more preferable than a multi-word predicate, so you must try to construct the PLN expressions using predicates that are as atomic and elementary as possible
- if a multi-word predicate is really neccessary to be created, you must also unpack and elaborate its meaning by creating additional `Implication` expression(s) that shows how it can be created by simplar predicates, e.g. `(Implication (And (<predicate_x> $a $b) (<predicate_y> $c)) (<predicate_xy> $a $b $c))`, or `(Implication (And (<predicate_x> $a) (<predicate_y> $a)) (<predicate_xy> $a))`, or similarly as needed
- you must create a type definition for every new predicate that you are creating and using
- you should create an instance for each specific (i.e. non-generic) entity/concept that exists in the input sentence; if it's a pronoun and you know what/who it's referring to, you can use the built-in operator `Similarity` to equate the instance of the pronoun and the instance of that particular entity/concept with a sensible truth value attached
- for quantification, it's represented using the built-in quantifier `Implication`, and reflecting the level of quantification via `STV`, e.g. 'all' should have a relatively high truth value close to 1.0 while 'none' should have a relatively low truth value close to 0.0, and fuzzy quantifiers like 'most', 'many', 'some', 'a few', etc... should have a truth value that lies somewhat in between
- typically `Implication` is used to express "IsA" or if-then relationship with generic entities/concepts involved
- variables defined/used on the left hand side of an `Implication` is considered scoped within that `Implication` (and everything nested within it if any), while variables used/existed only on the right hand side of an `Implication` is considered unscoped (or globally scoped in the knowledge base)
- if there is more than one PLN expression involved in representing the meaning of the given text, these expressions must be connected together, forming something like an interconnected graph
- if the given sentence can be interpreted/understood in more than one way, e.g. literal vs idiomatic, literal vs metaphoric, and/or coreferences being resolved differently, etc., you should capture all these possible interpretations in the PLN expressions as well, with a truth value reflecting how likely each interpretation is based on your judgement on the given information
- for quantities you can create and make use of a `Quantity` predicate with type definition `(: Quantity (-> Concept Number Type))`, then e.g. "3 apples" can be represented as `(And (Apples apples) (Quantity apples 3))`, and similarly for named entities e.g. `(: Name (-> Concept String Type))`, or units e.g. `(: Unit (-> Concept String Type)), or anything of this nature
- for possession and relationships e.g. "Ben's cat" or "my sister", etc., you can create and make use of a `Possess` predicate with type definition `(: Possess (-> Concept Concept Type))`
- if you decide to represent an action using the event-based (e.g., Neo-Davidsonian) style because it has complex modifiers (e.g., tools, locations), you MUST ALSO generate a generic bridging `Implication` that maps this specific event structure back to its simple, core multi-arity format so as to ensure the knowledge base can connect complex events to simple queries, for example when parsing "Bob cut the apple with a knife", you can generate the specific event representation like `(And (CutEvent cut_evt) (Subject cut_evt bob) (Object cut_evt apple) (Instrument cut_evt knife))`, and because you used the event style for "Cut", you MUST also output the bridging rule for it: `(: bridge_cut_event (Implication (And (CutEvent $e) (Subject $e $s) (Object $e $o)) (Cut $s $o)) (STV 1.0 1.0))`
- for modal verbs (e.g., could, might, should), treat them as higher-order predicates that take a proposition as their argument e.g., having a type definition of `(: <ModalName> (-> Type Type))`
- if applying a modal to a multi-arity predicate, wrap the entire expression: `(Could (Cut bob_5 apple_2))`; if applying a modal to a Neo-Davidsonian event, apply it to the existence of the event: `(Could (Cut cut_event))`
- when encountering a "collective AND" where multiple entities perform an action together rather than independently (e.g., "Alice and Bob lifted the piano together"), DO NOT use the logical `And` operator to duplicate the action but instead create an instance representing the group (e.g., `group_ab`), and use a `Member` predicate with type definition `(: Member (-> Concept Concept Type))` to assign the individual entities to that group (e.g., `(And (Member alice_1 group_ab) (Member bob_1 group_ab) (Lift group_ab piano_1))`)

For temporal information, it should be captured using the following instances and predicates:
- sentence_creation_time: a placeholder instance indicating the time that the input text is parsed, just use it as-is wherever needed in the PLN expressions and the system will post-process it accordingly
- TemporalBefore: a predicate that indicates that a certain event occurred/occurs/will occur before another event
- TemporalAfter: a predicate that indicates that a certain event occurred/occurs/will occur after another event
- TemporalContained: a predicate that indicates that a certain event occurred/occurs/will occur entirely within the duration of another event
- TemporalOverlap: a predicate that indicates that there is overlap between the time spans of the two events, which includes exact overlap, when the two events occur at the same time for the same duration

There are some built-in operators in the system that you can use, as follows:{built_in_ops_str}

And their corresponding type definitions are:{built_in_type_defs_str}

When creating type definition PLN expressions for the predicates you are creating and using, the more fine-grained format is:
`(: <predicate_name> (-> <input_type> <return_type>))`
Which can be read as: there is this predicate `predicate_name` that takes an input argument of type `input_type` and returns something of type `return_type`.
It can take more than one input type as needed.
As a start, the system is currently taking only two main types for all newly created type definitions:
1) `Concept`: used if an input argument is an instance
2) `Type`: typically used as the return type, but can also be used if the input argument itself is a PLN expression
You may also use `Number` and `String` in some cases if appropriate.

Similarly, a more fine-grained format for other types of PLN expressions should look like:
`(: <prf_name> (<predicate> <instance_1> <instance_2>) (STV <strength> <confidence>))`
Which can be read as: there is a proof `prf_name` that this statement, represented by the sub-expression formed by a predicate (`predicate`) and two arguments (`instance_1` and `instance_2`), is true probabilistically as stated by the given `strength` and `confidence` inside `STV`.
Again, the sub-expression can have fewer or more than two instances as needed, and you will need to come up with a sensible `strength` and `confidence` values for each of these statements.

As a final note, when doing this conversion for a given sentence, you MUST identify all of its linguistic atomic semantic units first, and then construct the PLN expression(s) using these semantic units as basic buildling blocks.
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
