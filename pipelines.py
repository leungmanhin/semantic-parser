import re

from checker_functions import *
from chaining import chaining
from extra_exprs import *
from graphs import *
from llm import *
from prompts import *
from utils import *
from vector_index import *

def format_check_correct(llm_outputs, chat_history, output_format, max_back_forth=10, related_exprs={}):
    while True:
        attempts = int((len(chat_history)-1)/2)
        print(f"[attempts = {attempts}]")

        type_defs = llm_outputs["type_defs"]
        stmts = llm_outputs["stmts"] if "stmts" in llm_outputs else llm_outputs["rules"]
        queries = llm_outputs["queries"] if "queries" in llm_outputs else []

        if attempts > max_back_forth:
            print(f"Maximum back-and-forth's ({max_back_forth} times) with the LLM has reached!")
            return None

        print(f"Format checking for:\n```\ntype_defs = {type_defs}\nstmts = {stmts}\nqueries = {queries}\n```\n")

        type_def_check_pass = True
        for type_def in type_defs:
            expr_check_result, expr_check_exception = expr_format_check(type_def)
            e = "" if expr_check_exception == None else f"{expr_check_exception}".strip()
            if not (expr_check_result and type_def_check(type_def)):
                print(f"... retrying type_def_check for type_def '{type_def}'\n")
                llm_outputs = to_openrouter(create_nl2pln_correction_prompt(f"One of your type_defs ('{type_def}') doesn't pass the format check" + (f" with an exception '{e}', " if e else ", ") + "please make the correction and regenerate all the output fields."), output_format=output_format, history=chat_history)
                type_def_check_pass = False
                break
        if not type_def_check_pass:
            continue

        stmts_check_pass = True
        for stmt in stmts:
            stmt_check_result, stmt_check_exception = stmt_format_check(stmt)
            e = "" if stmt_check_exception == None else f"{stmt_check_exception}".strip()
            if not stmt_check_result:
                print(f"... retrying stmt_format_check for stmt '{stmt}'\n")
                llm_outputs = to_openrouter(create_nl2pln_correction_prompt(f"One of your stmts ('{stmt}') doesn't pass the format check" + (f" with an exception '{e}', " if e else ", ") + "please make the correction and regenerate all the output fields."), output_format=output_format, history=chat_history)
                stmts_check_pass = False
                break
        if not stmts_check_pass:
            continue

        query_check_pass = True
        for query in queries:
            query_check_result, query_check_exception = query_format_check_1(query)
            e = "" if query_check_exception == None else f"{query_check_exception}".strip()
            if not query_check_result:
                print(f"... retrying query_format_check_1 for query '{query}'\n")
                llm_outputs = to_openrouter(create_nl2pln_correction_prompt(f"One of your queries ('{query}') doesn't pass the format check" + (f" with an exception '{e}', " if e else ", ") + "please make the correction and regenerate all the output fields."), output_format=output_format, history=chat_history)
                query_check_pass = False
                break
            if not query_format_check_2(query):
                print(f"... retrying query_format_check_2 for query '{query}'\n")
                llm_outputs = to_openrouter(create_nl2pln_correction_prompt(f"Make sure the proof name and the truth value of your query '{query}' are variables in order to make it a valid query. Please make the improvement and regenerate all the output fields."), output_format=output_format, history=chat_history)
                query_check_pass = False
                break
        if not query_check_pass:
            continue

        # # TODO: temporarily disable type-checking to reduce LLM calls as we're not strictly using it at the moment
        # metta_type_check_pass = True
        # for expr in stmts + queries:
        #     check_result, check_exception = metta_type_check(type_defs + built_in_type_defs, expr)
        #     e = "" if check_exception == None else f"{check_exception}".strip()
        #     if not check_result:
        #         print(f"... retrying metta_type_check for: {expr} | {type_defs}\n")
        #         llm_outputs = to_openrouter(create_nl2pln_correction_prompt(f"One of your PLN expressions ('{expr}') doesn't pass type checking in the system based on your type_defs ({type_defs})" + (f" with an exception '{e}', " if e else ", ") + "please make the correction and regenerate all the output fields."), output_format=output_format, history=chat_history)
        #         metta_type_check_pass = False
        #         break
        # if not metta_type_check_pass:
        #     continue

        rtn = unused_preds_check(
            type_defs + (related_exprs["type_defs"] if related_exprs else []),
            stmts + queries + ((related_exprs["stmts"] + related_exprs["queries"]) if related_exprs else [])
        )
        if not rtn[0]:
            print(f"... retrying for unused_preds: {rtn[1]}\n")
            llm_outputs = to_openrouter(create_nl2pln_correction_prompt(f"You have defined one or more predicates but left unused:\n{rtn[1]}\n\nPlease make the correction and regenerate all the output fields."), output_format=output_format, history=chat_history)
            continue

        rtn = undefined_preds_check(
            type_defs + (related_exprs["type_defs"] if related_exprs else []),
            stmts + queries + ((related_exprs["stmts"] + related_exprs["queries"]) if related_exprs else [])
        )
        if not rtn[0]:
            print(f"... retrying for undefined_preds: {rtn[1]}\n")
            llm_outputs = to_openrouter(create_nl2pln_correction_prompt(f"You have used one or more predicates that are not defined:\n{rtn[1]}\n\nPlease make the correction and regenerate all the output fields."), output_format=output_format, history=chat_history)
            continue

        if not connectivity_check(stmts + (related_exprs["stmts"] if related_exprs else [])):
            print(f"... retrying for connectivity_check for: {stmts}\n")
            llm_outputs = to_openrouter(create_nl2pln_correction_prompt(f"Some of your 'stmts' are disconnected from the rest. Please make the correction and regenerate all the output fields."), output_format=output_format, history=chat_history)
            continue

        print(f"PASSED FORMAT CHECK!!\n")
        break

    return (type_defs, stmts, queries)


# mode = "parsing" | "querying"
def nl2pln(sentence, context=[], mode="parsing", max_back_forth=10):
    system_prompt = nl2pln_querying_system_prompt if mode == "querying" else nl2pln_parsing_system_prompt
    output_format = PLNQueryExprs if mode == "querying" else PLNExprs

    print(f'\n... parsing "{sentence}" | context: {context}')

    # reset chat_history for each input sentence
    chat_history = [{
        "role": "system",
        "content": system_prompt
    }]

    # will try to resolve cross-sentence coreferences if a context is given
    llm_outputs = to_openrouter(create_nl2pln_parsing_prompt(sentence, context), output_format=output_format, history=chat_history)

    if mode == "querying":
        while (len(chat_history)-1)/2 > max_back_forth:
            if (not llm_outputs["queries"]):
                llm_outputs = to_openrouter(create_nl2pln_correction_prompt(f"Make sure you structure one or more queries from the `input_question` and return it in the 'queries' output field. Please make the correction and regenerate all the output fields."), output_format=output_format, history=chat_history)
                continue
            break

    type_defs, stmts, queries = format_check_correct(llm_outputs, chat_history, output_format, max_back_forth=max_back_forth)

    sent_links = [f'(SentenceLink {re.search(r'\(: (.+?) \(.+\)\)', re.sub(r'\n\s*', ' ', stmt)).group(1)} "{sentence}")' for stmt in stmts]

    print(f"### {sentence} ###\n```", *(type_defs + stmts + queries), "```\n", sep="\n")

    print(f"... creating Equivalence with existing predicates")
    extra_exprs = []
    pred_arity_list = extract_predicates_with_arity(stmts + queries)
    for pa in pred_arity_list:
        pred = pa[0]
        arity = pa[1]
        similar_preds = faiss_store.search_and_store(pred, arity)["matches"]
        for s_pred in similar_preds:
            eq_expr = generate_equivalence_expr(pred, s_pred[0], arity, s_pred[1])
            extra_exprs.append(eq_expr)
            print(f'... generated: "{eq_expr}"')
            # # XXX TODO: just a quick hack for testing since Equivalence doesn't work in chaining
            # eq_expr1 = generate_equivalence_expr(pred, s_pred[0], arity, s_pred[1]).replace("Equivalence", "Implication")
            # eq_expr2 = generate_equivalence_expr(s_pred[0], pred, arity, s_pred[1]).replace("Equivalence", "Implication")
            # extra_exprs.append(eq_expr1)
            # extra_exprs.append(eq_expr2)
            # print(f'... generated: "{eq_expr1} {eq_expr2}"')

    return (type_defs, stmts, queries, extra_exprs, sent_links)

def nl2pln_multi_runs(sentence, context=[], mode="parsing", max_back_forth=10, runs=3):
    all_type_defs = []
    all_stmts = []
    all_queries = []
    all_extra_exprs = []
    all_sent_links = []

    # TODO: parallelize
    for idx in range(0, runs):
        print(f"... nl2pln runs {idx+1} of {runs}")
        outputs = nl2pln(sentence, context=context, mode=mode, max_back_forth=max_back_forth)
        all_type_defs.append(outputs[0])
        all_stmts.append(outputs[1])
        all_queries.append(outputs[2])
        all_extra_exprs.append(outputs[3])
        all_sent_links.append(outputs[4])

    bridging_rules = generate_bridging_rules(all_stmts)
    all_extra_exprs += bridging_rules

    return (all_type_defs, all_stmts, all_queries, all_extra_exprs, all_sent_links)

def assisted_qa(all_type_defs, all_stmts, query, kb_nl="", query_nl="", max_back_forth=10, sibling_queries=[]):
    chat_history = [{
        "role": "system",
        "content": add_missing_knowledge_system_prompt
    }]

    a_type_defs = []
    a_rules = []
    a_rules_nl = ""

    while True:
        attempts = int((len(chat_history)-1)/2)
        all_type_defs = list(set(all_type_defs + a_type_defs))
        all_stmts = list(set(all_stmts + a_rules))

        chaining_result = chaining(all_type_defs + all_stmts, query)

        if chaining_result:
            return (chaining_result, a_type_defs, a_rules, a_rules_nl)
        # only 1 attempt will be made for now, until we find a way to know the failure is due to bad representations
        elif attempts >= 1:
            print(f"Failed to answer '{query}', skipping for now...")
            # for later debugging
            print_test_case(all_type_defs + all_stmts, query, kb_nl=kb_nl, query_nl=query_nl)
            break
        else:
            print(f"... trying to fill in missing pieces before retrying for '{query}'")
            llm_outputs = to_openrouter(create_missing_exprs_prompt(all_stmts, query), history=chat_history, output_format=AddPLNExprs)

            # # TODO: re-enable this to be more strict
            # format_check_result = format_check_correct(llm_outputs, chat_history, AddPLNExprs, max_back_forth=max_back_forth, related_exprs={"type_defs": all_type_defs, "stmts": all_stmts, "queries": sibling_queries})
            # if format_check_result:
            #     a_type_defs, a_rules, _ = format_check_result
            #     a_rules_nl = llm_outputs["rules_nl"]
            #     print(f"Newly proposed:\n```\na_type_defs = {a_type_defs}\na_rules = {a_rules}\na_rules_nl = {a_rules_nl}\n```\n")
            a_type_defs, a_rules, a_rules_nl = llm_outputs["type_defs"], llm_outputs["rules"], llm_outputs["rules_nl"]
            print(f"Newly proposed:\n```\na_type_defs = {a_type_defs}\na_rules = {a_rules}\na_rules_nl = {a_rules_nl}\n```\n")

    return ([], a_type_defs, a_rules, a_rules_nl)

def pln2nl(chaining_results):
    def extract_grounded_expr(text: str) -> str | None:
        try:
            content_start = text.find('(: ')
            if content_start == -1:
                return None
            content_start += 3
            items = []
            balance = 0
            item_start_index = 0
            for i, char in enumerate(text[content_start:], start=content_start):
                if char == '(':
                    if balance == 0:
                        item_start_index = i
                    balance += 1
                elif char == ')':
                    balance -= 1
                    if balance == 0:
                        items.append(text[item_start_index : i + 1])
                if balance < 0:
                    break
            if len(items) >= 2:
                return items
        except (IndexError, TypeError):
            return None
        return None

    chat_history = [{
        "role": "system",
        "content": pln2nl_system_prompt
    }]

    target_exprs = []
    for chaining_result in chaining_results:
        target_exprs.append(extract_grounded_expr(chaining_result)[-2])

    # remove duplicates if any
    target_exprs = list(set(target_exprs))

    llm_outputs = to_openrouter(create_pln2nl_prompt(target_exprs), history=chat_history, output_format=NLSents)
    sentences = llm_outputs["sentences"]

    return sentences
