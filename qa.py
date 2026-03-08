import json
from datetime import datetime, timezone, timedelta

from pipelines import *

while True:
    mode = input("Enter either:\n- '1' to parse a sentence as the KB\n- '2' to read a KB from a file\n>> ")
    if mode == "1":
        sentence = input("Enter a sentence: ")
        sent_result = nl2pln(sentence, mode="parsing")
        if sent_result is None:
            print(f"Failed parsing the sentence: '{sentence}', please try another one.")
            continue
        else:
            type_defs, stmts, _, extra_exprs, _ = sent_result
            break
    elif mode == "2":
        kb_filename = input("Enter the file name in path: ")

    if mode == "2":
        with open(f"{kb_filename}", "r") as fp:
            data = json.load(fp)
            if isinstance(data, list):
                while True:
                    max_idx = len(data) - 1
                    sentence_idx = int(input(f"Enter a sentence index (0-{max_idx}): "))
                    if sentence_idx < 0 or sentence_idx > max_idx:
                        print("Invalid sentence index!")
                        continue
                    else:
                        break
                sentence, type_defs, stmts, extra_exprs = [(s["sentence"], s["type_defs"], s["stmts"], s["extra_exprs"]) for s in data if s["sentence_idx"] == sentence_idx][0]
            elif isinstance(data, dict):
                sentence, type_defs, stmts, extra_exprs = data["sentence"], data["type_defs"], data["stmts"], data["extra_exprs"]
            print(f"Successfully loaded!\n```\nsentence = \"{sentence}\"\ntype_defs = {type_defs}\nstmts = {stmts}\nextra_exprs = {extra_exprs}\n```\n")
            break

while True:
    qcmd = input("\n====== ['/exit' to exit | '/save' to save] ======\n\nEnter a question: ")

    if qcmd == "/exit":
        print("... exiting")
        break
    elif qcmd == "/save":
        current_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d-%H-%M-%S")
        qa_out_file = f"qa_{current_time}.json"
        print(f"... saving to {qa_out_file}\n")
        output_to_json_file({
                "sentence": sentence,
                "type_defs": type_defs,
                "stmts": stmts,
                "question": question,
                "q_type_defs": q_type_defs,
                "q_stmts": q_stmts,
                "q_queries": q_queries,
                "q_extra_exprs": q_extra_exprs,
                "a_type_defs": a_type_defs,
                "a_rules_nl": a_rules_nl,
                "a_rules": a_rules,
                "chaining_result": chaining_result
            },
            qa_out_file)
        continue

    question = qcmd
    ques_result = nl2pln(question, mode="querying")
    if ques_result is None:
        print(f"Failed parsing the question: '{question}', please try another one.")
        continue
    else:
        q_type_defs, q_stmts, q_queries, q_extra_exprs, _ = ques_result

        for q_idx, query in enumerate(q_queries):
            print(f"... handling query ({q_idx+1} of {len(q_queries)})")
            qa_result = assisted_qa(type_defs + q_type_defs, stmts + q_stmts + q_extra_exprs, query, kb_nl=sentence, query_nl=question)
            chaining_result, a_type_defs, a_rules, a_rules_nl = qa_result

            if chaining_result:
                print(f"ANSWER FOUND!!\n\n... needed to add:\na_type_defs = {a_type_defs}\na_rules = {a_rules}\na_rules_nl = {a_rules_nl}\n")
                # TODO: maybe gather the incoming set of the instances involved as well
                print(f"... constructing the answer")
                answer = pln2nl(chaining_result)
                print(f"Answer: {answer}\n")
            else:
                print(f"ANSWER NOT FOUND!!\n\n... tried adding:\na_type_defs = {a_type_defs}\na_rules = {a_rules}\na_rules_nl = {a_rules_nl}\n")
