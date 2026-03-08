import json
from datetime import datetime, timezone, timedelta

from pipelines import *
from vector_index import *

current_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d-%H-%M-%S")
sp_out_file = f"data/parsed/semantic_parsing_{current_time}.json"

previous_parses = []
all_outputs = []
failed_cases = []

filename = input("Enter the full name of the JSON file to be parsed: ")

with open(f"{filename}", "r") as fp:
    data = json.load(fp)

    if isinstance(data, list):
        sentences = [d["sentence"] for d in data]
    elif isinstance(data, dict):
        sentences = [data["sentence"]]

    data_len = len(sentences)
    max_idx = data_len - 1
    print(f"Successfully loaded {data_len} sentences!\n```\nsentences = {sentences}\n```\n")

start_idx = input("Start parsing from index (default is 0): ")
try:
    start_idx = min(int(start_idx), max_idx)
except Exception:
    start_idx = 0

end_idx = input(f"End parsing at index (default is {max_idx}): ")
try:
    end_idx = min(int(end_idx), max_idx)
except Exception:
    end_idx = max_idx

enable_context = True if input("Enable coreference resolution across sentences (Y/N): ").lower() == "y" else False

faiss_size = sum([index.ntotal for index in faiss_store.indices.values()])
if faiss_size and input(f"FAISS has {faiss_size} vectors, remove them (Y/N): ").lower() == "y":
    faiss_store.clear()

print(f"... going to parse sentences from idx {start_idx} to {end_idx}")
for i in range(start_idx, end_idx + 1):
    sentence = sentences[i]

    print(f"... looking at sentence (idx = {i}): {sentence}")

    # TODO: find a better way to identify what's a relevant context for the sentence, now it's only naively passing the previous N sentences
    sent_result = nl2pln(sentence, mode="parsing", context=previous_parses[-5:])
    if sent_result is not None:
        type_defs, stmts, _, extra_exprs, sent_links = sent_result

        print(f"---\n\nSentence #{i}:\n{sentence}\n\ntype_defs:\n{type_defs}\n\nstmts:\n{stmts}\n\nextra_exprs:\n{extra_exprs}\n\n---\n")
        all_outputs.append({
            "sentence_idx": i,
            "sentence": sentence,
            "type_defs": type_defs,
            "stmts": stmts,
            "extra_exprs": extra_exprs,
        })
        output_to_json_file(all_outputs, sp_out_file)
        if enable_context:
            previous_parses.append({
                "sentence": sentence,
                "stmts": stmts
            })
    else:
        failed_cases.append(i)

if failed_cases:
    print(f"Failed to parse: {failed_cases}")

finish_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
print(f"FINISHED PARSING at {finish_time}")
