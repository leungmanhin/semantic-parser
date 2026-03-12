import json
import os
from datetime import datetime, timezone, timedelta

from pipelines import *
from vector_index import *

FAISS_DIR = "data/faiss"

current_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d-%H-%M-%S")
sp_out_file = f"data/parsed/semantic_parsing_{current_time}.json"

previous_parses = []
all_outputs = []
failed_cases = []

if os.path.exists(os.path.join(FAISS_DIR, "config.json")):
    if input(f"Existing FAISS store found in '{FAISS_DIR}', load it? (Y/N): ").lower() == "y":
        faiss_store = SemanticArityIndex.load(FAISS_DIR)
        sentence_parses_store = SentenceParsesIndex.load(FAISS_DIR)

filename = input("Enter the full name of the JSON file to be parsed: ")

with open(f"{filename}", "r") as fp:
    data = json.load(fp)

    if isinstance(data, list) and data and "sentences" in data[0]:
        sentences = [s for entry in data for s in entry["sentences"]]
    elif isinstance(data, list):
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

print(f"... going to parse sentences from idx {start_idx} to {end_idx}")
for i in range(start_idx, end_idx + 1):
    sentence = sentences[i]

    print(f"... looking at sentence (idx = {i}): {sentence}")

    # TODO: find a better way to identify what's a relevant context for the sentence, now it's only naively passing the previous N sentences
    prev = previous_parses[-5:]
    sent_result = nl2pln(sentence, mode="parsing", context=[{"title": "Previously parsed sentence-stmts pairs that may be relevant to the input_text", "entries": prev}] if prev else [])
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
        faiss_store.save(FAISS_DIR)
        sentence_parses_store.save(FAISS_DIR)
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
