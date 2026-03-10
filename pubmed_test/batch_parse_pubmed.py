import json
import os
import glob
from datetime import datetime, timezone, timedelta

from pipelines import *
from vector_index import *

model = "gpt-5.4"
effort = "none"

script_dir = os.path.dirname(os.path.abspath(__file__))
FAISS_DIR = "data/faiss"

all_outputs = []
failed_cases = []

if os.path.exists(os.path.join(FAISS_DIR, "config.json")):
    if input(f"Existing FAISS store found in '{FAISS_DIR}', load it? (Y/N): ").lower() == "y":
        faiss_store = SemanticArityIndex.load(FAISS_DIR)

json_files = sorted(glob.glob(os.path.join(script_dir, "*_sentences.json")))
if not json_files:
    raise FileNotFoundError(f"No *_sentences.json files found in {script_dir}")
elif len(json_files) == 1:
    filename = json_files[0]
    print(f"Found sentences file: {filename}")
else:
    print("Multiple sentences files found:")
    for i, f in enumerate(json_files):
        print(f"  [{i}] {os.path.basename(f)}")
    choice = input("Select file index (default is 0): ")
    try:
        filename = json_files[int(choice)]
    except Exception:
        filename = json_files[0]
    print(f"Using: {filename}")

base_name = os.path.basename(filename).replace("_sentences.json", "")

with open(filename, "r") as fp:
    data = json.load(fp)

abstracts = data
num_abstracts = len(abstracts)
max_ab_idx = num_abstracts - 1
print(f"Successfully loaded {num_abstracts} abstracts.\n")

start_ab = input("Start parsing from abstract index (default is 0): ")
try:
    start_ab = min(int(start_ab), max_ab_idx)
except Exception:
    start_ab = 0

end_ab = input(f"End parsing at abstract index (default is {max_ab_idx}): ")
try:
    end_ab = min(int(end_ab), max_ab_idx)
except Exception:
    end_ab = max_ab_idx

total_sentences = sum(len(abstracts[i]["sentences"]) for i in range(start_ab, end_ab + 1))
confirm = input(f"About to parse {end_ab - start_ab + 1} abstracts ({total_sentences} sentences total). Confirm? (Y/N): ")
if confirm.lower() != "y":
    print("Aborted.")
    exit(0)

print(f"... going to parse abstracts from index {start_ab} to {end_ab}\n")

for ab_i in range(start_ab, end_ab + 1):
    entry = abstracts[ab_i]
    abstract_id = entry.get("pmid", entry.get("idx", ab_i))
    sentences = entry["sentences"]
    sp_out_file = os.path.join(script_dir, f"{base_name}_parses_{model}_{effort}_{abstract_id}.json")

    print(f"=== Abstract {ab_i} (pmid={abstract_id}, {len(sentences)} sentences) ===")

    abstract_outputs = []
    previous_parses = []

    for sent_i, sentence in enumerate(sentences):
        print(f"... looking at sentence: {sentence}")

        sent_result = nl2pln(sentence, mode="parsing", context=previous_parses, model=model, effort=effort)
        if sent_result is not None:
            type_defs, stmts, _, extra_exprs, sent_links = sent_result

            print(f"---\n\nSentence:\n{sentence}\n\ntype_defs:\n{type_defs}\n\nstmts:\n{stmts}\n\nextra_exprs:\n{extra_exprs}\n\n---\n")
            abstract_outputs.append({
                "abstract_idx": ab_i,
                "abstract_id": abstract_id,
                "sentence_idx": sent_i,
                "sentence": sentence,
                "type_defs": type_defs,
                "stmts": stmts,
                "extra_exprs": extra_exprs,
            })
            output_to_json_file(abstract_outputs, sp_out_file)
            faiss_store.save(FAISS_DIR)
            previous_parses = (previous_parses + [{"sentence": sentence, "stmts": stmts}])[-5:]
        else:
            failed_cases.append((ab_i, sentence))

if failed_cases:
    print(f"Failed to parse {len(failed_cases)} sentence(s):")
    for ab_i, sent in failed_cases:
        print(f"  abstract {ab_i}: {sent}")

finish_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
print(f"FINISHED PARSING at {finish_time}")
