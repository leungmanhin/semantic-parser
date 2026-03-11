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

articles = data
num_articles = len(articles)
max_art_idx = num_articles - 1
print(f"Successfully loaded {num_articles} articles.\n")

indices_file = input("Path to indices JSON file (leave blank to specify a range instead): ").strip()
if indices_file:
    with open(indices_file, "r") as f:
        art_indices = json.load(f)
    art_indices = [i for i in art_indices if 0 <= i <= max_art_idx]
else:
    start_art = input("Start parsing from article index (default is 0): ")
    try:
        start_art = min(int(start_art), max_art_idx)
    except Exception:
        start_art = 0

    end_art = input(f"End parsing at article index (default is {max_art_idx}): ")
    try:
        end_art = min(int(end_art), max_art_idx)
    except Exception:
        end_art = max_art_idx

    art_indices = list(range(start_art, end_art + 1))

total_sentences = sum(len(articles[i]["sentences"]) for i in art_indices)
print(f"\n{'IDX':>6}  {'SENTENCES':>9}  TITLE")
print("-" * 60)
for i in art_indices:
    entry = articles[i]
    print(f"{i:>6}  {len(entry['sentences']):>9}  {entry.get('title', i)}")
print("-" * 60)
confirm = input(f"\nAbout to parse {len(art_indices)} articles ({total_sentences} sentences total). Confirm? (Y/N): ")
if confirm.lower() != "y":
    print("Aborted.")
    exit(0)

print(f"... going to parse {len(art_indices)} articles\n")

for art_i in art_indices:
    entry = articles[art_i]
    title = entry.get("title", entry.get("idx", art_i))
    sentences = entry["sentences"]
    title_slug = title.replace(" ", "_")
    sp_out_file = os.path.join(script_dir, f"{base_name}_parses_{model}_{effort}_{title_slug}.json")

    print(f"=== Article {art_i} (title={title}, {len(sentences)} sentences) ===")

    article_outputs = []
    previous_parses = []

    for sent_i, sentence in enumerate(sentences):
        print(f"... looking at sentence ({sent_i} out of {len(sentences)}): {sentence}")

        sent_result = nl2pln(sentence, mode="parsing", context=previous_parses, model=model, effort=effort)
        if sent_result is not None:
            type_defs, stmts, _, extra_exprs, sent_links = sent_result

            print(f"---\n\nSentence:\n{sentence}\n\ntype_defs:\n{type_defs}\n\nstmts:\n{stmts}\n\nextra_exprs:\n{extra_exprs}\n\n---\n")
            article_outputs.append({
                "article_idx": art_i,
                "title": title,
                "sentence_idx": sent_i,
                "sentence": sentence,
                "type_defs": type_defs,
                "stmts": stmts,
                "extra_exprs": extra_exprs,
            })
            output_to_json_file(article_outputs, sp_out_file)
            faiss_store.save(FAISS_DIR)
            previous_parses = (previous_parses + [{"sentence": sentence, "stmts": stmts}])[-5:]
        else:
            failed_cases.append((art_i, sentence))

if failed_cases:
    print(f"Failed to parse {len(failed_cases)} sentence(s):")
    for art_i, sent in failed_cases:
        print(f"  article {art_i}: {sent}")

finish_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
print(f"FINISHED PARSING at {finish_time}")
