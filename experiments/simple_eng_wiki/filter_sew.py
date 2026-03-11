import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sentences_file = os.path.join(script_dir, "sew_sentences.json")

with open(sentences_file, "r") as f:
    articles = json.load(f)

print(f"Loaded {len(articles)} articles.\n")

min_sents = input("Show articles with at least how many sentences (default is 1): ")
try:
    min_sents = int(min_sents)
except Exception:
    min_sents = 1

max_sents = input("Show articles with at most how many sentences (leave blank for no limit): ")
try:
    max_sents = int(max_sents)
except Exception:
    max_sents = None

matches = [
    {"idx": entry["idx"], "title": entry["title"], "num_sentences": len(entry["sentences"])}
    for entry in articles
    if len(entry["sentences"]) >= min_sents
    and (max_sents is None or len(entry["sentences"]) <= max_sents)
]

range_desc = f">= {min_sents}" + (f" and <= {max_sents}" if max_sents is not None else "")
print(f"\nFound {len(matches)} articles with {range_desc} sentences:\n")
print(f"{'IDX':>6}  {'SENTENCES':>9}  TITLE")
print("-" * 60)
for m in matches:
    print(f"{m['idx']:>6}  {m['num_sentences']:>9}  {m['title']}")

save = input("\nSave these indices to a file for batch_parse_sew.py? (Y/N): ")
if save.lower() == "y":
    indices = [m["idx"] for m in matches]
    min_part = f"gte{min_sents}"
    max_part = f"_lte{max_sents}" if max_sents is not None else ""
    out_path = os.path.join(script_dir, f"sew_filter_{min_part}{max_part}sents.json")
    with open(out_path, "w") as f:
        json.dump(indices, f, indent=2)
    print(f"Saved {len(indices)} indices to {out_path}")
