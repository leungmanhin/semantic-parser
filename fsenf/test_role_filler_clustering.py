"""
End-to-end test for the paper-faithful Role-Filler Distribution Clustering.

Uses filler-name embeddings instead of type-label embeddings.
"""

import json
import numpy as np
from pathlib import Path
from role_filler_clustering_2 import (
    extract_slot_fillers,
    embed_slot_fillers,
    suggest_merges,
    generate_consolidation_rules,
)
from role_filler_clustering import cluster_slots

CACHE_PATH = Path(__file__).parent / "embedding_cache_fillers.npz"


def load_embedding_cache() -> dict:
    if CACHE_PATH.exists():
        data = np.load(CACHE_PATH)
        cache = {k: data[k] for k in data.files}
        print(f"  Loaded {len(cache)} cached embedding(s) from {CACHE_PATH.name}")
        return cache
    return {}


def save_embedding_cache(cache: dict) -> None:
    np.savez(CACHE_PATH, **cache)
    print(f"  Saved {len(cache)} embedding(s) to {CACHE_PATH.name}")

# === Tunable parameters 
MIN_FILLERS = 2               # slots with fewer distinct fillers are skipped
SIMILARITY_THRESHOLD = 0.5   # cosine similarity cutoff for clustering slots
# 

SEW_DIR = Path(__file__).parent.parent / "experiments" / "simple_eng_wiki"
FILE_PATTERN = "sew_parses_gpt-5.4*"

matched_files = sorted(SEW_DIR.glob(FILE_PATTERN))
if not matched_files:
    raise FileNotFoundError(
        f"No files matched pattern {FILE_PATTERN!r} in {SEW_DIR}"
    )

print(f"Loading {len(matched_files)} file(s) matching '{FILE_PATTERN}':")
corpus = []
for path in matched_files:
    print(f"  {path.name}")
    with open(path) as f:
        corpus.extend(json.load(f))
print(f"Total corpus size: {len(corpus)} item(s)\n")

print("=== Step 1: Extracting slot fillers ")
slot_fillers = extract_slot_fillers(corpus, min_fillers=MIN_FILLERS)
print(f"  {len(slot_fillers)} predicate-slot(s) with >= {MIN_FILLERS} distinct fillers.\n")
for slot, fillers in sorted(slot_fillers.items()):
    print(f"  {slot} ({len(fillers)} fillers):")
    for f in fillers:
        print(f"    {f}")
    # for f in fillers[:8]:
    #     print(f"    {f}")
    # if len(fillers) > 8:
    #     print(f"    ... and {len(fillers) - 8} more")

print()
print("=== Step 2: Embedding filler names ")
embedding_cache = load_embedding_cache()
slot_centroids, embedding_cache = embed_slot_fillers(
    slot_fillers,
    embedding_cache=embedding_cache,
    on_new_embedding=save_embedding_cache,
    verbose=True,
)
print(f"  {len(slot_centroids)} slot centroid(s) computed.")
print(f"  {len(embedding_cache)} unique filler name(s) in cache.")

print()
print("=== Step 3: Clustering slots ")
all_clusters = cluster_slots(slot_centroids, similarity_threshold=SIMILARITY_THRESHOLD)
print(f"  Threshold: {SIMILARITY_THRESHOLD}")
print(f"  {len(all_clusters)} cluster(s) total.\n")

merge_candidates = suggest_merges(all_clusters)
print(f"  {len(merge_candidates)} merge candidate group(s):\n")
if merge_candidates:
    for i, group in enumerate(merge_candidates, 1):
        print(f"  [{i}]")
        for slot in group:
            fillers = slot_fillers.get(slot, [])
            print(f"      {slot}  →  {fillers[:6]}")
else:
    print("  No merge candidates at this threshold.")
    print("  Try lowering SIMILARITY_THRESHOLD or adding more corpus data.")

print()
print("=== Step 4: LLM-based consolidation rule generation ")
consolidation = generate_consolidation_rules(
    merge_candidates=merge_candidates,
    slot_fillers=slot_fillers,
    verbose=True,
)
print()
if consolidation.get("type_defs"):
    print("  Type definitions:")
    for td in consolidation["type_defs"]:
        print(f"    {td}")
    print()
if consolidation.get("rules"):
    print(f"  {len(consolidation['rules'])} rule(s):")
    for r in consolidation["rules"]:
        label = r.get("source_group", "?")
        candidates = r.get("candidates", [])
        print(f"    [{label}] {r['rule']}")
        print(f"      # {r.get('rule_nl', '')}")
        print(f"      from: {candidates}")
else:
    print("  No consolidation rules generated.")
    print("  (Either no merge candidates, or the LLM judged all suggestions spurious.)")
