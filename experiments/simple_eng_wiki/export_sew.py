import json
import os
import spacy
from datasets import load_dataset

script_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Load the spaCy English model
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# Increase max_length in case some Wikipedia articles are exceptionally long
nlp.max_length = 2000000

# 2. Load the latest Simple English Wikipedia dataset
# Using streaming=True so we don't have to download the entire dataset at once
print("Connecting to Hugging Face dataset...")
dataset = load_dataset(
    "omarkamali/wikipedia-monthly",
    "latest.simple", # "latest.simple" fetches the most recent Simple English dump
    split="train",
    streaming=True
)

# 3. Process articles
print("\nProcessing articles...\n" + "="*40)

output = []
for idx, article in enumerate(dataset):
    title = article['title']
    text = article['text']

    doc = nlp(text)
    sentences = [" ".join(sent.text.split()) for sent in doc.sents]
    sentences = [s for s in sentences if s]

    if sentences:
        output.append({"idx": idx, "title": title, "sentences": sentences})
        print(f"[{idx}] {title} ({len(sentences)} sentences)")

out_path = os.path.join(script_dir, "sew_sentences.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nWrote {len(output)} articles to {out_path}")
