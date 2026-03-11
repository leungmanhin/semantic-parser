import ftplib
import gzip
import json
import os
import re
import urllib.request
import pubmed_parser as pp
import spacy

nlp = spacy.load("en_core_web_sm")

script_dir = os.path.dirname(os.path.abspath(__file__))

def is_valid_gz(path):
    try:
        with gzip.open(path) as f:
            f.read()
        return True
    except Exception:
        return False

ftp = ftplib.FTP("ftp.ncbi.nlm.nih.gov")
ftp.login()
ftp.cwd("/pubmed/baseline/")
files = sorted([f for f in ftp.nlst() if f.endswith('.xml.gz')])
filename = files[-1]
ftp.quit()

local_path = os.path.join(script_dir, filename)

if not os.path.exists(local_path) or not is_valid_gz(local_path):
    url = f"https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/{filename}"
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, local_path)
    print("Download complete.")
else:
    print(f"Using cached {filename}")

dicts_out = list(pp.parse_medline_xml(local_path, author_list=False, reference_list=False))

print(f"Total entries: {len(dicts_out)}")

output = []
for idx, entry in enumerate(dicts_out):
    abstract = entry['abstract']
    if not abstract:
        continue
    abstract = re.sub(r'\n{3}[A-Z][A-Z\s/()\-]+\n', '\n\n\n', abstract)
    abstract = re.sub(r'^[A-Z][A-Z\s/()\-]+\n', '', abstract)
    doc = nlp(abstract)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if sentences:
        output.append({"idx": idx, "pmid": entry['pmid'], "sentences": sentences})

out_path = os.path.join(script_dir, filename.replace('.xml.gz', '_sentences.json'))
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Wrote {len(output)} abstracts to {out_path}")
