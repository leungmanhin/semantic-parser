import os
import json
import time
import warnings
import urllib.request
from http.client import RemoteDisconnected
import numpy as np
from pydantic import create_model

MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds

def to_openrouter(prompt, model="gpt-5.4", effort="none", history=None, tools=[{"type": "web_search"}], output_format=create_model('StrResp', response=(str, ...))):
    history = [] if history is None else history
    history.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": history,
        "tools": tools,
        "max_tokens": 128000,
        "reasoning": {"effort": effort},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": output_format.__name__,
                "schema": output_format.model_json_schema(),
                "strict": True
            }
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data = data,
        headers = {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json"
        }
    )
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            history.append({"role": "assistant", "content": content})
            return json.loads(content)
        except RemoteDisconnected as e:
            if attempt < MAX_RETRIES - 1:
                print(f"Connection dropped, retrying in {RETRY_DELAY}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                raise

def get_embedding(word, model="text-embedding-3-large"):
    payload = {
        "model": model,
        "input": [word.replace("\n", " ")]
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/embeddings",
        data = data,
        headers = {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json"
        }
    )
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            v = np.array(result["data"][0]["embedding"], dtype=np.float32)
            return v / np.linalg.norm(v)
        except RemoteDisconnected as e:
            if attempt < MAX_RETRIES - 1:
                print(f"Connection dropped, retrying in {RETRY_DELAY}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                raise
