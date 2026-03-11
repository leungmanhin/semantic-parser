import os
import json
import pickle
import faiss
import numpy as np
from llm import get_embedding

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072  # dimension for text-embedding-3-large

class SemanticArityIndex:
    def __init__(self):
        self.indices = {}      # Format: { arity_int: faiss_index }
        self.id_to_word = {}   # Format: { arity_int: { faiss_id: "word" } }
        self.word_to_id = {}   # Format: { arity_int: { "word": faiss_id } }

    def _get_or_create_index(self, arity: int):
        if arity not in self.indices:
            # Use IndexFlatIP (Inner Product).
            # With normalized vectors, this equals Cosine Similarity.
            self.indices[arity] = faiss.IndexFlatIP(EMBEDDING_DIM)
            self.id_to_word[arity] = {}
            self.word_to_id[arity] = {}
        return self.indices[arity]

    def store(self, word: str, arity: int):
        """
        Generates embedding for 'word' and stores it in the correct arity-index (if not already present).
        """
        embedding = get_embedding(word).reshape(1, -1)
        index = self._get_or_create_index(arity)
        word_map = self.word_to_id[arity]
        id_map = self.id_to_word[arity]

        if word not in word_map:
            new_id = index.ntotal
            index.add(embedding)
            word_map[word] = new_id
            id_map[new_id] = word

    def search(self, word: str, arity: int, n_closest: int = 10, threshold: float = 0.7):
        """
        Returns the N closest stored predicates to 'word' (by cosine similarity) that meet the threshold.
        'word' must already be stored; raises KeyError otherwise.
        """
        index = self._get_or_create_index(arity)
        word_map = self.word_to_id[arity]
        id_map = self.id_to_word[arity]

        if word not in word_map:
            raise KeyError(f"'{word}' (arity {arity}) is not in the index. Call store() first.")

        embedding = get_embedding(word).reshape(1, -1)
        distances, indices = index.search(embedding, n_closest + 1)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            found_word = id_map[idx]
            if found_word == word:
                continue
            if score < threshold:
                break
            results.append((found_word, float(score)))
            if len(results) >= n_closest:
                break

        return results

    def clear(self):
        self.indices = {}
        self.id_to_word = {}
        self.word_to_id = {}
        print("All indices and metadata have been cleared.")

    # =========================
    # Persistence: SAVE / LOAD
    # =========================

    def save(self, folder_path: str = "data/faiss/"):
        os.makedirs(folder_path, exist_ok=True)

        config = {
            "dimension": EMBEDDING_DIM,
            "arities": list(self.indices.keys()),
        }
        with open(os.path.join(folder_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)

        for arity, index in self.indices.items():
            index_path = os.path.join(folder_path, f"arity_{arity}.index")
            meta_path = os.path.join(folder_path, f"arity_{arity}_meta.pkl")

            faiss.write_index(index, index_path)

            meta = {
                "id_to_word": self.id_to_word[arity],
                "word_to_id": self.word_to_id[arity],
            }
            with open(meta_path, "wb") as f:
                pickle.dump(meta, f)

    @classmethod
    def load(cls, folder_path: str):
        config_path = os.path.join(folder_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        instance = cls()

        for arity in config["arities"]:
            index_path = os.path.join(folder_path, f"arity_{arity}.index")
            meta_path = os.path.join(folder_path, f"arity_{arity}_meta.pkl")

            instance.indices[arity] = faiss.read_index(index_path)

            with open(meta_path, "rb") as f:
                meta = pickle.load(f)

            instance.id_to_word[arity] = meta["id_to_word"]
            instance.word_to_id[arity] = meta["word_to_id"]

        return instance

faiss_store = SemanticArityIndex()
