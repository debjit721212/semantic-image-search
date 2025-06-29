# ✅ Updated `faiss_cache_manager.py`
# Includes:
# 1. Persistent loading/saving
# 2. add_embedding_to_index(path, embedding)
# 3. remove_embedding_from_index(path)

import os
import numpy as np
import faiss
import pickle
from config import FAISS_INDEX_PATH, FAISS_PATHS_PATH

# Ensure FAISS storage directory exists
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)



_faiss_state = {
    "index": None,
    "paths": [],
    "dim": None,
}

def load_faiss_cache():
    if os.path.exists(FAISS_INDEX_PATH):
        _faiss_state["index"] = faiss.read_index(str(FAISS_INDEX_PATH))
    if os.path.exists(FAISS_PATHS_PATH):
        with open(FAISS_PATHS_PATH, "rb") as f:
            _faiss_state["paths"] = pickle.load(f)
    if _faiss_state["index"]:
        _faiss_state["dim"] = _faiss_state["index"].d

def save_faiss_cache():
    if _faiss_state["index"]:
        faiss.write_index(_faiss_state["index"], str(FAISS_INDEX_PATH))
        with open(FAISS_PATHS_PATH, "wb") as f:
            pickle.dump(_faiss_state["paths"], f)

def get_faiss_state():
    return _faiss_state

def rebuild_index(embeddings, paths):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype(np.float32))
    _faiss_state.update({
        "index": index,
        "paths": paths,
        "dim": dim,
    })
    save_faiss_cache()

def add_embedding_to_index(path, embedding):
    if _faiss_state["index"] is None:
        dim = embedding.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embedding)
        index.add(embedding.astype(np.float32))
        _faiss_state.update({
            "index": index,
            "paths": [path],
            "dim": dim,
        })
    else:
        faiss.normalize_L2(embedding)
        _faiss_state["index"].add(embedding.astype(np.float32))
        _faiss_state["paths"].append(path)
    save_faiss_cache()

def remove_embedding_from_index(path):
    if path not in _faiss_state["paths"]:
        return
    idx = _faiss_state["paths"].index(path)
    _faiss_state["paths"].pop(idx)

    if _faiss_state["paths"]:
        # Rebuild index with remaining
        # Assume embeddings are in cache (or reload from metadata db)
        from cache_manager import load_cache
        all_paths, all_embeddings = load_cache()
        filtered = [(p, e) for p, e in zip(all_paths, all_embeddings) if p in _faiss_state["paths"]]
        if filtered:
            new_paths, new_embs = zip(*filtered)
            new_embs = np.vstack(new_embs).astype(np.float32)
            rebuild_index(new_embs, list(new_paths))
    else:
        _faiss_state["index"] = None
        _faiss_state["dim"] = None
        save_faiss_cache()
        