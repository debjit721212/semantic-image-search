import numpy as np
import torch
from config import CACHE_PATH, TOP_K_RESULTS
from cache_manager import load_cache
import logging
from database import get_metadata
import os
import faiss
from faiss_utils import build_faiss_index, search_faiss

logger = logging.getLogger(__name__)

_faiss_state = {
    "index": None,
    "paths": None,
    "embedding_dim": None
}

def encode_prompt(prompt, model, processor, device):
    inputs = processor.tokenizer(text=[prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    return text_embedding.cpu().numpy()

def prepare_faiss_index(image_embeddings, image_paths):
    dim = image_embeddings.shape[1]

    # Clear existing index before rebuilding
    index = build_faiss_index(image_embeddings, dim)

    _faiss_state.clear()  # <<< clear all previous state
    _faiss_state["index"] = index
    _faiss_state["paths"] = image_paths
    _faiss_state["embedding_dim"] = dim

def search_prompt(prompt, model, processor, device, top_k=TOP_K_RESULTS):
    prompt_emb = encode_prompt(prompt, model, processor, device)

    if _faiss_state["index"] is None or _faiss_state["paths"] is None:
        logger.warning("FAISS index or paths not initialized. Returning no results.")
        return []

    sims, indices = search_faiss(_faiss_state["index"], prompt_emb, top_k)

    results = []
    for i in range(len(indices)):
        idx = indices[i]
        if 0 <= idx < len(_faiss_state["paths"]):
            results.append((str(_faiss_state["paths"][idx]), float(sims[i])))
        else:
            logger.warning(f"Skipped invalid FAISS index {idx}, out of bounds for paths")
    return results

def perform_search(prompt, threshold=0.3, model=None, processor=None, device=None, return_metadata=False, file_list=None):
    paths, embeddings = load_cache(CACHE_PATH)
    if not paths or embeddings is None:
        return []

    if file_list:
        file_list = set(os.path.abspath(f) for f in file_list)
        filtered_pairs = [(p, emb) for p, emb in zip(paths, embeddings) if os.path.abspath(p) in file_list]
        if not filtered_pairs:
            return []
        paths, embeddings = zip(*filtered_pairs)

    embeddings = np.vstack(embeddings).astype(np.float32)

    prepare_faiss_index(embeddings, list(paths))
    results = search_prompt(prompt, model, processor, device, top_k=TOP_K_RESULTS)
    filtered = [(p, s) for p, s in results if s >= threshold]

    if return_metadata:
        return [(p, s, get_metadata(p)) for p, s in filtered]

    return filtered
