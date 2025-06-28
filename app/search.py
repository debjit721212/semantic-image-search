import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from config import CACHE_PATH, TOP_K_RESULTS
from cache_manager import load_cache
import logging
from database import get_metadata
import os

logger = logging.getLogger(__name__)

def encode_prompt(prompt, model, processor, device):
    inputs = processor.tokenizer(text=[prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    return text_embedding.cpu().numpy()

def search_prompt(prompt, image_embeddings, image_paths, model, processor, device, top_k=5):
    prompt_emb = encode_prompt(prompt, model, processor, device)

    print(">> Prompt embedding shape:", prompt_emb.shape)
    print(">> Image embeddings shape:", image_embeddings.shape)
    print("search image path ---> ",image_paths)

    if image_embeddings.shape[0] == 0:
        logger.warning("Empty image embeddings. Returning no results.")
        return []

    sims = cosine_similarity(prompt_emb, image_embeddings)[0]
    indices = np.argsort(sims)[::-1][:top_k]
    results = [(image_paths[i], sims[i]) for i in indices]
    return results

# def perform_search(prompt, threshold, model, processor, device,return_metadata=False ):
#     paths, embeddings = load_cache(CACHE_PATH)
#     if not paths or embeddings is None:
#         return []

#     results = search_prompt(prompt, embeddings, paths, model, processor, device, top_k=TOP_K_RESULTS)
#     filtered = [(p, s) for p, s in results if s >= threshold]
#     return filtered


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
        embeddings = np.vstack(embeddings)

    results = search_prompt(prompt, embeddings, paths, model, processor, device, top_k=TOP_K_RESULTS)
    filtered = [(p, s) for p, s in results if s >= threshold]

    if return_metadata:
        return [(p, s, get_metadata(p)) for p, s in filtered]

    return filtered
