# app/search.py

import logging
from utils import retrieve_top_k_from_chroma
from database import get_metadata

logger = logging.getLogger(__name__)

def perform_search(
    prompt,
    threshold=0.3,
    text_encoder=None,
    chroma_collection=None,
    return_metadata=False,
    file_list=None,
    k=5,
    time_filter=None,
    camera_filter=None
):
    """
    Semantic search using ChromaDB.
    Args:
        prompt: User query string.
        threshold: Minimum similarity score to include result.
        text_encoder: SentenceTransformer or CLIP text encoder.
        chroma_collection: ChromaDB collection object.
        return_metadata: If True, return metadata with results.
        file_list: Optional list of file paths to restrict search.
        k: Number of top results to retrieve.
        time_filter: Optional time filter for ChromaDB.
        camera_filter: Optional camera filter for ChromaDB.
    Returns:
        List of (image_path, score, [metadata]) tuples.
    """
    results = retrieve_top_k_from_chroma(
        prompt, text_encoder, chroma_collection, k=k, time_filter=time_filter, camera_filter=camera_filter
    )
    if not results or not results.get("metadatas") or not any(results["metadatas"][0]):
        return []

    filtered = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        if dist >= threshold:
            if file_list and meta["image_path"] not in file_list:
                continue
            if return_metadata:
                filtered.append((meta["image_path"], dist, meta))
            else:
                filtered.append((meta["image_path"], dist))
    return filtered