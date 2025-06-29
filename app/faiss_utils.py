import faiss
import numpy as np

def build_faiss_index(embeddings, dim):
    embeddings = embeddings.astype(np.float32)  # Convert to float32 first
    faiss.normalize_L2(embeddings)              # Then normalize
    index = faiss.IndexFlatIP(dim)              # Inner product index
    index.add(embeddings)                       # Add normalized vectors
    return index

def search_faiss(index, query_embedding, top_k=5):
    faiss.normalize_L2(query_embedding)
    sims, indices = index.search(query_embedding.astype(np.float32), top_k)
    return sims[0], indices[0]