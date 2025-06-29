import numpy as np
from app.faiss_cache_manager import (
    save_faiss_cache, load_faiss_cache,
    add_embedding_to_index, remove_embedding_from_index,
    get_faiss_state, rebuild_index
)
# from app.config import FAISS_INDEX_PATH, FAISS_PATHS_PATH

def test_add_embedding_to_index():
    state = get_faiss_state()
    dummy_vector = np.random.rand(1, 512).astype(np.float32)
    dummy_path = "/tmp/new.jpg"
    
    # Clear state before test
    rebuild_index(dummy_vector, [dummy_path])

    add_embedding_to_index(dummy_path, dummy_vector)
    assert dummy_path in state["paths"]
    assert state["index"].ntotal >= 1

def test_remove_embedding_from_index():
    state = get_faiss_state()
    dummy_vector = np.random.rand(1, 512).astype(np.float32)
    dummy_path = "/tmp/remove.jpg"
    
    rebuild_index(dummy_vector, [dummy_path])
    assert dummy_path in state["paths"]

    remove_embedding_from_index(dummy_path)
    assert dummy_path not in state["paths"]

def test_save_and_load_faiss_cache(tmp_path):
    
    dummy_vectors = np.random.rand(5, 512).astype(np.float32)
    dummy_paths = [f"/tmp/img_{i}.jpg" for i in range(5)]

    rebuild_index(dummy_vectors, dummy_paths)
    save_faiss_cache()

    # Reset state to simulate fresh load
    state = get_faiss_state()
    state["index"] = None
    state["paths"] = []

    load_faiss_cache()
    assert state["index"].ntotal == 5
    assert state["paths"] == dummy_paths
