# app/cache_manager.py
import numpy as np
import os
import logging
import torch
from config import IMAGE_DIR, CACHE_PATH
from utils import load_images_batched,load_model, get_image_embedding,get_or_cache_image_embeddings,encode_image_file
from PIL import Image
from pathlib import Path

logger = logging.getLogger(__name__)

def load_cache(cache_path):
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            return data['paths'].tolist(), data['embeddings']
        except Exception as e:
            logger.warning(f"[cache] Failed to load cache: {e}")
    return [], np.array([])

def save_cache(cache_path, image_paths, embeddings):
    try:
        np.savez(cache_path, paths=np.array(image_paths), embeddings=embeddings)
        logger.info(f"[cache] Saved cache with {len(image_paths)} images")
    except Exception as e:
        logger.error(f"[cache] Error saving cache: {e}")
        
def ensure_cache(model, processor, device="cuda" if torch.cuda.is_available() else "cpu"):
    if os.path.exists(CACHE_PATH):
        logging.info(f"[Cache] Using existing cache at {CACHE_PATH}")
        data = np.load(CACHE_PATH)
        return data['paths'], data['embeddings']
    else:
        logging.info(f"[Cache] Building cache from {IMAGE_DIR}...")
        image_paths = []
        embeddings = []

        for image, path in load_images_batched(IMAGE_DIR):
            emb = get_image_embedding(model, processor, image, device=device)
            image_paths.append(path)
            embeddings.append(emb.cpu().numpy())

        embeddings = np.vstack(embeddings)
        np.savez(CACHE_PATH, paths=np.array(image_paths), embeddings=embeddings)
        logging.info(f"[Cache] Saved to {CACHE_PATH}")
        return image_paths, embeddings


def build_full_cache():
    print("[CACHE] Building full image cache...")
    model, processor, _ = load_model()
    image_paths, image_embeddings = get_or_cache_image_embeddings(
        model, processor, image_dir=IMAGE_DIR, cache_path=CACHE_PATH
    )
    print(f"[CACHE] Cached {len(image_paths)} image embeddings.")
    

def update_cache_with_new_image(image_path, model, processor):
    image_path = Path(image_path).resolve()  # ✅ Ensure absolute path

    # Load existing cache
    try:
        cache = np.load(CACHE_PATH, allow_pickle=True)
        paths = cache['paths'].tolist()
        embeddings = cache['embeddings']
    except FileNotFoundError:
        paths = []
        embeddings = np.empty((0, 512))

    # Avoid duplicate entries
    if str(image_path) in paths:
        print(f"[ℹ️ Already Cached] {image_path.name}")
        return

    print(f"[🧠 Encoding] {image_path.name}")
    new_emb = encode_image_file(image_path, model, processor)

    # Append and save
    paths.append(str(image_path))  # ✅ Now always absolute
    embeddings = np.vstack([embeddings, new_emb])
    np.savez(CACHE_PATH, paths=np.array(paths), embeddings=embeddings)
    print(f"[✅ Cache Updated] Added {image_path.name}")


# def update_cache_with_new_image(image_path, model, processor):
#     # Load existing cache
#     try:
#         cache = np.load(CACHE_PATH, allow_pickle=True)
#         paths = cache['paths'].tolist()
#         embeddings = cache['embeddings']
#     except FileNotFoundError:
#         paths = []
#         embeddings = np.empty((0, 512))

#     # Avoid duplicate entries
#     if str(image_path) in paths:
#         print(f"[ℹ️ Already Cached] {image_path.name}")
#         return

#     print(f"[🧠 Encoding] {image_path.name}")
#     new_emb = encode_image_file(image_path, model, processor)

#     # Append and save
#     paths.append(str(image_path))
#     embeddings = np.vstack([embeddings, new_emb])
#     np.savez(CACHE_PATH, paths=np.array(paths), embeddings=embeddings)
#     print(f"[✅ Cache Updated] Added {image_path.name}")


