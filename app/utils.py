# app/utils.py
import os
from PIL import Image, UnidentifiedImageError
import logging
from typing import List
import torch
import clip
import numpy as np

from config import DEVICE, CLIP_MODEL_NAME
from datetime import datetime, timedelta
from config import IMAGE_DIR, IMAGE_RETENTION_DAYS,CACHE_PATH
# from cache_manager import rebuild_cache_without_files
from database import delete_metadata_for_images

logger = logging.getLogger(__name__)
from transformers import CLIPProcessor, CLIPModel

def load_image_safe(path):
    try:
        image = Image.open(path).convert("RGB")
        return image
    except UnidentifiedImageError:
        logger.warning(f"[utils] Corrupted image skipped: {path}")
        return None
    except Exception as e:
        logger.error(f"[utils] Error loading image {path}: {e}")
        return None

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
def is_image_file(filename):
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

def get_date_folder_name():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

def load_images_batched(image_paths: List[str], batch_size: int = 128) -> List[List[Image.Image]]:
    """
    Splits a list of image paths into batches and loads the images.
    Returns a list of batches, each containing PIL.Image objects.
    """
    batches = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                logging.warning(f"[load_images_batched] Failed to load image {path}: {e}")
        if images:
            batches.append(images)
    return batches

def get_image_embedding(model, processor, image: Image.Image, device="cuda" if torch.cuda.is_available() else "cpu"):
    # inputs = processor(images=image, return_tensors="pt").to(device)
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs[0]  # Single image


def load_model():

    model_name = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    model = model.to(device)  # 

    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device

def get_or_cache_image_embeddings(model, processor, image_dir, cache_path):


    if os.path.exists(cache_path):
        print(f"[CACHE] Loading image embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["paths"], data["embeddings"]

    print("[CACHE] Building image embeddings...")
    image_paths = []
    embeddings = []

    for root, _, files in os.walk(image_dir):
        for fname in sorted(files):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(root, fname)
            try:
                image = Image.open(fpath).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    embedding = model.get_image_features(**inputs)
                    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
                    embeddings.append(embedding.squeeze().cpu().numpy())
                    image_paths.append(fpath)
            except Exception as e:
                print(f" Failed to process {fpath}: {e}")

    embeddings = np.stack(embeddings)
    np.savez(cache_path, paths=image_paths, embeddings=embeddings)
    print(f"[CACHE] Saved to {cache_path}")
    return image_paths, embeddings

def encode_image_file(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs).cpu().numpy()
    return embedding

def save_cache(cache_path, image_paths, embeddings):
    try:
        np.savez(cache_path, paths=np.array(image_paths), embeddings=embeddings)
        logger.info(f"[cache] Saved cache with {len(image_paths)} images")
    except Exception as e:
        logger.error(f"[cache] Error saving cache: {e}")

def rebuild_cache_without_files(files_to_exclude):
    try:
        if not os.path.exists(CACHE_PATH):
            logger.warning("[cache] Cache file not found. Nothing to rebuild.")
            return

        cache = np.load(CACHE_PATH, allow_pickle=True)
        paths = cache['paths'].tolist()
        embeddings = cache['embeddings']

        # Build new lists
        new_paths = []
        new_embeddings = []
        for i, p in enumerate(paths):
            if p not in files_to_exclude:
                new_paths.append(p)
                new_embeddings.append(embeddings[i])

        if new_embeddings:
            new_embeddings = np.vstack(new_embeddings)
        else:
            new_embeddings = np.empty((0, 512))

        save_cache(CACHE_PATH, new_paths, new_embeddings)
        logger.info(f"[cache] Rebuilt cache excluding {len(files_to_exclude)} files")

    except Exception as e:
        logger.error(f"[cache] Error rebuilding cache: {e}")

def cleanup_old_images():
    now = datetime.now()
    cutoff = now - timedelta(days=IMAGE_RETENTION_DAYS)
    deleted_files = []
    
    logging.info(f"[CLEANUP] Starting cleanup. Deleting images older than {cutoff.strftime('%Y-%m-%d')}")

    for root, dirs, files in os.walk(IMAGE_DIR, topdown=False):
        for name in files:
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            file_path = os.path.join(root, name)
            try:
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_mtime < cutoff:
                    os.remove(file_path)
                    deleted_files.append(file_path)
                    logging.info(f"[CLEANUP] Deleted: {file_path}")
            except Exception as e:
                logging.error(f"[CLEANUP] Failed to delete {file_path}: {e}")

        # Remove empty dated folders
        if not os.listdir(root) and root != IMAGE_DIR:
            try:
                os.rmdir(root)
                logging.info(f"[CLEANUP] Removed empty folder: {root}")
            except Exception as e:
                logging.warning(f"[CLEANUP] Could not remove folder {root}: {e}")

    if deleted_files:
        # Step 2: Delete from metadata DB
        delete_metadata_for_images(deleted_files)

        # Step 3: Rebuild embeddings cache without deleted images
        rebuild_cache_without_files(deleted_files)

    logging.info(f"[CLEANUP] Finished. Total files deleted: {len(deleted_files)}")
