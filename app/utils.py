import os
import logging
from PIL import Image, UnidentifiedImageError
from typing import List
import torch
import numpy as np
from datetime import datetime, timedelta
import chromadb
from transformers import CLIPProcessor, CLIPModel, pipeline
from config import DEVICE, CLIP_MODEL_NAME, IMAGE_DIR, IMAGE_RETENTION_DAYS, CACHE_PATH,lora_weights_path
from database import delete_metadata_for_images
import time
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from peft import PeftModel, LoraConfig, get_peft_model

logger = logging.getLogger(__name__)

# --- ChromaDB Setup ---
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(name="surveillance_embeddings")

def insert_embedding_chroma(embedding, image_path, caption, camera_id, timestamp):
    chroma_collection.add(
        embeddings=[embedding],
        metadatas=[{
            "image_path": str(image_path),
            "caption": caption,
            "camera_id": camera_id,
            "timestamp": timestamp
        }],
        ids=[str(image_path)]
    )
    logger.info(f"[ChromaDB] Inserted embedding for {image_path}")

def delete_embeddings_by_image_paths(image_paths):
    chroma_collection.delete(ids=[str(p) for p in image_paths])
    logger.info(f"[ChromaDB] Deleted embeddings for {len(image_paths)} images")

def compute_embedding(image_path, model, processor, device=DEVICE):
    image = load_image_safe(image_path)
    if image is None:
        return None
    return get_image_embedding(model, processor, image, device)

# --- RAG LLM Pipeline ---
rag_llm = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)

def call_llm_with_context(user_question, context_captions):
    context = "\n".join(context_captions)
    prompt = (
        f"Based on the following surveillance events, answer the question: '{user_question}'\n\n"
        f"Events:\n{context}\n\n"
        "Answer:"
    )
    response = rag_llm(prompt)
    return response[0]["generated_text"].strip()

def retrieve_top_k_from_chroma(query, text_encoder, chroma_collection, k=5, time_filter=None, camera_filter=None):
    query_embedding = text_encoder.encode(query).tolist()
    where = {}
    if time_filter:
        where["timestamp"] = {"$gte": time_filter}
    if camera_filter:
        where["camera_id"] = camera_filter
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["metadatas", "distances"],
        where=where if where else None
    )
    return results

def get_langchain_agent(rag_llm, chroma_collection):
    llm = HuggingFacePipeline(pipeline=rag_llm)
    vectorstore = Chroma(collection=chroma_collection)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

# ------------------ Image Utilities ------------------

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
    return datetime.now().strftime("%Y-%m-%d")

def load_images_batched(image_paths: List[str], batch_size: int = 128) -> List[List[Image.Image]]:
    batches = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.warning(f"[load_images_batched] Failed to load image {path}: {e}")
        if images:
            batches.append(images)
    return batches

# def get_image_embedding(model, processor, image: Image.Image, device=DEVICE):
#     inputs = processor(images=image, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model.get_image_features(**inputs)
#     return outputs[0].cpu().tolist()  # Return as list for ChromaDB

def get_image_embedding(model, processor, image: Image.Image, device):
    import sys
    print(f"[DEBUG] device passed to get_image_embedding: {device}")
    print(f"[DEBUG] device type: {type(device)}, value: {device}")
    inputs = processor(images=image, return_tensors="pt")
    # Force to CUDA for testing
    inputs["pixel_values"] = inputs["pixel_values"].to("cuda")
    model = model.to("cuda")
    print(f"[DEBUG] (After moving) Model device: {next(model.parameters()).device}")
    print(f"[DEBUG] (After moving) Input device: {inputs['pixel_values'].device}")
    sys.stdout.flush()
    assert inputs["pixel_values"].device == next(model.parameters()).device, "Device mismatch!"
    with torch.no_grad():
        outputs = model.get_image_features(pixel_values=inputs["pixel_values"])
    return outputs[0].cpu().tolist()

def load_lora_model(lora_weights_path=lora_weights_path, base_model_name=CLIP_MODEL_NAME, device=DEVICE):
    """
    Loads a CLIP model with LoRA weights applied.
    """
    base_model = CLIPModel.from_pretrained(base_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(base_model_name)
    # Load LoRA weights
    lora_model = PeftModel.from_pretrained(base_model, lora_weights_path)
    lora_model = lora_model.to(device)
    return lora_model, processor, device


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

        if not os.listdir(root) and root != IMAGE_DIR:
            try:
                os.rmdir(root)
                logging.info(f"[CLEANUP] Removed empty folder: {root}")
            except Exception as e:
                logging.warning(f"[CLEANUP] Could not remove folder: {root}: {e}")

    if deleted_files:
        delete_metadata_for_images(deleted_files)
        rebuild_cache_without_files(deleted_files)

    logging.info(f"[CLEANUP] Finished. Total files deleted: {len(deleted_files)}")

def get_file_list_by_time_and_camera(minutes_back, camera_id=None, image_root="images"):
    now = time.time()
    threshold_time = now - (minutes_back * 60)
    results = []
    for root, _, files in os.walk(image_root):
        if camera_id and camera_id not in root:
            continue
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            path = os.path.join(root, fname)
            try:
                if os.path.getmtime(path) >= threshold_time:
                    results.append(path)
            except Exception:
                continue
    return results


# --- Video Search & Metadata Helpers ---

def compute_video_embedding(key_frame_paths, model, processor, device):
    """
    Compute a video-level embedding as the mean of key frame embeddings.
    """
    import numpy as np
    embeddings = []
    for f in key_frame_paths:
        image = load_image_safe(f)
        if image is not None:
            emb = get_image_embedding(model, processor, image, device)
            embeddings.append(emb)
    if embeddings:
        return np.mean(embeddings, axis=0).tolist()
    else:
        return [0.0] * 512  # fallback

def generate_video_summary_caption(key_frame_paths):
    """
    Generate a summary caption for a video by concatenating key frame captions.
    """
    captions = []
    for f in key_frame_paths:
        try:
            cap = generate_caption(f)
            captions.append(cap)
        except Exception as e:
            logger.warning(f"[utils] Failed to generate caption for {f}: {e}")
    return " ; ".join(captions)

def get_video_key_frames(frames_dir, interval=10):
    """
    Select key frames from a directory (e.g., every Nth frame).
    """
    files = [os.path.join(frames_dir, f) for f in sorted(os.listdir(frames_dir)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    key_frames = [f for idx, f in enumerate(files) if idx % interval == 0]
    return key_frames

def video_to_frame_mapping(video_metadata_df):
    """
    Build a mapping from video_path to its key frames (for analytics/UI).
    """
    mapping = {}
    for _, row in video_metadata_df.iterrows():
        video_path = row["video_path"]
        key_frames = row["key_frames"].split(",") if row["key_frames"] else []
        mapping[video_path] = key_frames
    return mapping

def insert_video_metadata_chroma(
    video_embedding,
    video_path,
    start_time,
    end_time,
    camera_id,
    summary_caption,
    key_frames,
    chroma_collection=chroma_collection
    ):
    """
    Insert video-level embedding and metadata into ChromaDB.
    """
    chroma_collection.add(
        embeddings=[video_embedding],
        metadatas=[{
            "video_path": str(video_path),
            "start_time": start_time,
            "end_time": end_time,
            "camera_id": camera_id,
            "summary_caption": summary_caption,
            "key_frames": key_frames  # list of frame paths
        }],
        ids=[str(video_path)]
    )
    logger.info(f"[ChromaDB] Inserted video metadata for {video_path}")