import os
import cv2
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
from utils import get_image_embedding, load_image_safe, chroma_collection
from captioner import generate_caption
from config import DEVICE
from database import insert_video_metadata
import datetime

def extract_frames_from_video(
    video_path,
    output_dir,
    every_n_seconds=1,
    prefix="frame"
):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:
        fps = 25

    frame_interval = int(fps * every_n_seconds)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    basename = os.path.splitext(os.path.basename(video_path))[0]
    frames_info = []

    print(f"[INFO] Extracting frames from {video_path} ({duration:.1f}s, {frame_count} frames, {fps:.1f} fps)")

    frame_idx = 0
    saved_idx = 0
    pbar = tqdm(total=frame_count)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp_sec = frame_idx / fps
            timestamp_str = str(timedelta(seconds=int(timestamp_sec)))
            frame_filename = f"{prefix}_{basename}_{saved_idx:05d}_{timestamp_str.replace(':', '-')}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frames_info.append({
                "frame_path": frame_path,
                "timestamp": timestamp_str,
                "frame_idx": frame_idx
            })
            saved_idx += 1

        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    print(f"[INFO] Saved {saved_idx} frames to {output_dir}")
    return frames_info

def compute_video_metadata(video_path, frames_info, model, processor, device):
    key_frames = [f["frame_path"] for f in frames_info[:3] if os.path.exists(f["frame_path"])]
    import numpy as np
    embeddings = []
    for f in key_frames:
        image = load_image_safe(f)
        if image is not None:
            emb = get_image_embedding(model, processor, image, device)
            embeddings.append(emb)
    if embeddings:
        video_embedding = np.mean(embeddings, axis=0).tolist()
    else:
        video_embedding = [0.0] * 512
    summary_caption = " ; ".join([generate_caption(f) for f in key_frames])
    return key_frames, video_embedding, summary_caption

def safe_str(val):
    return "" if val is None else str(val)

def process_video_and_index(
    video_path,
    output_dir,
    model,
    processor,
    device,
    camera_id=None,
    every_n_seconds=1,
    prefix="frame"
):
    start_time = datetime.fromtimestamp(os.path.getctime(video_path)).isoformat()
    frames_info = extract_frames_from_video(video_path, output_dir, every_n_seconds, prefix)
    key_frames, video_embedding, summary_caption = compute_video_metadata(
        video_path, frames_info, model, processor, device
    )

    if not key_frames:
        print(f"[WARN] No key frames extracted for {video_path}. Skipping video metadata insert.")
        return
    end_time = datetime.fromtimestamp(os.path.getmtime(video_path)).isoformat()
    meta_dict = {
        "video_path": safe_str(video_path),
        "start_time": safe_str("N/A"),
        "end_time": safe_str("N/A"),
        "camera_id": safe_str(camera_id),
        "summary_caption": safe_str(summary_caption),
        "key_frames": ",".join(key_frames) if key_frames else ""
    }
    print("[DEBUG] Video metadata to insert:", meta_dict)
    for k, v in meta_dict.items():
        print(f"[DEBUG] {k}: {v} (type: {type(v)})")

    # Now insert into ChromaDB
    chroma_collection.add(
        embeddings=[video_embedding],
        metadatas=[meta_dict],
        ids=[safe_str(video_path)]
    )
    insert_video_metadata(
    video_path=str(video_path),
    start_time=str("NA"),
    end_time=str("NA"),
    camera_id=str(camera_id) if camera_id is not None else "",
    summary_caption=str(summary_caption) if summary_caption is not None else "",
    key_frames=",".join(key_frames) if key_frames else ""
    )
    

if __name__ == "__main__":
    import argparse
    from utils import load_lora_model

    parser = argparse.ArgumentParser(description="Extract frames from video(s) and index video metadata.")
    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save frames")
    parser.add_argument("--every_n_seconds", type=int, default=1, help="Extract a frame every N seconds")
    parser.add_argument("--prefix", type=str, default="frame", help="Prefix for saved frame files")

    args = parser.parse_args()

    model, processor, device = load_lora_model()

    if args.video:
        process_video_and_index(
            args.video, args.output_dir, model, processor, device, every_n_seconds=args.every_n_seconds, prefix=args.prefix
        )
    else:
        print("Please provide --video")