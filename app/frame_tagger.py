import os
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from utils import load_lora_model, get_image_embedding, insert_embedding_chroma, load_image_safe
from captioner import generate_caption  # Your BLIP/QLoRA captioning function

def tag_frames(
    frames_dir,
    model,
    processor,
    camera_id=None,
    chroma_insert_fn=insert_embedding_chroma,
    select_key_frames=True,
    key_frame_interval=10  # e.g., every 10th frame is a key frame
):
    """
    Tag all images in frames_dir with CLIP/LoRA embedding and BLIP/QLoRA caption.
    Insert results into ChromaDB.
    Optionally, select and return key frames for video preview.
    """
    files = [f for f in sorted(os.listdir(frames_dir)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"[INFO] Tagging {len(files)} frames in {frames_dir}...")

    key_frames = []
    for idx, fname in enumerate(tqdm(files)):
        img_path = os.path.join(frames_dir, fname)
        image = load_image_safe(img_path)
        if image is None:
            continue

        # Generate caption (BLIP/QLoRA)
        caption = generate_caption(img_path)

        # Generate embedding (CLIP/LoRA)
        embedding = get_image_embedding(model, processor, image)

        # Extract timestamp from filename if present, else use now
        try:
            timestamp_str = fname.split("_")[-1].replace(".jpg", "").replace("-", ":")
            timestamp = timestamp_str if ":" in timestamp_str else datetime.now().isoformat()
        except Exception:
            timestamp = datetime.now().isoformat()

        # Insert into ChromaDB
        chroma_insert_fn(
            embedding=embedding,
            image_path=img_path,
            caption=caption,
            camera_id=camera_id,
            timestamp=timestamp
        )

        # Select key frames (e.g., every Nth frame)
        if select_key_frames and (idx % key_frame_interval == 0):
            key_frames.append(img_path)

        print(f"[ChromaDB] Tagged and inserted {fname}")

    # Return key frames for video preview/metadata
    return key_frames

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tag frames/images with CLIP/LoRA and BLIP/QLoRA, insert into ChromaDB.")
    parser.add_argument("--frames_dir", required=True, help="Folder containing frames/images to tag")
    parser.add_argument("--camera_id", default=None, help="Camera ID (optional)")
    parser.add_argument("--key_frame_interval", type=int, default=10, help="Interval for selecting key frames")
    args = parser.parse_args()

    # Load CLIP/LoRA model and processor
    model, processor, _ = load_lora_model()

    key_frames = tag_frames(
        frames_dir=args.frames_dir,
        model=model,
        processor=processor,
        camera_id=args.camera_id,
        select_key_frames=True,
        key_frame_interval=args.key_frame_interval
    )

    print(f"[INFO] Key frames selected: {key_frames}")