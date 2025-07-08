import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from pathlib import Path
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor
from utils import compute_embedding, insert_embedding_chroma, load_image_safe, get_image_embedding, chroma_collection
from captioner import generate_caption
from video_chunker import extract_frames_from_video, compute_video_metadata,process_video_and_index
from frame_tagger import tag_frames
from config import IMAGE_DIR, FRAMES_DIR
from database import insert_metadata, insert_video_metadata

logging.basicConfig(
    filename="observer.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

executor = ThreadPoolExecutor(max_workers=4)

class UnifiedEventHandler(FileSystemEventHandler):
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        suffix = path.suffix.lower()
        if suffix in [".jpg", ".jpeg", ".png"]:
            executor.submit(self.process_image, path)
        elif suffix in [".mp4", ".avi", ".mov", ".mkv"]:
            executor.submit(self.process_video, path)

    def process_image(self, path):
        try:
            logging.info(f"[ New Image Detected] {path.name}")
            print(f"[ New Image Detected] {path.name}")
            image = load_image_safe(path)
            if image is None:
                logging.warning(f"[WARN] Skipping embedding for {path} (image load failed)")
                print(f"[WARN] Skipping embedding for {path} (image load failed)")
                return
            caption = generate_caption(path)
            embedding = compute_embedding(path, self.model, self.processor, self.device)
            match = re.match(r"(cam_\d+)_.*", path.stem)
            camera_id = match.group(1) if match else None
            insert_embedding_chroma(
                embedding=embedding,
                image_path=path,
                caption=caption,
                camera_id=camera_id,
                timestamp=datetime.now().isoformat()
            )
            insert_metadata(
                filename=str(path.resolve()),
                timestamp=datetime.now().isoformat(),
                camera_id=camera_id,
                caption=caption,
                confidence=0.9
            )
            logging.info(f"[ChromaDB] Inserted embedding for {path.name}")
            print(f"[ChromaDB] Inserted embedding for {path.name}")
        except Exception as e:
            logging.error(f"[ERROR] Exception in process_image for {path}: {e}", exc_info=True)
            print(f"[ERROR] Exception in process_image for {path}: {e}")

    def process_video(self, path):
        try:
            logging.info(f"[ New Video Detected] {path.name}")
            print(f"[ New Video Detected] {path.name}")
            video_name = path.stem
            frames_output_dir = Path(FRAMES_DIR) / video_name
            frames_output_dir.mkdir(parents=True, exist_ok=True)
            # Call the robust, tested function from video_chunker.py
            process_video_and_index(
                str(path),
                str(frames_output_dir),
                self.model,
                self.processor,
                self.device,
                camera_id=None,
                every_n_seconds=1,
                prefix="frame"
            )
            
            logging.info(f"[Video Processed] {path.name}")
            print(f"[Video Processed] {path.name}")
        except Exception as e:
            logging.error(f"[ERROR] Exception in process_video for {path}: {e}", exc_info=True)
            print(f"[ERROR] Exception in process_video for {path}: {e}")

def start_observer(model, processor, device=None):
    event_handler = UnifiedEventHandler(model, processor, device)
    observer = Observer()
    logging.info(f"************************  {IMAGE_DIR}")
    print("************************ ", IMAGE_DIR)
    observer.schedule(event_handler, str(IMAGE_DIR), recursive=True)
    observer.start()
    logging.info("[👁️ Unified File Observer Started]")
    print("[👁️ Unified File Observer Started]")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()