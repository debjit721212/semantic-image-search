# observer.py
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from pathlib import Path
from utils import load_model
from cache_manager import update_cache_with_new_image
from config import IMAGE_DIR
from database import insert_metadata
from datetime import datetime
from captioner import generate_caption
import re

class ImageEventHandler(FileSystemEventHandler):
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            return

        print(f"[📥 New Image Detected] {path.name}")

        # ✅ Generate caption
        caption = generate_caption(path)

        # ✅ Update embedding cache
        update_cache_with_new_image(path, self.model, self.processor)

        # ✅ Parse camera_id from filename
        match = re.match(r"(cam_\d+)_.*", path.stem)
        camera_id = match.group(1) if match else None

        # ✅ Insert metadata
        insert_metadata(
            filename=str(path.resolve()),
            timestamp=datetime.now().isoformat(),
            camera_id=camera_id,
            caption=caption,
            embedding_path=None
        )

def start_observer(model, processor):
    event_handler = ImageEventHandler(model, processor)
    observer = Observer()
    print("************************ ",IMAGE_DIR)
    observer.schedule(event_handler, str(IMAGE_DIR), recursive=True)
    observer.start()
    print("[👁️ File Observer Started]")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()




# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler
# import time
# from pathlib import Path
# from utils import load_model
# from cache_manager import update_cache_with_new_image
# from config import IMAGE_DIR
# from database import insert_metadata
# from datetime import datetime
# from captioner import generate_caption

# class ImageEventHandler(FileSystemEventHandler):
#     def __init__(self, model, processor):
#         self.model = model
#         self.processor = processor

#     def on_created(self, event):
#         if event.is_directory:
#             return
#         path = Path(event.src_path)
#         if path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
#             print(f"[📥 New Image Detected] {path.name}")
#             update_cache_with_new_image(path, self.model, self.processor)
#             caption = generate_caption(path)
#             insert_metadata(
#                 filename=str(path.resolve()),
#                 timestamp=datetime.now().isoformat(),
#                 camera_id=None,
#                 caption=caption,
#                 embedding_path=None
#             )

# def start_observer(model, processor):
#     event_handler = ImageEventHandler(model, processor)
#     observer = Observer()
#     observer.schedule(event_handler, str(IMAGE_DIR), recursive=False)
#     observer.start()
#     print("[👁️ File Observer Started]")
#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         observer.stop()
#     observer.join()
