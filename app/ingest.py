import os
import time
from pathlib import Path
from config import DEFAULT_IMAGE_DIR, POLL_INTERVAL_SEC, CACHE_PATH
from cache_manager import add_new_images_to_cache
from utils import is_image_file, get_date_folder_name
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Ingestor")

def poll_folder_and_ingest():
    logger.info("📡 Starting folder polling loop...")
    seen_files = set()

    while True:
        current_files = set()

        for root, _, files in os.walk(DEFAULT_IMAGE_DIR):
            for file in files:
                if is_image_file(file):
                    full_path = os.path.join(root, file)
                    current_files.add(full_path)

        new_files = current_files - seen_files

        if new_files:
            logger.info(f"🆕 Detected {len(new_files)} new image(s). Processing...")
            add_new_images_to_cache(new_files, CACHE_PATH)
        else:
            logger.debug("No new files detected.")

        seen_files = current_files
        time.sleep(POLL_INTERVAL_SEC)

if __name__ == "__main__":
    poll_folder_and_ingest()
