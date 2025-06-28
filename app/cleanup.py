
import os
import time
import logging
from datetime import datetime, timedelta
from config import IMAGE_DIR, IMAGE_RETENTION_DAYS
from database import delete_metadata_for_images
from cache_manager import rebuild_cache_without_files

from pathlib import Path

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cleanup_old_images()
