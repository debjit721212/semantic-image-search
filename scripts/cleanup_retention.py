import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
from utils import delete_embeddings_by_image_paths
from config import IMAGE_DIR, FRAMES_DIR, IMAGE_RETENTION_DAYS

logging.basicConfig(level=logging.INFO)

def find_old_files(root_dir, days_old):
    """
    Returns a list of file paths older than days_old in root_dir (recursively).
    """
    cutoff = datetime.now() - timedelta(days=days_old)
    old_files = []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
                if mtime < cutoff:
                    old_files.append(fpath)
            except Exception as e:
                logging.warning(f"Could not check {fpath}: {e}")
    return old_files

def delete_files(file_paths):
    """
    Deletes files from disk.
    """
    deleted = []
    for f in file_paths:
        try:
            os.remove(f)
            deleted.append(f)
            logging.info(f"Deleted file: {f}")
        except Exception as e:
            logging.warning(f"Failed to delete {f}: {e}")
    return deleted

def cleanup_empty_dirs(root_dir):
    """
    Removes empty directories under root_dir.
    """
    for root, dirs, files in os.walk(root_dir, topdown=False):
        if not dirs and not files:
            try:
                os.rmdir(root)
                logging.info(f"Removed empty folder: {root}")
            except Exception as e:
                logging.warning(f"Could not remove folder: {root}: {e}")

def cleanup_retention():
    days = IMAGE_RETENTION_DAYS
    logging.info(f"Starting cleanup for files older than {days} days...")

    # 1. Find and delete old images
    old_images = find_old_files(IMAGE_DIR, days)
    deleted_images = delete_files(old_images)
    if deleted_images:
        delete_embeddings_by_image_paths(deleted_images)

    # 2. Find and delete old frames
    old_frames = find_old_files(FRAMES_DIR, days)
    deleted_frames = delete_files(old_frames)
    if deleted_frames:
        delete_embeddings_by_image_paths(deleted_frames)

    # 3. (Optional) Add video cleanup if you store videos in a separate folder
    # VIDEO_DIR = ...
    # old_videos = find_old_files(VIDEO_DIR, days)
    # deleted_videos = delete_files(old_videos)
    # if deleted_videos:
    #     delete_embeddings_by_image_paths(deleted_videos)

    # 4. Cleanup empty directories
    cleanup_empty_dirs(IMAGE_DIR)
    cleanup_empty_dirs(FRAMES_DIR)
    # cleanup_empty_dirs(VIDEO_DIR)

    logging.info("Cleanup finished.")

if __name__ == "__main__":
    cleanup_retention()