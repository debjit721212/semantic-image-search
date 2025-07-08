import os
import shutil
import time
from datetime import datetime
import subprocess

SOURCE_DIR = "/home/debjit/Videos/debjit_project/smart_search/semantic-image-search/video_chunks"
DEST_BASE_DIR = "/home/debjit/Videos/debjit_project/smart_search/semantic-image-search/images"
today = datetime.today().strftime("%Y-%m-%d")
DEST_DIR = os.path.join(DEST_BASE_DIR, today)
os.makedirs(DEST_DIR, exist_ok=True)

pushed_set = set(os.listdir(DEST_DIR))
camera_count = 10
video_counter = 0

def is_valid_video(path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return result.returncode == 0 and result.stdout.strip() != ""
    except Exception:
        return False

all_videos = sorted([
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
])

for filename in all_videos:
    name_base = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1].lower()
    cam_id = f"cam_{(video_counter % camera_count) + 1:03d}"
    now_str = datetime.now().strftime("%H-%M-%S")
    new_filename = f"{cam_id}_violation_{name_base}_{now_str}{ext}"

    if new_filename in pushed_set:
        continue

    src_path = os.path.join(SOURCE_DIR, filename)
    dest_path = os.path.join(DEST_DIR, new_filename)
    temp_path = dest_path + ".tmp"

    # Validate video before copying
    if not is_valid_video(src_path):
        print(f"❌ Skipping invalid/corrupt video: {filename}")
        continue

    # Copy to temp file first
    shutil.copy2(src_path, temp_path)
    # Only after copy is complete, rename to final filename
    os.rename(temp_path, dest_path)

    print(f"✅ Pushed {new_filename}")
    pushed_set.add(new_filename)
    video_counter += 1
    time.sleep(20)  # Simulate real-time upload

print("✅ All videos pushed to observer folder.")