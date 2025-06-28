import os
import shutil
from datetime import datetime

SOURCE_DIR = "/home/debjit/spacy/image_search_project_v2/images"
DEST_BASE_DIR = "/home/debjit/spacy/image_search_project_v2/images"

# Use today's date as the folder name
today = datetime.today().strftime("%Y-%m-%d")
DEST_DIR = os.path.join(DEST_BASE_DIR, today)
os.makedirs(DEST_DIR, exist_ok=True)

# Counter to simulate 10 camera IDs
camera_count = 10
image_counter = 0

for filename in sorted(os.listdir(SOURCE_DIR)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    src_path = os.path.join(SOURCE_DIR, filename)
    cam_id = f"cam_{(image_counter % camera_count) + 1:03d}"
    name_base = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1].lower()

    new_filename = f"{cam_id}_violation_{name_base}{ext}"
    dest_path = os.path.join(DEST_DIR, new_filename)

    shutil.move(src_path, dest_path)
    image_counter += 1

print(f"✅ Moved {image_counter} images into {DEST_DIR}")
