import os
import shutil
import time
import random
from datetime import datetime
from PIL import Image

# Source and destination directories
SOURCE_DIR = "/home/debjit/spacy/image_search_from_folder/images/archive/images"
DEST_BASE_DIR = "/home/debjit/Videos/debjit_project/smart_search/semantic-image-search/images"
today = datetime.today().strftime("%Y-%m-%d")
DEST_DIR = os.path.join(DEST_BASE_DIR, today)
os.makedirs(DEST_DIR, exist_ok=True)

# Track already pushed files
pushed_set = set(os.listdir(DEST_DIR))

camera_count = 10
image_counter = 0

all_images = sorted([
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# Optionally, randomize camera IDs for more realism
camera_ids = [f"cam_{i:03d}" for i in range(1, camera_count + 1)]

for filename in all_images:
    name_base = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1].lower()
    cam_id = random.choice(camera_ids)
    new_filename = f"{cam_id}_violation_{name_base}{ext}"

    if new_filename in pushed_set:
        continue

    src_path = os.path.join(SOURCE_DIR, filename)
    dest_path = os.path.join(DEST_DIR, new_filename)
    temp_path = dest_path + ".tmp"

    # Validate before copying
    try:
        img = Image.open(src_path)
        img.verify()
    except Exception as e:
        print(f"Skipping {filename}: {e}")
        continue

    # Atomic move: copy to temp, then rename
    shutil.copy2(src_path, temp_path)
    os.rename(temp_path, dest_path)
    print(f"✅ Pushed {new_filename}")
    pushed_set.add(new_filename)
    image_counter += 1

    # Simulate real-time ingestion (optional)
    time.sleep(20)  # ⏱️ wait 20 seconds before pushing next

print("✅ All images pushed to observer folder.")