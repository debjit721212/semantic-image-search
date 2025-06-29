import os
import shutil
import time
from datetime import datetime
from app.database import save_metadata  # Make sure this is implemented

SOURCE_DIR = "/home/debjit/spacy/image_search_project_v2/images"
DEST_BASE_DIR = "/home/debjit/spacy/image_search_project_v2/images"
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

for filename in all_images:
    name_base = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1].lower()

    # Generate final filename for DEST_DIR
    cam_id = f"cam_{(image_counter % camera_count) + 1:03d}"
    new_filename = f"{cam_id}_violation_{name_base}{ext}"

    if new_filename in pushed_set:
        continue  # Skip already pushed

    src_path = os.path.join(SOURCE_DIR, filename)
    dest_path = os.path.join(DEST_DIR, new_filename)

    shutil.copy2(src_path, dest_path)  # ✅ keep original

    # Save metadata
    save_metadata({
        "filename": dest_path,
        "timestamp": datetime.now().isoformat(),
        "camera_id": cam_id,
        "confidence": round(0.6 + 0.4 * (image_counter % 10) / 10.0, 2)
    })

    print(f"✅ Pushed {new_filename}")
    image_counter += 1
    time.sleep(20)  # ⏱️ wait 20 seconds before pushing next
