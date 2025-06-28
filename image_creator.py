import os
import shutil
import time
from datetime import datetime

SOURCE_DIR = "/home/debjit/spacy/image_search_from_folder/images/archive/images"  # 🔁 Replace with actual path
DEST_BASE_DIR = "/home/debjit/spacy/image_search_project_v2/images"  # 🔁 Replace with actual path

# ✅ Ensure DEST_DIR is date-based
today = datetime.today().strftime("%Y-%m-%d")
DEST_DIR = os.path.join(DEST_BASE_DIR, today)
os.makedirs(DEST_DIR, exist_ok=True)

# ✅ Create backup dir inside source
BACKUP_DIR = os.path.join(SOURCE_DIR, "backup")
os.makedirs(BACKUP_DIR, exist_ok=True)

camera_count = 10
counter = 0

for filename in sorted(os.listdir(SOURCE_DIR)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    if filename in os.listdir(BACKUP_DIR):  # ✅ FIXED: Check inside full backup path
        continue

    src_path = os.path.join(SOURCE_DIR, filename)
    cam_id = f"cam_{(counter % camera_count) + 1:03d}"
    name_base = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1].lower()
    new_name = f"{cam_id}_violation_{name_base}{ext}"
    dest_path = os.path.join(DEST_DIR, new_name)

    try:
        # Backup
        shutil.copy(src_path, os.path.join(BACKUP_DIR, filename))

        # Send to monitored folder
        shutil.copy(src_path, dest_path)

        print(f"📦 Sent: {new_name}")
        time.sleep(40)
        counter += 1

    except Exception as e:
        print(f"❌ Error with {filename}: {e}")