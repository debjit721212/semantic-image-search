# scripts/add_to_db.py

import os
import sqlite3
from datetime import datetime
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.config import METADATA_DB, IMAGE_DIR

def insert_images_from_folder(date_folder):
    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()

    inserted = 0
    for filename in sorted(os.listdir(date_folder)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        full_path = os.path.abspath(os.path.join(date_folder, filename))

        # Extract camera_id from filename
        parts = filename.split("_")
        if len(parts) < 3:
            continue
        camera_id = parts[0]
        timestamp = datetime.now().isoformat()  # ISO format to match DB

        # Check for duplicates
        cursor.execute("SELECT 1 FROM image_metadata WHERE filename = ?", (full_path,))
        if cursor.fetchone():
            continue

        cursor.execute(
            "INSERT INTO image_metadata (filename, timestamp, camera_id) VALUES (?, ?, ?)",
            (full_path, timestamp, camera_id)
        )
        inserted += 1

    conn.commit()
    conn.close()
    print(f"✅ Inserted {inserted} new records into DB.")

if __name__ == "__main__":
    target_folder = os.path.join(IMAGE_DIR, "2025-06-25")
    insert_images_from_folder(target_folder)
