import time
import logging
from app.ingest import process_new_images
from app.config import IMAGE_DIR, SCAN_INTERVAL_SECS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_continuous_ingestion():
    logging.info("🚀 Starting continuous folder watcher for new images...")
    while True:
        try:
            added = process_new_images()
            if added:
                logging.info(f"✅ Ingested {added} new images.")
        except Exception as e:
            logging.error(f"❌ Error during ingestion: {e}")
        time.sleep(SCAN_INTERVAL_SECS)

if __name__ == "__main__":
    run_continuous_ingestion()