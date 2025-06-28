import logging
from app.cleanup import remove_old_images
from app.utils import setup_logger


def main():
    setup_logger("clean_old_images.log")
    logging.info("[CleanupCLI] Running image cleanup manually...")
    removed_count = remove_old_images()
    logging.info(f"[CleanupCLI] Removed {removed_count} old images.")


if __name__ == "__main__":
    main()