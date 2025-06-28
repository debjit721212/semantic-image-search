import os
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.cache_manager import build_full_cache
from app.config import IMAGE_DIR, CACHE_PATH

def main():
    parser = argparse.ArgumentParser(description="Build or rebuild image embedding cache.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild cache even if it exists")
    args = parser.parse_args()

    if os.path.exists(CACHE_PATH):
        if args.rebuild:
            os.remove(CACHE_PATH)
            print("[CACHE] Existing cache removed. Rebuilding...")
        else:
            print("[CACHE] Cache already exists. Use --rebuild to force regeneration.")
            return

    build_full_cache()
    print("[CACHE] Cache build completed.")

if __name__ == "__main__":
    main()

