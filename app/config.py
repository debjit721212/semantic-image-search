import os
from pathlib import Path
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "metadata.db"
# DB_PATH = os.path.join("data", "metadata.db")


# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "images"
CACHE_DIR = BASE_DIR / "data" / "cache"
LOG_DIR = BASE_DIR / "data" / "logs"
METADATA_DB = BASE_DIR / "data" / "metadata.db"

# Runtime parameters
CACHE_PATH = CACHE_DIR / "embeddings_cache.npz"
RETENTION_DAYS = 7
SIMILARITY_THRESHOLD = 0.3
TOP_K_RESULTS = 5
IMAGE_RETENTION_DAYS = 30

# Model settings
# CLIP_MODEL_NAME = "ViT-L/14"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# Device settings
DEVICE = "cuda" if os.environ.get("USE_CUDA", "1") == "1" and torch.cuda.is_available() else "cpu"






