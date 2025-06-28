import numpy as np
from pathlib import Path

CACHE_PATH = Path("/home/debjit/spacy/image_search_project_v2/data/cache/embeddings_cache.npz")

if not CACHE_PATH.exists():
    print("❌ Cache file not found!")
    exit()

data = np.load(CACHE_PATH, allow_pickle=True)
paths = data["paths"]
embeddings = data["embeddings"]

print(f"✅ Loaded cache from: {CACHE_PATH}")
print(f"🖼️  Total images: {len(paths)}")
print(f"📐 Embeddings shape: {embeddings.shape}")

# Check dimensionality
if embeddings.shape[1] != 512:
    print(f"⚠️ WARNING: Expected 512-d embeddings, got {embeddings.shape[1]}-d")
else:
    print("✅ Embedding dimensions are correct (512)")

# Show sample entries
print(f"\n🔍 Sample path: {paths[0]}")
print(f"🔢 Sample embedding (first 5 dims): {embeddings[0][:5]}")
