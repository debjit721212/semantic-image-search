# 🔍 semantic-image-search

**AI-powered semantic image search with live indexing and natural language prompts.**

This project enables real-time, natural-language-driven search across a folder of images — ideal for video surveillance and security analytics applications.

---

## ✨ Features

-  **Live Image Indexing**: New images are auto-monitored and indexed from a watched folder.
-  **CLIP-powered Semantics**: Uses [OpenAI CLIP](https://github.com/openai/CLIP) to understand both text and images.
-  **Fast Similarity Search**: Cosine similarity over cached image embeddings for quick lookup.
-  **Flexible Prompt Support**: Supports rich queries like:
  - `no helmet in last 10 minutes from camera_3`
  - `person wearing red shirt`
  - `bike near entrance from camera_1 past hour`
-  **Streamlit UI**: Clean interface to test prompt-based search visually.
-  **Metadata-aware**: Filters based on image timestamp, camera ID, and auto-generated captions.

---

## 📌 Use Case

Built for intelligent surveillance, monitoring safety violations, or any real-time image retrieval scenario based on human-readable prompts.

---

## 🛠️ Getting Started

```bash
# Clone the repository
git clone https://github.com/debjit721212/semantic-image-search.git
cd semantic-image-search

# (Optional) Create a virtual environment
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app/main_app.py
Make sure your images and embeddings folder are configured inside config.py.

🧭 Project Structure
bash
Copy
Edit
semantic-image-search/
├── app/
│   ├── main_app.py           # Streamlit UI for prompt-based search
│   ├── search.py             # Semantic CLIP-based search engine logic
│   ├── observer.py           # Watches image directory and updates cache
│   ├── cache_manager.py      # Handles loading/saving image embeddings
│   ├── captioner.py          # (Optional) Image captioning logic
│   ├── caption.py            # Generates captions for images
│   ├── config.py             # Project-wide configurations
│   ├── utils.py              # Utility functions (time filtering, etc.)
│   ├── database.py           # Handles SQLite metadata storage
│   ├── ingest.py             # Image ingestion logic for testing
│   ├── cleanup.py            # Cleans old images and updates DB/cache
│   ├── test_database.py      # Simple test to inspect the database
│   ├── just_test.py          # Miscellaneous test script
│   └── __pycache__/          # Python bytecode (ignored in Git)
│
├── scripts/
│   ├── build_cache.py        # Build cache of embeddings from image folder
│   ├── clean_old_images.py   # Clean images older than N days
│   ├── inspact_acche.py      # Inspect cache file contents
│   ├── add_to_db.py          # Add metadata from image files to SQLite
│   └── start_watcher.py      # Script to start file observer
│
├── data/
│   ├── cache/                # Stores embeddings_cache.npz
│   └── metadata.db           # SQLite DB with image metadata
│
├── dummy_image_generator.py  # Script to create test images
├── image_creator.py          # Creates sample captioned/test images
├── README.md 
🚀 Future Plans
 Replace SQLite with vector database (e.g., FAISS / Qdrant).

 Integrate real-time video stream captioning + analysis.

 Support visual filters (e.g., bounding box-based search).

 Dockerized deployment & REST API.

 Multi-camera dashboard for alerts & analytics.

🙌 Acknowledgments
Built with:

OpenAI CLIP

Streamlit

PyTorch

📜 License
MIT License – see LICENSE file for details.
