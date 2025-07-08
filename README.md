# 🔎 ClipCap Vision: Semantic Image & Video Search + Captioning

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-ready-brightgreen)](https://streamlit.io/)

> **AI-powered, real-time search, captioning, and analytics for surveillance, media, and enterprise.**
>  
> **Find, caption, and analyze both images and videos using CLIP+LoRA, BLIP, and a modern RAG Q&A pipeline.**

---

## 🚀 Features

- 🔎 **Multi-Modal Semantic Search**  
  - Natural language queries for both images and videos  
  - Example:  
    - `no helmet in last 10 minutes from camera_1`
    - `person with a dog`
    - `show me video of a man in a red shirt`

- 📸 **Real-Time Ingestion & Indexing**  
  - Auto-monitors folders for new images and videos  
  - Extracts frames from videos, tags with CLIP/LoRA + BLIP

- 🤖 **RAG Q&A (Retrieval-Augmented Generation)**  
  - Ask questions about your data (images or videos)  
  - Centralized, context-aware answers (not just search)

- 📊 **Live Analytics Dashboard**  
  - Hourly trends, camera-wise distribution, confidence histograms  
  - Video analytics: violations per video, duration, event timeline

- ⚙️ **Modern Streamlit UI**  
  - Tabs for image and video search  
  - Upload, search, caption, and analytics in one place

- 🏗️ **Production-Ready Pipeline**  
  - ChromaDB for fast, scalable vector search  
  - SQLite for metadata/analytics  
  - Docker support for easy deployment

---

## 🧠 How It Works

[Images/Videos] → [Observer] → [Frame Extraction] → [CLIP/LoRA + BLIP Tagging] ↓ [ChromaDB (Embeddings + Captions)] ← [RAG Q&A] ← [User Query] ↓ [Streamlit UI: Search, Q&A, Analytics] ↓ [SQLite: Metadata, Analytics]


- **Images and videos are ingested in real time.**
- **Videos are split into frames, each frame is tagged and indexed.**
- **All embeddings and captions are stored in ChromaDB for fast semantic search.**
- **Metadata is stored in SQLite for analytics and reporting.**
- **RAG Q&A lets you ask questions about your data and get context-aware answers.**
- **Streamlit UI provides a modern, user-friendly interface for search, upload, and analytics.**

---

## 🏋️‍♂️ Model Training Pipeline

- **CLIP + LoRA**:  
  - Pretrained on Flickr30k for general vision-language grounding  
  - Fine-tuned on your custom surveillance/violation dataset for domain adaptation

- **BLIP**:  
  - Used for automatic, high-quality captioning of both images and video frames

- **RAG Q&A**:  
  - Centralized, multi-modal search and Q&A over both images and videos

---

## 🖼️ Screenshots

### Image Search
![Image Search UI](./doc/image_search_ui.png)

### Video Search
![Video Search UI](./doc/video_search_ui.png)

### Analytics Dashboard
![Analytics Dashboard](./doc/analytics_dashboard.png)

### Top Violation Types
![Top Violation Types](./doc/violation_types.png)

### Video Event Timeline
![Video Event Timeline](./doc/video_event_timeline.png)

---

## 🧭 Project Structure

```text
semantic-image-search/
├── app/
│   ├── main_app.py           # Streamlit UI (image/video search, Q&A)
│   ├── observer.py           # Real-time file/video watcher
│   ├── video_chunker.py      # Frame extraction + video metadata
│   ├── frame_tagger.py       # Frame tagging (CLIP/LoRA + BLIP)
│   ├── database.py           # SQLite metadata manager
│   ├── analytics_dashboard.py # Analytics dashboard
│   └── utils.py              # Embedding, captioning, helpers
│
├── scripts/                  # CLI tools for dev/test
│   ├── dummy_image_violation_generator.py
│   ├── dummy_video_violation_generator.py
│   └── video_splitter.py
│
├── data/
│   ├── cache/
│   └── metadata.db
├── images/                   # Indexed images/videos
├── frames/                   # Extracted video frames
├── video_chunks/             # Video chunks for testing
├── doc/                      # README assets, screenshots, etc.
└── requirements.txt
🛠️ Getting Started
# 1. Clone the repository
git clone https://github.com/debjit721212/semantic-image-search.git
cd semantic-image-search

# 2. (Optional) Setup virtual environment
python3 -m venv env
source env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run app/main_app.py

🐳 Docker Support
# Build image
docker build -t clipcap-vision .

# Run the app
docker run -p 8501:8501 -v $(pwd)/images:/app/images clipcap-vision

# Or use docker-compose (if needed)
docker-compose up --build

# Stop it
docker-compose down

📊 Live Analytics Dashboard
Filter by camera ID or hours
Hourly violation bar chart
Camera usage pie chart
Confidence histogram
Video analytics: violations per video, duration, event timeline
Full metadata table
🧪 Future Enhancements
✅ REST API for remote querying
✅ Docker + deployment automation
🔲 Multi-modal alert system
🔲 Replace SQLite with Qdrant or Weaviate
🔲 Live camera stream captioning
🔲 Video event summarization with LLMs
🙌 Acknowledgements
OpenAI CLIP
Salesforce BLIP
Streamlit
ChromaDB
SQLite
Docker
📜 License
MIT License — See LICENSE

Built with ❤️ by @debjit721212

💡 Why This Project?
Solves real-world surveillance and media search problems
Handles both images and videos, with RAG Q&A
Production-ready, scalable, and open source
Modern, modular, and easy to extend

Try it, contribute, and make AI-powered vision search better for everyone!