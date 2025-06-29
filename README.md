# 🎯 ClipCap Vision: Semantic Image Search + Captioning

> Empower surveillance and media intelligence with real-time CLIP-based image search and BLIP-powered natural language captioning.

![App Screenshot](./doc/Screenshot%20from%202025-06-29%2017-28-17.png)

---

## 🚀 Features

- 🔎 **Prompt-Based Semantic Search**
  - Natural language queries like:
    - `no helmet in last 10 minutes from camera_1`
    - `person wearing red shirt`
    - `bike near entrance past hour`

- 📸 **Real-time Image Indexing**
  - Auto-monitors a folder for new images, updates the FAISS index instantly.

- 🧠 **CLIP + BLIP**
  - Uses CLIP for semantic similarity and BLIP for caption generation.

- 📊 **Live Violation Dashboard**
  - Hourly trends, camera-wise distribution, and confidence histograms.

- ⚙️ **Streamlit UI**
  - Modern, intuitive interface for search, upload, captioning, and analytics.

---

## 🧭 Project Structure

```
semantic-image-search/
├── app/
│   ├── main_app.py           # Streamlit interface
│   ├── observer.py           # Watchdog-based live folder monitor
│   ├── search.py             # CLIP-based similarity logic
│   ├── caption.py            # BLIP caption generator
│   ├── config.py             # Configurations (paths, thresholds, etc.)
│   ├── cache_manager.py      # FAISS index + npz embedding cache
│   ├── database.py           # SQLite metadata manager
│   ├── analytics_dashboard.py # Violation dashboard
│   └── utils.py              # Timestamp filters, helpers
│
├── scripts/                  # CLI tools for dev/test
│   ├── start_watcher.py
│   ├── build_cache.py
│   ├── clean_old_images.py
│   └── add_to_db.py
│
├── data/
│   ├── cache/                # FAISS index + embeddings_cache.npz
│   └── metadata.db           # SQLite metadata
│
├── images/                   # Your indexed images go here
├── doc/                      # README assets, screenshots, etc.
└── requirements.txt
```

---

## 📊 Live Violation Dashboard

![Dashboard](./doc/Screenshot%20from%202025-06-29%2017-16-17.png)

- Filter by camera ID or hours
- Hourly violation bar chart
- Camera usage pie chart
- Confidence histogram
- Full metadata table

---

## 🛠️ Getting Started

```bash
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
```

---

## 🐳 Docker Support

```bash
# Build image
docker build -t clipcap-vision .

# Run the app
docker run -p 8501:8501 -v $(pwd)/images:/app/images clipcap-vision

# Or use docker-compose (if needed)
docker-compose up --build

# Stop it
docker-compose down
```

---

## 🧪 Future Enhancements

- ✅ REST API for remote querying
- ✅ Docker + deployment automation
- 🔲 Multi-modal alert system
- 🔲 Replace SQLite with Qdrant or Weaviate
- 🔲 Live camera stream captioning

---

## 🙌 Acknowledgements

- OpenAI CLIP
- Salesforce BLIP
- Streamlit
- FAISS

---

## 📜 License

MIT License — See LICENSE

Built with ❤️ by [@debjit721212](https://github.com/debjit721212)