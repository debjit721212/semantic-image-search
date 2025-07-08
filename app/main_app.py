import streamlit as st
from PIL import Image, UnidentifiedImageError
import os
import re
from datetime import datetime, timedelta
from config import IMAGE_DIR, METADATA_DB
from search import perform_search
from database import init_db, get_all_metadata, get_recent_metadata
from utils import chroma_collection, call_llm_with_context, load_lora_model
import threading
from observer import start_observer
from captioner import generate_caption, load_blip_model
import schedule
import time
import sys
import logging
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.cleanup_retention import cleanup_retention
from transformers import pipeline

# --- Initialize DB ---
os.makedirs(os.path.dirname(METADATA_DB), exist_ok=True)
init_db()

# --- Logging Setup ---
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# --- Moderation Setup ---
BANNED_KEYWORDS = [
    "nude", "naked", "sex", "sexual", "porn", "erotic", "genital", "breast", "penis", "vagina", "rape", "child porn", "abuse"
]
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert", device=-1)  # CPU

def is_toxic(query, threshold=0.5):
    try:
        result = toxicity_classifier(query)[0]
        return result['label'] == 'toxic' and result['score'] > threshold
    except Exception as e:
        logging.error(f"Toxicity model error: {e}")
        return False

def is_safe_query(query):
    query_lower = query.lower()
    for word in BANNED_KEYWORDS:
        if word in query_lower:
            logging.warning(f"Blocked by banned word: {query}")
            return False
    if is_toxic(query):
        logging.warning(f"Blocked by toxicity model: {query}")
        return False
    return True

# --- Singleton Model Loading ---
if "clip_lora_model" not in st.session_state:
    st.session_state["clip_lora_model"], st.session_state["clip_processor"], st.session_state["clip_device"] = load_lora_model()
model = st.session_state["clip_lora_model"]
processor = st.session_state["clip_processor"]
device = st.session_state["clip_device"]

if "blip_model" not in st.session_state or "blip_processor" not in st.session_state:
    st.session_state["blip_model"], st.session_state["blip_processor"] = load_blip_model(device="cpu")

# --- Observer Setup ---
if "observer_started" not in st.session_state:
    observer_thread = threading.Thread(
        target=start_observer, args=(model, processor), daemon=True
    )
    observer_thread.start()
    st.session_state["observer_started"] = True

# --- SCHEDULER SETUP ---
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)

if "scheduler_started" not in st.session_state:
    schedule.every().day.at("02:00").do(cleanup_retention)
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    st.session_state["scheduler_started"] = True

st.set_page_config(page_title="Image & Video Search", layout="wide")
st.title("🔍 ClipCap Vision: Semantic Image & Video Search + Captioning")
st.caption("Empower surveillance and media with AI-driven image/video search and natural captioning using CLIP and BLIP.")

# --- Utility: CLIP/LoRA Text Embedding ---
def compute_text_embedding(prompt, model, processor, device):
    inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    return text_embedding[0].cpu().tolist()

# --- Utility: Display Results Grid ---
def display_results_grid(results, images_per_row=2):
    container = st.container()
    with container:
        for i in range(0, len(results), images_per_row):
            row = st.columns(images_per_row)
            for j in range(images_per_row):
                if i + j < len(results):
                    path, score, meta = results[i + j]
                    if not os.path.exists(path):
                        row[j].warning(f"❌ Missing File: {path}")
                        continue
                    try:
                        img = Image.open(path).resize((640, 360))
                        timestamp = meta.get("timestamp", "N/A")
                        camera_id = meta.get("camera_id", "Unknown")
                        caption_text = f"Score: {score:.3f}\n📷 {camera_id} | 🕒 {timestamp}"
                        row[j].image(img, caption=caption_text, use_container_width=True)
                    except Exception as e:
                        row[j].warning(f"Error loading image: {e}")

# --- Utility: Parse time & camera from prompt ---
def parse_time_and_camera(prompt):
    prompt = prompt.lower()
    minutes_match = re.search(r"(last|past)?\s*(\d+)\s*(min|mins|minute|minutes)\b", prompt)
    hours_match = re.search(r"(last|past)?\s*(\d+)\s*(hr|hrs|hour|hours)\b", prompt)
    ago_minutes = re.search(r"(\d+)\s*(min|mins|minute|minutes)\s*ago", prompt)
    ago_hours = re.search(r"(\d+)\s*(hr|hrs|hour|hours)\s*ago", prompt)
    delta = timedelta()
    if minutes_match:
        delta = timedelta(minutes=int(minutes_match.group(2)))
    elif hours_match:
        delta = timedelta(hours=int(hours_match.group(2)))
    elif ago_minutes:
        delta = timedelta(minutes=int(ago_minutes.group(1)))
    elif ago_hours:
        delta = timedelta(hours=int(ago_hours.group(1)))
    camera_match = re.search(r"(camera|cam)[_\s]*(\d+)", prompt)
    camera_id = f"camera_{camera_match.group(2)}" if camera_match else None
    return delta, camera_id

# --- Upload Section ---
st.sidebar.header("📤 Upload Images")
uploaded_files = st.sidebar.file_uploader("Upload images (jpg/png)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        try:
            img = Image.open(file)
            img.verify()
            file.seek(0)
            with open(os.path.join(IMAGE_DIR, file.name), "wb") as f:
                f.write(file.read())
            st.sidebar.image(img.resize((120, 80)), caption=file.name)
        except UnidentifiedImageError:
            st.sidebar.error(f"Invalid image skipped: {file.name}")
    st.sidebar.success("✅ Uploaded successfully!")

# --- Tabs for Image Search and Video Search ---
tab1, tab2 = st.tabs(["Image Search", "Video Search"])

with tab1:
    st.subheader("🔎 Image Search")
    prompt = st.text_input("Enter image search prompt", placeholder="e.g., a person without helmet", key="image_search")
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.22, 0.01, key="image_threshold")

    if st.button("Search", key="image_search_btn") and prompt.strip():
        with st.spinner("Searching..."):
            query_embedding = compute_text_embedding(prompt, model, processor, device)
            results = chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=["metadatas", "distances"]
            )
            if not results or not results.get("metadatas") or not any(results["metadatas"][0]):
                st.warning("No matches found above threshold.")
            else:
                st.success(f"Found {len(results['metadatas'][0])} results")
                display_results_grid([
                    (meta.get("image_path", ""), dist, meta)
                    for meta, dist in zip(results["metadatas"][0], results["distances"][0])
                    if dist >= threshold and meta.get("image_path", "") and os.path.isfile(meta.get("image_path", ""))
                ])

with tab2:
    st.subheader("🎥 Video Search")
    video_query = st.text_input("Enter video search prompt", key="video_search")
    if st.button("Search Videos"):
        query_embedding = compute_text_embedding(video_query, model, processor, device)
        video_results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["metadatas", "distances"]
        )
        if not video_results or not video_results.get("metadatas") or not any(video_results["metadatas"][0]):
            st.warning("No matching videos found.")
        else:
            for meta in video_results["metadatas"][0]:
                st.video(meta.get("video_path", ""))
                # Defensive check for key frames
                key_frames_list = []
                if "key_frames" in meta and meta["key_frames"]:
                    if isinstance(meta["key_frames"], str):
                        key_frames_list = [f for f in meta["key_frames"].split(",") if f.strip()]
                    else:
                        key_frames_list = meta["key_frames"]
                if key_frames_list and os.path.isfile(key_frames_list[0]):
                    st.image(key_frames_list[0], caption="Preview Frame")
                else:
                    st.warning("No valid preview frame available.")
                st.write(f"Camera: {meta.get('camera_id', 'N/A')}, Duration: {meta.get('duration', 'N/A')}")
                st.write(f"Summary: {meta.get('summary_caption', '')}")

# --- RAG Q&A Section with Moderation and LLM ---
rag_llm = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)

def call_llm_with_context(user_question, context_captions):
    context = "\n".join(context_captions)
    prompt = (
        f"Based on the following surveillance events, answer the question: '{user_question}'\n\n"
        f"Events:\n{context}\n\n"
        "Answer:"
    )
    try:
        response = rag_llm(prompt)
        return response[0]["generated_text"].strip()
    except Exception as e:
        logging.error(f"LLM error: {e}")
        return "⚠️ AI error: Unable to generate answer at this time."

# --- Q&A Section with Moderation and Chat History ---
st.subheader("🤖 Ask a Question (RAG Q&A)")
if "qa_history" not in st.session_state:
    st.session_state["qa_history"] = []

user_question = st.text_input("Ask a question about your surveillance data (RAG):")

if st.button("Ask (RAG)") and user_question.strip():
    if not is_safe_query(user_question):
        response = "❌ This type of search is not allowed."
        st.session_state["qa_history"].append(("You", user_question))
        st.session_state["qa_history"].append(("AI", response))
    else:
        with st.spinner("Retrieving and generating answer..."):
            query_embedding = compute_text_embedding(user_question, model, processor, device)
            results = chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=["metadatas", "distances"]
            )
            st.session_state["qa_history"].append(("You", user_question))
            if results and results["metadatas"] and any(results["metadatas"][0]):
                context_captions = [meta.get("caption", "") for meta in results["metadatas"][0]]
                llm_answer = call_llm_with_context(user_question, context_captions)
                st.session_state["qa_history"].append(("AI", llm_answer))
                for meta in results["metadatas"][0]:
                    # Defensive: show image if available, else video if available
                    image_path = meta.get("image_path")
                    if image_path and os.path.isfile(image_path):
                        st.image(image_path, caption=meta.get("caption", ""))
                    elif meta.get("video_path"):
                        st.video(meta["video_path"])
                    else:
                        st.warning("No valid image or video to display.")
            else:
                response = "No relevant results found. Please ask a question related to surveillance events, people, or objects."
                st.session_state["qa_history"].append(("AI", response))

# --- Display Q&A Chat History ---
if st.session_state["qa_history"]:
    st.markdown("### 💬 Q&A History")
    for speaker, message in st.session_state["qa_history"]:
        if speaker == "You":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**AI:** {message}")

# --- Captioning Section ---
st.subheader("🖼️ Image Captioning")
caption_files = st.file_uploader("Upload images for captioning", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="caption")
if caption_files:
    for file in caption_files:
        try:
            img = Image.open(file).convert("RGB")
            caption = generate_caption(img)
            st.image(img, caption=f"📝 {caption}", use_container_width=True)
        except Exception as e:
            st.error(f"Error with {file.name}: {e}")

# --- Metadata Viewer ---
st.sidebar.markdown("## 🔍 Metadata Viewer")
if st.sidebar.button("📄 View Image Metadata"):
    metadata = get_all_metadata()
    if metadata is not None:
        st.write("### 📄 Image Metadata Table")
        st.dataframe(metadata, use_container_width=True)
    else:
        st.info("No metadata available yet.")

# --- Time + Camera Prompt-Based Filter ---
st.subheader("🕒 Search Recent Violations")
recent_prompt = st.text_input("Prompt (e.g. 'last 10 minutes noHelmet from camera_1')")

if st.button("Search Recent"):
    with st.spinner("Filtering recent images..."):
        delta, camera_filter = parse_time_and_camera(recent_prompt)
        if not delta:
            st.error("❌ Couldn't parse time window from prompt.")
        else:
            df = get_recent_metadata(hours=delta.total_seconds() / 3600)
            if camera_filter:
                df = df[df["camera_id"].astype(str).str.contains(camera_filter, case=False, na=False)]
            recent_files = df["filename"].tolist()
            if not recent_files:
                st.warning("No recent images found.")
            else:
                query_embedding = compute_text_embedding(recent_prompt, model, processor, device)
                results = chroma_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=5,
                    include=["metadatas", "distances"]
                )
                if results and results["metadatas"] and any(results["metadatas"][0]):
                    display_results_grid([
                        (meta.get("image_path", ""), dist, meta)
                        for meta, dist in zip(results["metadatas"][0], results["distances"][0])
                        if dist >= threshold and meta.get("image_path", "") and os.path.isfile(meta.get("image_path", ""))
                    ])
                else:
                    st.warning("No matching results found.")