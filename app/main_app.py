import streamlit as st
from PIL import Image, UnidentifiedImageError
import os
import re
from datetime import datetime, timedelta

from config import IMAGE_DIR
from search import perform_search
from cache_manager import ensure_cache
from database import  init_db,get_all_metadata, get_recent_metadata  #initialize_db,
from utils import load_model
import threading
from observer import start_observer
from captioner import generate_caption

# Initialize DB and cleanup
# initialize_db()
init_db()

if "cleanup_done" not in st.session_state:
    from utils import cleanup_old_images
    cleanup_old_images()
    st.session_state["cleanup_done"] = True

# Ensure image directory exists
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load CLIP model and start file observer
model, processor, device = load_model()
if "observer_started" not in st.session_state:
    observer_thread = threading.Thread(
        target=start_observer, args=(model, processor), daemon=True
    )
    observer_thread.start()
    st.session_state["observer_started"] = True

st.set_page_config(page_title="Image Search + Captioning", layout="wide")
st.title("🔍 CLIP Image Search + 🖼️ BLIP Captioning")
st.caption("Upload images, perform prompt-based image search, or generate automatic captions")
os.makedirs(IMAGE_DIR, exist_ok=True)

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

    # Try to extract time using various patterns
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

    # Camera match: camera_1, camera 1, cam_001, etc.
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

# --- Prompt-Based Search ---
st.subheader("🔎 Search with Prompt")
prompt = st.text_input("Enter search prompt", placeholder="e.g., a person without helmet")
threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.3, 0.01)

if st.button("Search") and prompt.strip():
    with st.spinner("Searching..."):
        ensure_cache(model, processor)
        results = perform_search(prompt, threshold, model, processor, device, return_metadata=True)
        if not results:
            st.warning("No matches found above threshold.")
        else:
            st.success(f"Found {len(results)} results")
            display_results_grid(results)

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
                results = perform_search(
                    prompt=recent_prompt,
                    threshold=threshold,
                    model=model,
                    processor=processor,
                    device=device,
                    file_list=recent_files,
                    return_metadata=True
                )
                if results:
                    display_results_grid(results)
                else:
                    st.warning("No matching results found.")
