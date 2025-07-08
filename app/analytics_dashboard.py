import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from streamlit_autorefresh import st_autorefresh
from database import get_all_metadata, get_all_video_metadata

st.set_page_config(page_title="📊 Violation Dashboard", layout="wide")
st_autorefresh(interval=20000, key="refresh_dashboard")
st.title("📊 Violation Analytics Dashboard")

tab1, tab2 = st.tabs(["Image Analytics", "Video Analytics"])

# --- Image Analytics Tab ---
with tab1:
    data = get_all_metadata()
    if data is None or data.empty:
        st.warning("No metadata available.")
        st.stop()
    df = pd.DataFrame(data)
    if df.empty:
        st.warning("Metadata is empty.")
        st.stop()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["confidence"] = pd.to_numeric(df.get("confidence", 0), errors="coerce")
    df["camera_id"] = df.get("camera_id", "Unknown").fillna("Unknown")
    df = df.dropna(subset=["timestamp"])

    with st.sidebar:
        st.header("🔍 Filters")
        camera_filter = st.multiselect("Select Camera ID", options=df["camera_id"].unique(), default=df["camera_id"].unique())
        time_range = st.slider("Select Time Range (Hours)", 0, 24, (0, 24))

    start_hour, end_hour = time_range
    df = df[df["camera_id"].isin(camera_filter)]
    df = df[(df["timestamp"].dt.hour >= start_hour) & (df["timestamp"].dt.hour <= end_hour)]

    col1, col2, col3 = st.columns(3)

    # Chart 1: Hourly Violation Count
    with col1:
        st.markdown("### ⏰ Violations by Hour")
        hourly = df["timestamp"].dt.hour.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=hourly.index, y=hourly.values, hue=hourly.index, legend=False, ax=ax, palette="viridis")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Violations")
        ax.set_title("Hourly Violation Count")
        st.pyplot(fig)

    # Chart 2: Violations by Camera
    with col2:
        st.markdown("### 🎥 Camera Distribution")
        cam_counts = df["camera_id"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(cam_counts.values, labels=cam_counts.index, autopct="%1.1f%%", startangle=140)
        ax.axis("equal")
        st.pyplot(fig)

    # Chart 3: Confidence Distribution
    with col3:
        st.markdown("### 📈 Confidence Score Histogram")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["confidence"], bins=20, kde=True, ax=ax, color="skyblue")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Frequency")
        ax.set_title("Confidence Distribution")
        st.pyplot(fig)

    # --- Advanced Analytics ---
    st.markdown("### 🔥 Violation Heatmap (Hour vs Camera)")
    heatmap_df = df.copy()
    heatmap_df["hour"] = heatmap_df["timestamp"].dt.hour
    pivot = pd.pivot_table(heatmap_df, index="hour", columns="camera_id", values="id", aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
    ax.set_xlabel("Camera ID")
    ax.set_ylabel("Hour")
    ax.set_title("Violations Heatmap (Hour vs Camera)")
    st.pyplot(fig)

    st.markdown("### 📈 Violation Timeline")
    timeline = df.copy()
    timeline["date"] = timeline["timestamp"].dt.date
    daily_counts = timeline.groupby("date").size()
    fig, ax = plt.subplots(figsize=(8, 4))
    daily_counts.plot(ax=ax, marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Violations")
    ax.set_title("Violations Over Time")
    st.pyplot(fig)

    st.markdown("### 🏷️ Top Violation Types (from Captions)")
    from collections import Counter
    import itertools
    keywords = ["helmet", "vest", "phone", "mask", "smoking"]
    all_words = list(itertools.chain.from_iterable([str(c).lower().split() for c in df["caption"]]))
    violation_counts = {k: all_words.count(k) for k in keywords}
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=list(violation_counts.keys()), y=list(violation_counts.values()), ax=ax, palette="coolwarm")
    ax.set_xlabel("Violation Type")
    ax.set_ylabel("Count")
    ax.set_title("Top Violation Types")
    st.pyplot(fig)

    st.markdown("---")
    with st.expander("🗃️ View Raw Metadata Table"):
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)

# --- Video Analytics Tab ---
with tab2:
    st.header("🎥 Video Analytics")
    video_data = get_all_video_metadata()
    if video_data is None or video_data.empty:
        st.warning("No video metadata available.")
        st.stop()
    vdf = pd.DataFrame(video_data)
    if vdf.empty:
        st.warning("Video metadata is empty.")
        st.stop()

    vdf["start_time"] = pd.to_datetime(vdf["start_time"], errors="coerce")
    vdf["end_time"] = pd.to_datetime(vdf["end_time"], errors="coerce")
    vdf["camera_id"] = vdf.get("camera_id", "Unknown").fillna("Unknown")
    vdf["duration"] = (vdf["end_time"] - vdf["start_time"]).dt.total_seconds() / 60

    with st.sidebar:
        st.header("🎬 Video Filters")
        video_camera_filter = st.multiselect("Select Camera ID (Videos)", options=vdf["camera_id"].unique(), default=vdf["camera_id"].unique())

    vdf = vdf[vdf["camera_id"].isin(video_camera_filter)]

    vcol1, vcol2, vcol3 = st.columns(3)

    # Chart 1: Violations per Video (by key frame count)
    with vcol1:
        st.markdown("### 🎞️ Violations per Video (Key Frames)")
        vdf["num_key_frames"] = vdf["key_frames"].apply(lambda x: len(x.split(",")) if isinstance(x, str) and x else 0)
        top_videos = vdf.sort_values("num_key_frames", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=top_videos["num_key_frames"], y=top_videos["video_path"], ax=ax, palette="magma")
        ax.set_xlabel("Number of Key Frames")
        ax.set_ylabel("Video Path")
        ax.set_title("Top Videos by Violations (Key Frames)")
        st.pyplot(fig)

    # Chart 2: Video Duration Distribution
    with vcol2:
        st.markdown("### ⏳ Video Duration Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(vdf["duration"].dropna(), bins=20, kde=True, ax=ax, color="green")
        ax.set_xlabel("Duration (minutes)")
        ax.set_ylabel("Number of Videos")
        ax.set_title("Video Duration Distribution")
        st.pyplot(fig)

    # Chart 3: Top Videos by Summary Caption
    with vcol3:
        st.markdown("### 📝 Top Video Summaries")
        for idx, row in top_videos.iterrows():
            st.write(f"**{row['video_path']}**: {row['summary_caption']}")

    # --- Advanced Analytics ---
    st.markdown("### 🎬 Video Event Timeline")
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, row in vdf.iterrows():
        if row["key_frames"]:
            times = []
            for f in row["key_frames"].split(","):
                try:
                    sec_str = f.split("_")[-1].replace(".jpg", "").replace("-", ":")
                    h, m, s = [int(x) for x in sec_str.split(":")]
                    total_sec = h * 3600 + m * 60 + s
                    times.append(total_sec)
                except Exception:
                    continue
            ax.scatter(times, [row["video_path"]]*len(times), label=row["video_path"], s=20)
    ax.set_xlabel("Frame Timestamp (seconds)")
    ax.set_ylabel("Video")
    ax.set_title("Key Events in Videos")
    st.pyplot(fig)

    st.markdown("---")
    with st.expander("🗃️ View Raw Video Metadata Table"):
        st.dataframe(vdf.sort_values("start_time", ascending=False), use_container_width=True)