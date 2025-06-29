import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from streamlit_autorefresh import st_autorefresh

from database import get_all_metadata

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="📊 Violation Dashboard", layout="wide")

# 🔄 Refresh every 20 seconds
st_autorefresh(interval=20000, key="refresh_dashboard")

st.title("📊 Violation Analytics Dashboard")

# Load data
data = get_all_metadata()

if data is None or data.empty:
    st.warning("No metadata available.")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(data)

if df.empty:
    st.warning("Metadata is empty.")
    st.stop()

# Clean and convert types
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["confidence"] = pd.to_numeric(df.get("confidence", 0), errors="coerce")
df["camera_id"] = df.get("camera_id", "Unknown").fillna("Unknown")
df = df.dropna(subset=["timestamp"])

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")
    camera_filter = st.multiselect("Select Camera ID", options=df["camera_id"].unique(), default=df["camera_id"].unique())
    time_range = st.slider("Select Time Range (Hours)", 0, 24, (0, 24))

# Apply filters
start_hour, end_hour = time_range
df = df[df["camera_id"].isin(camera_filter)]
df = df[(df["timestamp"].dt.hour >= start_hour) & (df["timestamp"].dt.hour <= end_hour)]

# Layout: Three columns for charts
col1, col2, col3 = st.columns(3)

# Chart 1: Hourly Violation Count
with col1:
    st.markdown("### ⏰ Violations by Hour")
    hourly = df["timestamp"].dt.hour.value_counts().sort_index()
    fig, ax = plt.subplots()
    sns.barplot(x=hourly.index, y=hourly.values, hue=hourly.index, legend=False, ax=ax, palette="viridis")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Violations")
    ax.set_title("Hourly Violation Count")
    st.pyplot(fig)

# Chart 2: Violations by Camera
with col2:
    st.markdown("### 🎥 Camera Distribution")
    cam_counts = df["camera_id"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(cam_counts.values, labels=cam_counts.index, autopct="%1.1f%%", startangle=140)
    ax.axis("equal")
    st.pyplot(fig)

# Chart 3: Confidence Distribution
with col3:
    st.markdown("### 📈 Confidence Score Histogram")
    fig, ax = plt.subplots()
    sns.histplot(df["confidence"], bins=20, kde=True, ax=ax, color="skyblue")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")
    ax.set_title("Confidence Distribution")
    st.pyplot(fig)

# Optional: Show raw data
st.markdown("---")
with st.expander("🗃️ View Raw Metadata Table"):
    st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
