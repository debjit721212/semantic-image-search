import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from config import METADATA_DB
import os
import logging

DB_PATH = METADATA_DB

def safe_str(val):
    return "" if val is None else str(val)

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS image_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            camera_id TEXT,
            caption TEXT,
            confidence REAL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS video_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_path TEXT NOT NULL,
            start_time TEXT,
            end_time TEXT,
            camera_id TEXT,
            summary_caption TEXT,
            key_frames TEXT  -- comma-separated list of frame paths
        )
    ''')
    conn.commit()
    conn.close()

def insert_video_metadata(video_path, start_time, end_time, camera_id, summary_caption, key_frames):
    logging.info(f"[DB] Inserting video metadata for {video_path}")
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO video_metadata (video_path, start_time, end_time, camera_id, summary_caption, key_frames)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        safe_str(video_path),
        safe_str(start_time),
        safe_str(end_time),
        safe_str(camera_id),
        safe_str(summary_caption),
        safe_str(key_frames)
    ))
    conn.commit()
    conn.close()

def insert_metadata(filename, timestamp, camera_id=None, caption=None, confidence=None):
    logging.info(f"[DB] Inserting image metadata for {filename}")
    conn = get_connection()
    filename = os.path.abspath(filename)
    c = conn.cursor()
    c.execute('''
        INSERT INTO image_metadata (filename, timestamp, camera_id, caption, confidence)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        safe_str(filename),
        safe_str(timestamp),
        safe_str(camera_id),
        safe_str(caption),
        confidence if confidence is not None else 0.0
    ))
    conn.commit()
    conn.close()

def get_all_metadata():
    try:
        conn = get_connection()
        df = pd.read_sql_query("SELECT * FROM image_metadata", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"[DB ERROR] Failed to fetch metadata: {e}")
        return pd.DataFrame()

def get_all_video_metadata():
    try:
        conn = get_connection()
        df = pd.read_sql_query("SELECT * FROM video_metadata", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"[DB ERROR] Failed to fetch video metadata: {e}")
        return pd.DataFrame()

def get_recent_metadata(hours=1):
    conn = get_connection()
    now = datetime.now()
    cutoff_time = now - timedelta(hours=hours)
    cutoff_iso = cutoff_time.isoformat()
    query = """
    SELECT * FROM image_metadata
    WHERE timestamp >= ?
    """
    df = pd.read_sql_query(query, conn, params=(cutoff_iso,))
    conn.close()
    return df

def get_recent_video_metadata(hours=1):
    conn = get_connection()
    now = datetime.now()
    cutoff_time = now - timedelta(hours=hours)
    cutoff_iso = cutoff_time.isoformat()
    query = """
    SELECT * FROM video_metadata
    WHERE start_time >= ?
    """
    df = pd.read_sql_query(query, conn, params=(cutoff_iso,))
    conn.close()
    return df

def delete_metadata_for_images(filenames):
    conn = get_connection()
    filenames = [os.path.abspath(f) for f in filenames]
    c = conn.cursor()
    placeholders = ','.join('?' for _ in filenames)
    query = f"DELETE FROM image_metadata WHERE filename IN ({placeholders})"
    c.execute(query, filenames)
    conn.commit()
    conn.close()

def get_metadata(filepath):
    filepath = os.path.abspath(filepath)
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT timestamp, camera_id, caption
        FROM image_metadata
        WHERE filename = ?
        LIMIT 1
    """, (filepath,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "timestamp": row[0],
            "camera_id": row[1],
            "caption": row[2]
        }
    else:
        return {}

def get_video_metadata(video_path):
    video_path = os.path.abspath(video_path)
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT start_time, end_time, camera_id, summary_caption, key_frames
        FROM video_metadata
        WHERE video_path = ?
        LIMIT 1
    """, (video_path,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "start_time": row[0],
            "end_time": row[1],
            "camera_id": row[2],
            "summary_caption": row[3],
            "key_frames": row[4].split(",") if row[4] else []
        }
    else:
        return {}

def save_metadata(data: dict):
    logging.info(f"[DB] Saving image metadata for {data.get('filename', '')}")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO image_metadata (filename, timestamp, camera_id, confidence)
        VALUES (?, ?, ?, ?)
    ''', (
        safe_str(data["filename"]),
        safe_str(data["timestamp"]),
        safe_str(data.get("camera_id", "unknown")),
        data.get("confidence", 0.0)
    ))
    conn.commit()
    conn.close()