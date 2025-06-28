# app/database.py

import sqlite3
from datetime import datetime,timedelta
import pandas as pd
from pathlib import Path
from config import METADATA_DB
import os

DB_PATH = Path("data/metadata.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS image_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            camera_id TEXT,
            caption TEXT,
            embedding_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_metadata(filename, timestamp, camera_id=None, caption=None, embedding_path=None):
    conn = sqlite3.connect(DB_PATH)
    filename = os.path.abspath(filename)  
    c = conn.cursor()
    c.execute('''
        INSERT INTO image_metadata (filename, timestamp, camera_id, caption, embedding_path)
        VALUES (?, ?, ?, ?, ?)
    ''', (filename, timestamp, camera_id, caption, embedding_path))
    conn.commit()
    conn.close()

def query_images_by_time_range(start_time: datetime, end_time: datetime):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT * FROM image_metadata
        WHERE timestamp BETWEEN ? AND ?
    ''', (start_time.isoformat(), end_time.isoformat()))
    rows = c.fetchall()
    conn.close()
    return rows

def get_all_metadata():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM image_metadata")
    rows = c.fetchall()
    conn.close()
    return rows

def initialize_db():
    """Create the metadata.db SQLite database if it doesn't exist."""
    os.makedirs(os.path.dirname(METADATA_DB), exist_ok=True)
    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()
    
    # Example table for storing image metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS image_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            camera_id TEXT,
            caption TEXT
        );
    """)
    
    conn.commit()
    conn.close()

def delete_metadata_for_images(filenames):
    conn = sqlite3.connect(DB_PATH)
    filenames = [os.path.abspath(f) for f in filenames]  # 🔒 Normalize to absolute path
    c = conn.cursor()
    placeholders = ','.join('?' for _ in filenames)
    query = f"DELETE FROM image_metadata WHERE filename IN ({placeholders})"
    c.execute(query, filenames)
    conn.commit()
    conn.close()

def get_recent_metadata(hours=1):
    conn = sqlite3.connect(DB_PATH)
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


def get_metadata(filepath):
    """Returns metadata for a given image filepath."""
    filepath = os.path.abspath(filepath)  # Normalize
    conn = sqlite3.connect(DB_PATH)
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