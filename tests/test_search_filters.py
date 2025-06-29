import os
import time
import pytest
from app.utils import get_file_list_by_time_and_camera

def test_filter_by_time_and_camera(monkeypatch):
    fake_time = time.time()

    dummy_files = [
        ("images/camera_1", [], ["a.jpg", "b.jpg"]),
        ("images/camera_2", [], ["c.jpg"])
    ]

    # Simulate os.walk to return predefined structure
    monkeypatch.setattr("os.walk", lambda root: dummy_files)

    # Simulate getmtime to return current time for 'a.jpg' and older time for 'b.jpg'
    def fake_getmtime(path):
        if "a.jpg" in path:
            return fake_time  # recent
        elif "b.jpg" in path:
            return fake_time - (60 * 60)  # 1 hour old
        elif "c.jpg" in path:
            return fake_time  # recent
        return fake_time

    monkeypatch.setattr("os.path.getmtime", fake_getmtime)

    # Run the function for camera_1 and last 30 minutes
    results = get_file_list_by_time_and_camera(30, camera_id="camera_1")

    assert any("a.jpg" in r for r in results)
    assert all("b.jpg" not in r for r in results)
    assert all("c.jpg" not in r for r in results)
