import os
import cv2 as cv
from pathlib import Path
from common import imread_url

def test_imread_url_file_local(tmp_path: Path):
    # Create a tiny 2x1 PNG locally
    tiny = (tmp_path / "tiny.png").as_posix()
    import numpy as np
    import cv2 as cv
    img = np.array([[[255,0,0],[0,255,0]]], dtype=np.uint8)  # 1x2 BGR
    cv.imwrite(tiny, img)
    url = "file://" + tiny
    out = imread_url(url, cv.IMREAD_UNCHANGED)
    assert out is not None and out.shape[:2] == (1, 2)

def test_imread_url_rejects_non_image(monkeypatch, tmp_path: Path):
    # Serve non-image via file:// to trigger MIME-less path; simulate with bad data
    bad = (tmp_path / "bad.txt")
    bad.write_text("<html>not an image</html>")
    url = "file://" + bad.as_posix()
    out = imread_url(url)  # will decode to None
    assert out is None

def test_imread_url_size_cap(tmp_path: Path):
    # Create a large random file to exceed cap
    big = (tmp_path / "big.bin")
    big.write_bytes(b"\x00" * (33 * 1024 * 1024))
    url = "file://" + big.as_posix()
    out = imread_url(url, max_bytes=32 * 1024 * 1024)
    assert out is None
