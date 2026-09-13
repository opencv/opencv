# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import itertools
import os
import pickle
import re
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np
import yaml

try:
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _pbar(iterable: Iterable, total: Optional[int] = None, desc: str = "") -> Iterable:
    if _HAS_TQDM:
        return _tqdm(iterable, total=total, desc=desc, leave=False, ncols=100)
    else:
        return iterable


def _warn(msg: str) -> None:
    warnings.warn(msg)


def _err(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)


IMG_EXTS = (
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.bmp",
    "*.tiff",
    "*.PNG",
    "*.JPG",
    "*.JPEG",
    "*.BMP",
    "*.TIFF",
)

FRAME_ID_REGEXES = [
    re.compile(r"(\d{6,})"),
    re.compile(r"(?:frame|img|f|i)[^0-9]*(\d{1,5})", re.IGNORECASE),
]


def _nat_key(s: str) -> List[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


@dataclass
class DetectResult:
    found: bool
    score: float
    features: Optional[np.ndarray]
    details: Dict[str, float]

    def to_light(self) -> Dict[str, object]:
        return {
            "found": bool(self.found),
            "score": float(self.score),
            "features": (None if self.features is None else [float(x) for x in self.features.ravel().tolist()]),
            "details": {k: float(v) for k, v in self.details.items()},
        }

    @staticmethod
    def from_light(d: Dict) -> "DetectResult":
        feats = d.get("features")
        return DetectResult(
            bool(d.get("found", False)),
            float(d.get("score", 0.0)),
            None if feats is None else np.array(feats, dtype=np.float32),
            {str(k): float(v) for k, v in d.get("details", {}).items()},
        )


def list_cameras(root: str) -> Dict[str, List[str]]:
    cams: Dict[str, List[str]] = {}
    for name in sorted(os.listdir(root), key=_nat_key):
        dpath = os.path.join(root, name)
        if not os.path.isdir(dpath):
            continue
        imgs: List[str] = []
        for ext in IMG_EXTS:
            imgs.extend(glob.glob(os.path.join(dpath, ext)))
        if imgs:
            cams[name] = sorted(imgs, key=_nat_key)
    if not cams:
        imgs: List[str] = []
        for ext in IMG_EXTS:
            imgs.extend(glob.glob(os.path.join(root, ext)))
        if imgs:
            cams["mono"] = sorted(imgs, key=_nat_key)
    if not cams:
        raise RuntimeError(f"No camera folders with images found under: {root}")
    return cams


def parse_frame_id(path: str) -> Optional[str]:
    base = os.path.basename(path)
    for rgx in FRAME_ID_REGEXES:
        m = rgx.search(base)
        if m:
            return m.group(1)
    parent = os.path.basename(os.path.dirname(path))

    for rgx in FRAME_ID_REGEXES:
        m = rgx.search(parent)
        if m:
            return m.group(1)
    return None


def _fingerprint_file(path: str) -> Tuple[float, int, str]:
    try:
        if "#frame" in path:
            vpath, _, fstr = path.rpartition("#frame")
            st = os.stat(vpath)
            content_hash = hashlib.sha256((vpath + fstr).encode()).hexdigest()
            return (st.st_mtime, st.st_size, content_hash)
        else:
            st = os.stat(path)
            with open(path, "rb") as f:
                hash_obj = hashlib.sha256(f.read(1024))
                content_hash = hash_obj.hexdigest()
            return (st.st_mtime, st.st_size, content_hash)
    except Exception:
        return (0.0, -1, "")


def _load_image_bgr(path: str, max_size: Optional[int] = None) -> Optional[np.ndarray]:
    try:
        if "#frame" in path:
            vpath, _, fstr = path.rpartition("#frame")
            fidx = int(fstr)
            cap = cv.VideoCapture(vpath)
            if not cap.isOpened():
                _warn(f"Failed to open video: {vpath}")
                return None
            cap.set(cv.CAP_PROP_POS_FRAMES, fidx)
            ok, img = cap.read()
            cap.release()
            if not ok:
                _warn(f"Failed to read frame {fidx} from video: {vpath}")
                return None
        else:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv.imdecode(data, cv.IMREAD_COLOR)  # BGR
        if img is None:
            _warn(f"Failed to decode image: {path}")
            return None
        if max_size and max_size > 0:
            h, w = img.shape[:2]
            scale = float(max_size) / float(max(h, w))
            if scale < 1.0:
                img = cv.resize(img, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)
        return img
    except Exception:
        _warn(f"Error reading image: {path}")
        return None


def _sharpness(gray: np.ndarray) -> float:
    lap = cv.Laplacian(gray, cv.CV_64F)
    var = lap.var() if lap.size > 0 else 0.0
    area = float(gray.shape[0] * gray.shape[1])
    return float(var / (area + 1e-6))


def _exposure_penalty(gray: np.ndarray) -> float:
    hist = cv.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = float(hist.sum() + 1e-6)
    bright = hist[240:256].sum() / total
    dark = hist[0:16].sum() / total
    return float(np.clip(1.0 - (bright + dark), 0.0, 1.0))


def _centre_scale_angle(corners: np.ndarray, h: int, w: int) -> Tuple[float, float, float, float]:
    pts = corners.reshape(-1, 2).astype(np.float32)
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    area = max(1.0, (x1 - x0) * (y1 - y0)) / (w * h)
    if pts.shape[0] >= 2:
        pts_c = pts - pts.mean(axis=0)
        cov = np.cov(pts_c.T)
        _, vecs = np.linalg.eigh(cov)
        v = vecs[:, 1]
        angle = float(np.degrees(np.arctan2(v[1], v[0])))
    else:
        angle = 0.0
    return cx / w, cy / h, float(np.log(area + 1e-6)), angle


def pattern_score(
    n_corners: int,
    pattern: str,
    rows: int,
    cols: int,
    sharp: float,
    exp_ok: float,
    area_ratio: float,
    expected_aruco_markers: Optional[int],
    w_sharp: float,
    w_exposure: float,
    w_corners: float,
    w_coverage: float,
) -> float:
    # clamp negative weights and normalise
    ws = [max(0.0, float(0.0 if w is None else w)) for w in (w_sharp, w_exposure, w_corners, w_coverage)]
    s = sum(ws)
    ws = [1.0 / 4.0] * 4 if s <= 0 else [w / s for w in ws]

    # expected number of corners/markers for normalising the corner term
    if pattern in ("chessboard", "circles", "acircles", "charuco"):
        expected = float(rows * cols)
    else:  # aruco_grid
        expected = 4.0 * (
            float(expected_aruco_markers) if expected_aruco_markers and expected_aruco_markers > 0 else float(rows * cols)
        )
    corners_norm = min(1.0, n_corners / (expected + 1e-6))
    sharp_term = min(1.0, sharp * 200.0)
    exposure_term = exp_ok
    coverage_term = float(np.clip(area_ratio, 0.0, 1.0))
    score = ws[0] * sharp_term + ws[1] * exposure_term + ws[2] * corners_norm + ws[3] * coverage_term
    return float(np.clip(score, 0.0, 1.0))


def _ensure_aruco_available(reason: str) -> None:
    if not hasattr(cv, "aruco"):
        _err(f"OpenCV was built without the ArUco module. Install 'opencv-contrib-python' to use {reason}.")
        sys.exit(1)


def get_aruco_dict(name: str) -> "cv.aruco.Dictionary":
    _ensure_aruco_available("ArUco/Charuco detection")
    key = name.strip()
    if hasattr(cv.aruco, key):
        dict_id = getattr(cv.aruco, key)
        if hasattr(cv.aruco, "Dictionary_get"):  # for older versions
            return cv.aruco.Dictionary_get(dict_id)
        if hasattr(cv.aruco, "getPredefinedDictionary"):
            return cv.aruco.getPredefinedDictionary(dict_id)
    fn = getattr(cv.aruco, key, None)
    if callable(fn):
        return fn()
    aliases = [
        key.replace("x", "X").upper(),
        f"DICT_{key.replace('x', 'X').upper()}",
    ]

    for alias in aliases:
        if hasattr(cv.aruco, alias):
            dict_id = getattr(cv.aruco, alias)
            if hasattr(cv.aruco, "Dictionary_get"):
                return cv.aruco.Dictionary_get(dict_id)
            if hasattr(cv.aruco, "getPredefinedDictionary"):
                return cv.aruco.getPredefinedDictionary(dict_id)
    raise ValueError(f"Unknown ArUco dictionary: {name}")


def detect_chessboard(gray: np.ndarray, rows: int, cols: int) -> Tuple[bool, Optional[np.ndarray]]:
    pattern_size = (cols, rows)
    if hasattr(cv, "findChessboardCornersSB"):
        flags_sb = cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY
        found, corners = cv.findChessboardCornersSB(gray, pattern_size, flags=flags_sb)
        if found:
            return True, corners

    flags = cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv.findChessboardCorners(gray, pattern_size, flags)
    if found:
        cv.cornerSubPix(
            gray,
            corners,
            (5, 5),
            (-1, -1),
            (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 30, 0.01),
        )
    return found, corners


def detect_circular_grid(
    gray: np.ndarray, rows: int, cols: int, asymmetric: bool = False
) -> Tuple[bool, Optional[np.ndarray], int]:
    pattern_size = (cols, rows)
    flags = cv.CALIB_CB_ASYMMETRIC_GRID if asymmetric else cv.CALIB_CB_SYMMETRIC_GRID
    found, centers = cv.findCirclesGrid(gray, pattern_size, flags=flags)
    count = int(rows * cols) if found else 0
    return found, centers, count


def detect_charuco(
    gray: np.ndarray,
    rows: int,
    cols: int,
    aruco_dict_name: str,
    square: float,
    marker: float,
) -> Tuple[bool, Optional[np.ndarray], int]:
    adict = get_aruco_dict(aruco_dict_name)
    sq = square if square > 0 else 1.0
    mk = marker if marker > 0 else 0.75 * sq
    board = cv.aruco.CharucoBoard((cols, rows), sq, mk, adict)
    if hasattr(cv.aruco, "CharucoDetector"):
        try:
            det_params = cv.aruco.DetectorParameters()
        except TypeError:
            det_params = cv.aruco.DetectorParameters_create()
        try:
            charuco_params = cv.aruco.CharucoParameters()
            charuco_params.tryRefineMarkers = True
        except Exception:
            charuco_params = None
        try:
            refine_params = cv.aruco.RefineParameters()
        except Exception:
            refine_params = None
        try:
            detector = cv.aruco.CharucoDetector(board, charuco_params, det_params, refine_params)
        except TypeError:
            detector = cv.aruco.CharucoDetector(board)
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        found = (ch_ids is not None) and (ch_corners is not None) and (len(ch_corners) >= 6)
        count = int(len(ch_corners)) if ch_corners is not None else 0
        return found, ch_corners, count
    try:
        params = cv.aruco.DetectorParameters()
        aruco_detector = cv.aruco.ArucoDetector(adict, params)
        marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)
    except Exception:
        # very old style
        marker_corners, marker_ids, _ = cv.aruco.detectMarkers(gray, adict)
    if marker_ids is None or len(marker_ids) == 0:
        return False, None, 0
    ok, ch_corners, ch_ids = cv.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
    if not ok or ch_corners is None or len(ch_corners) < 6:
        return False, None, int(len(ch_corners)) if ch_corners is not None else 0
    return True, ch_corners, int(len(ch_corners))


def detect_aruco_grid(
    gray: np.ndarray,
    aruco_dict_name: str,
    grid_cols: int,
    grid_rows: int,
    square: float,
    separation: float,
) -> Tuple[bool, Optional[np.ndarray], int]:
    adict = get_aruco_dict(aruco_dict_name)
    params = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(adict, params)
    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return False, None, 0
    # optional refine if square/separation > 0
    if grid_cols > 0 and grid_rows > 0 and square > 0 and separation > 0:
        board = cv.aruco.GridBoard((grid_cols, grid_rows), square, separation, adict)
        corners, ids, rejected, _ = detector.refineDetectedMarkers(gray, board, corners, ids, rejected)
    if ids is None or len(ids) == 0:
        return False, None, 0
    pts = np.concatenate([c.reshape(-1, 2) for c in corners], axis=0).astype(np.float32)
    return True, pts.reshape(-1, 1, 2), int(pts.shape[0])


def detect_pattern(
    img: np.ndarray,
    pattern: str,
    rows: int,
    cols: int,
    aruco_name: str,
    square: float,
    marker: float,
    separation: float,
) -> Tuple[bool, Optional[np.ndarray], int]:
    if pattern == "chessboard":
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        found, corners = detect_chessboard(gray, rows, cols)
        n_corners = rows * cols if found and corners is not None else 0
        return found, corners, n_corners

    if pattern in ("circles", "acircles"):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        asym = pattern == "acircles"
        found, centers, count = detect_circular_grid(gray, rows, cols, asymmetric=asym)
        return found, centers, count
    if pattern == "charuco":
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        found, corners, count = detect_charuco(gray, rows, cols, aruco_name, square, marker)
        return found, corners, count
    if pattern == "aruco_grid":
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        found, corners, count = detect_aruco_grid(gray, aruco_name, rows, cols, square, separation)
        return found, corners, count
    raise ValueError("Pattern must be one of: chessboard, charuco, aruco_grid, circles, acircles")


def evaluate_image(
    path: str,
    *,
    pattern: str,
    rows: int,
    cols: int,
    aruco_name: str,
    max_size: Optional[int],
    min_sharpness: float,
    min_corners: int,
    square: float,
    marker: float,
    separation: float,
    expected_aruco_markers: Optional[int],
    w_sharp: float,
    w_exposure: float,
    w_corners: float,
    w_coverage: float,
) -> DetectResult:
    img = _load_image_bgr(path, max_size=max_size)
    if img is None:
        return DetectResult(False, 0.0, None, {"read_fail": 1.0})

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    sharp = _sharpness(gray)
    exp_ok = _exposure_penalty(gray)
    details: Dict[str, float] = {
        "sharpness": sharp,
        "exposure_ok": exp_ok,
    }
    if sharp < min_sharpness:
        details.update({"found": 0.0, "n_corners": 0.0, "area_ratio": 0.0})
        return DetectResult(False, 0.0, None, details)
    found, corners, n_corners = detect_pattern(img, pattern, rows, cols, aruco_name, square, marker, separation)
    details.update({"found": 1.0 if found else 0.0, "n_corners": float(n_corners)})
    if n_corners < min_corners or not found or corners is None:
        details.setdefault("area_ratio", 0.0)
        return DetectResult(False, 0.0, None, details)
    # compute centre, scale and orientation
    cx, cy, log_scale, _ = _centre_scale_angle(corners, h, w)
    # compute convex hull area ratio (spatial coverage)
    try:
        hull = cv.convexHull(corners)
        area_ratio = float(cv.contourArea(hull) / float(w * h)) if w > 0 and h > 0 else 0.0
    except Exception:
        area_ratio = 0.0
    details["area_ratio"] = area_ratio
    score = pattern_score(
        n_corners,
        pattern,
        rows,
        cols,
        sharp,
        exp_ok,
        area_ratio,
        expected_aruco_markers,
        w_sharp,
        w_exposure,
        w_corners,
        w_coverage,
    )
    details.update({"center_x": cx, "center_y": cy, "log_scale": log_scale, "score": score})
    feats = np.array([cx, cy, log_scale], dtype=np.float32)
    return DetectResult(True, float(score), feats, details)


def _worker_init() -> None:
    try:
        cv.setNumThreads(1)
    except Exception:
        pass


def _evaluate_workers(
    args: Tuple[str, Dict],
) -> Tuple[str, float, int, str, Dict[str, object]]:
    path, ev_kwargs = args
    mtime, size, content_hash = _fingerprint_file(path)
    res = evaluate_image(path, **ev_kwargs)
    return path, mtime, size, content_hash, res.to_light()


def _build_kmeans_matrix(feats: np.ndarray, standardize: bool = True) -> np.ndarray:
    # feats columns are [cx, cy, log_scale] in that order
    km = feats.astype(np.float32, copy=False)
    if standardize and len(km) > 1:
        mu = km.mean(axis=0, dtype=np.float64)
        sd = km.std(axis=0, dtype=np.float64)
        sd[sd <= 0] = 1.0
        km = ((km - mu) / sd).astype(np.float32)
    return km


def select_kmeans(
    feats: np.ndarray,
    scores: np.ndarray,
    k: int,
    *,
    seed: int,
    standardize: bool,
) -> List[int]:
    n = len(feats)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    km_feats = _build_kmeans_matrix(feats.astype(np.float32), standardize)
    cv.setRNGSeed(seed)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    _, labels, _ = cv.kmeans(
        km_feats,
        K=k,
        bestLabels=None,
        criteria=criteria,
        attempts=5,
        flags=cv.KMEANS_PP_CENTERS,
    )
    labels = labels.ravel()
    sel: List[int] = []
    for c in range(k):
        cand = np.where(labels == c)[0]
        if cand.size == 0:
            continue
        sel.append(int(cand[np.argmax(scores[cand])]))
    if len(sel) < k:
        remaining = sorted(set(range(n)) - set(sel), key=lambda i: float(scores[i]), reverse=True)
        sel.extend(remaining[: (k - len(sel))])
    return sel


def select_greedy(
    feats: np.ndarray,
    scores: np.ndarray,
    k: int,
    *,
    min_dist: float,
    standardize: bool,
) -> List[int]:
    if len(feats) == 0 or k <= 0:
        return []
    k = min(k, len(feats))
    km = _build_kmeans_matrix(feats.astype(np.float32), standardize=standardize)
    order = list(sorted(range(len(feats)), key=lambda i: float(scores[i]), reverse=True))
    selected: List[int] = []
    taken = np.zeros(len(feats), dtype=bool)
    for i in order:
        if taken[i]:
            continue
        selected.append(i)
        if len(selected) >= k:
            break
        d = np.linalg.norm(km - km[i], axis=1)
        taken |= d < max(1e-6, min_dist)
    if len(selected) < k:
        for i in order:
            if not taken[i]:
                selected.append(i)
            if len(selected) >= k:
                break
    return selected


def select_random(
    feats: np.ndarray,
    scores: np.ndarray,
    k: int,
    *,
    seed: int,
) -> List[int]:
    n = len(feats)
    if n == 0 or k <= 0:
        return []
    rng = np.random.RandomState(seed)
    idx = np.argsort(-scores)  # high scores first
    rng.shuffle(idx[: min(k, n)])
    return idx[: min(k, n)].tolist()


def select_for_camera_cached(
    paths: List[str],
    *,
    per_camera: int,
    selector: str,
    seed: int,
    kmeans_standardize: bool,
    greedy_min_dist: float,
    cache_items: Dict[str, Dict],
) -> List[str]:
    if per_camera <= 0:
        return []
    feats_list: List[np.ndarray] = []
    scores_list: List[float] = []
    valid_paths: List[str] = []
    for p in paths:
        item = cache_items.get(p)
        if not item:
            continue
        light = item.get("result")
        if not light:
            continue
        res = DetectResult.from_light(light)
        if res.found and res.features is not None:
            feats_list.append(res.features)
            scores_list.append(res.score)
            valid_paths.append(p)
    if not valid_paths:
        return []
    fdim = len(feats_list[0])
    keep = [i for i, f in enumerate(feats_list) if len(f) == fdim]
    if len(keep) != len(feats_list):
        feats_list = [feats_list[i] for i in keep]
        scores_list = [scores_list[i] for i in keep]
        valid_paths = [valid_paths[i] for i in keep]
        if not feats_list:
            return []
    feats_arr = np.vstack(feats_list).astype(np.float32, copy=False)
    scores_arr = np.asarray(scores_list, dtype=np.float32)
    k = max(0, min(per_camera, len(valid_paths)))
    sel = selector.lower()
    if sel == "kmeans":
        idxs = select_kmeans(feats_arr, scores_arr, k, seed=seed, standardize=kmeans_standardize)
    elif sel == "greedy":
        idxs = select_greedy(
            feats_arr,
            scores_arr,
            k,
            min_dist=greedy_min_dist,
            standardize=kmeans_standardize,
        )
    elif sel == "random":
        idxs = select_random(feats_arr, scores_arr, k, seed=seed)
    else:
        raise ValueError(f"Unknown selector: {selector}")
    return [valid_paths[i] for i in idxs]


def group_by_frame_id(paths: Sequence[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for p in paths:
        fid = parse_frame_id(p)
        if fid is None:
            continue
        groups.setdefault(fid, []).append(p)
    return groups


def select_joint_consistent_cached(
    cams: Dict[str, List[str]],
    *,
    per_camera: int,
    selector: str,
    seed: int,
    kmeans_standardize: bool,
    greedy_min_dist: float,
    strict: bool,
    cache_items: Dict[str, Dict],
) -> Dict[str, List[str]]:
    cam_groups = {cam: group_by_frame_id(paths) for cam, paths in cams.items()}
    common_fids: Iterable[str] = set.intersection(*[set(g.keys()) for g in cam_groups.values()]) if cams else set()
    if not common_fids:
        if strict:
            raise RuntimeError("--require-all-cams specified but no common frame identifiers found across cameras.")
        return {
            cam: select_for_camera_cached(
                paths,
                per_camera=per_camera,
                selector=selector,
                seed=seed,
                kmeans_standardize=kmeans_standardize,
                greedy_min_dist=greedy_min_dist,
                cache_items=cache_items,
            )
            for cam, paths in cams.items()
        }
    joint: List[Tuple[str, Dict[str, Tuple[str, float, np.ndarray]]]] = []
    for fid in sorted(common_fids, key=_nat_key):
        row: Dict[str, Tuple[str, float, np.ndarray]] = {}
        ok_all = True
        for cam, groups in cam_groups.items():
            best = None
            for p in sorted(groups[fid], key=_nat_key):
                item = cache_items.get(p)
                if not item:
                    continue
                light = item.get("result")
                res = DetectResult.from_light(light)
                if res.found and res.features is not None:
                    if best is None or res.score > best[1]:
                        best = (p, res.score, res.features)
            if best is None:
                ok_all = False
                break
            row[cam] = best
        if ok_all:
            joint.append((fid, row))
    if not joint:
        if strict:
            raise RuntimeError(
                "--require-all-cams specified but no frames with valid detections across all cameras were found."
            )
        return {
            cam: select_for_camera_cached(
                paths,
                per_camera=per_camera,
                selector=selector,
                seed=seed,
                kmeans_standardize=kmeans_standardize,
                greedy_min_dist=greedy_min_dist,
                cache_items=cache_items,
            )
            for cam, paths in cams.items()
        }
    feats: List[np.ndarray] = []
    scores: List[float] = []
    for _, row in joint:
        fstack = np.vstack([row[c][2] for c in row])
        feats.append(np.median(fstack, axis=0))  # median for robustness
        scores.append(np.mean([row[c][1] for c in row]))
    feats_arr = np.vstack(feats) if feats else np.empty((0, 3), dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32) if scores else np.empty(0, dtype=np.float32)
    if selector == "kmeans":
        sel = select_kmeans(feats_arr, scores_arr, per_camera, seed=seed, standardize=kmeans_standardize)
    elif selector == "greedy":
        sel = select_greedy(
            feats_arr,
            scores_arr,
            per_camera,
            min_dist=greedy_min_dist,
            standardize=kmeans_standardize,
        )
    elif selector == "random":
        sel = select_random(feats_arr, scores_arr, per_camera, seed=seed)
    else:
        raise ValueError(f"Unknown selector: {selector}")
    selected_fids = {joint[i][0] for i in sel if i < len(joint)}
    result: Dict[str, List[str]] = {cam: [] for cam in cams}
    for fid, row in joint:
        if fid in selected_fids:
            for cam in cams:
                result[cam].append(row[cam][0])
    return result


def select_pairwise_consistent_cached(
    cams: Dict[str, List[str]],
    *,
    per_camera: int,
    cache_items: Dict[str, Dict],
) -> Dict[str, List[str]]:
    cam_groups = {cam: group_by_frame_id(paths) for cam, paths in cams.items()}
    all_fids = sorted({fid for groups in cam_groups.values() for fid in groups.keys()}, key=_nat_key)
    cams_list = sorted(cams.keys())
    # best detection per (cam, fid)
    best_of: Dict[Tuple[str, str], Tuple[str, float, np.ndarray]] = {}
    for cam in cams_list:
        for fid, plist in cam_groups[cam].items():
            best = None
            for p in sorted(plist, key=_nat_key):
                item = cache_items.get(p)
                if not item:
                    continue
                light = item.get("result")
                res = DetectResult.from_light(light if light else {})
                if res.found and res.features is not None:
                    if best is None or res.score > best[1]:
                        best = (p, res.score, res.features)
            if best:
                best_of[(cam, fid)] = best
    pair_count: Dict[Tuple[str, str], int] = defaultdict(int)
    chosen_fids: List[str] = []

    def gain(fid: str) -> Tuple[int, int]:
        cams_here = [cam for cam in cams_list if (cam, fid) in best_of]
        pairs = list(itertools.combinations(cams_here, 2))
        g = sum(1 for ab in pairs if pair_count[ab] == 0)
        return g, len(cams_here)

    while len(chosen_fids) < per_camera and all_fids:
        fid, gbest, mbest = None, -1, -1
        for f in all_fids:
            if f in chosen_fids:
                continue
            g, m = gain(f)
            if g > gbest or (g == gbest and m > mbest):
                fid, gbest, mbest = f, g, m
        if fid is None or gbest <= 0:
            break
        chosen_fids.append(fid)
        cams_here = [cam for cam in cams_list if (cam, fid) in best_of]
        for ab in itertools.combinations(cams_here, 2):
            pair_count[ab] += 1

    result = {cam: [] for cam in cams_list}
    for fid in chosen_fids:
        for cam in cams_list:
            key = (cam, fid)
            if key in best_of and len(result[cam]) < per_camera:
                result[cam].append(best_of[key][0])
    return result


def write_camera_yaml(
    out_dir: str,
    cam_name: str,
    img_paths: List[str],
    relative_to: Optional[str],
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    if relative_to:
        rel_root = os.path.abspath(relative_to)

        def to_rel(p: str) -> str:
            try:
                return os.path.relpath(os.path.abspath(p), start=rel_root)
            except Exception:
                return os.path.abspath(p)

        images = [to_rel(p) for p in img_paths]
    else:
        images = [os.path.abspath(p) for p in img_paths]
    ypath = os.path.join(out_dir, f"{cam_name}.yaml")
    with open(ypath, "w") as f:
        f.write("%YAML:1.0\n")
        yaml.safe_dump({"image_list": images}, f, sort_keys=False)
    return ypath


def write_master_yaml(out_dir: str, cam_to_yaml: Dict[str, str]) -> str:
    data = {"cameras": [{"name": cam, "yaml": os.path.abspath(yp)} for cam, yp in sorted(cam_to_yaml.items())]}
    ypath = os.path.join(out_dir, "master.yaml")
    with open(ypath, "w") as f:
        f.write("%YAML:1.0\n")
        yaml.safe_dump(data, f, sort_keys=False)
    return ypath


class MetricsWriter:
    _DEFAULT_KEYS = [
        "read_fail",
        "found",
        "sharpness",
        "exposure_ok",
        "n_corners",
        "center_x",
        "center_y",
        "log_scale",
        "area_ratio",
        "score",
    ]

    def __init__(self, csv_path: Optional[str], append: bool = False) -> None:
        self.csv_path = csv_path
        self.writer: Optional[csv.writer] = None
        self.file = None
        if csv_path:
            mode = "a" if append else "w"
            self.file = open(csv_path, mode, newline="")
            self.writer = csv.writer(self.file)
            if not append:
                self.writer.writerow(["image_path", "mtime", "size", "content_hash"] + self._DEFAULT_KEYS)

    def write(
        self,
        img_path: str,
        mtime: float,
        size: int,
        content_hash: str,
        details: Dict[str, float],
    ) -> None:
        if not self.writer:
            return
        row = [img_path, f"{mtime:.3f}", size, content_hash] + [
            details.get(k, float("nan")) for k in self._DEFAULT_KEYS
        ]
        self.writer.writerow(row)

    def close(self) -> None:
        if self.file:
            self.file.close()


_CACHE_VERSION = 7


def _config_hash(d: Dict) -> str:
    keys = [
        "pattern",
        "rows",
        "cols",
        "aruco_dict",
        "max_size",
        "min_sharpness",
        "min_corners",
        "square",
        "marker",
        "separation",
        "expected_aruco_markers",
        "w_sharp",
        "w_exposure",
        "w_corners",
        "w_coverage",
    ]
    s = "|".join(f"{k}={d.get(k)!r}" for k in keys)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def load_cache(path: str) -> Optional[Dict]:
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or data.get("version") != _CACHE_VERSION:
            return None
        return data
    except Exception:
        return None


def save_cache(path: str, data: Dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _ensure_matplotlib() -> bool:
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa

        return True
    except Exception:
        return False


def viz_per_camera(
    viz_dir: str,
    cam: str,
    all_feats: np.ndarray,
    all_scores: np.ndarray,
    sel_feats: np.ndarray,
    sel_scores: np.ndarray,
) -> None:
    ok = _ensure_matplotlib()
    if not ok:
        _warn("matplotlib not available; skipping visualization")
        return
    import matplotlib.pyplot as plt

    os.makedirs(viz_dir, exist_ok=True)

    # centers
    plt.figure(figsize=(5, 5))
    if len(all_feats):
        plt.scatter(all_feats[:, 0], all_feats[:, 1], s=6, alpha=0.25, label="all")
    if len(sel_feats):
        plt.scatter(
            sel_feats[:, 0],
            sel_feats[:, 1],
            s=20,
            alpha=0.9,
            label="selected",
            marker="o",
        )
    plt.gca().invert_yaxis()
    plt.xlabel("center_x")
    plt.ylabel("center_y")
    plt.title(f"{cam}: centers")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{cam}_centers.png"))
    plt.close()

    # scores
    plt.figure(figsize=(5, 3))
    if len(all_scores):
        plt.hist(all_scores, bins=20, alpha=0.4, label="all")
    if len(sel_scores):
        plt.hist(sel_scores, bins=20, alpha=0.9, label="selected")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.title(f"{cam}: score distribution")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{cam}_score_hist.png"))
    plt.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Select best calibration images and emit YAML lists",
        epilog=(
            "Examples:\n"
            "  python3 multicam_image_selector.py --root /data/multicam --out ./yaml --pattern chessboard --rows 7 --cols 10 --per-camera 80 --seed 42 --max-size 1600 --dump-metrics metrics.csv --jobs 8 --viz-out ./viz --cache-file .selector_cache.pkl --resume\n"
            "  python3 multicam_image_selector.py --root /data --out yaml --pattern aruco_grid --rows 7 --cols 5 --aruco-dict DICT_4X4_1000 --square 0.03 --separation 0.006 --expected-aruco-markers 20 --selector greedy --greedy-min-dist 0.5 --kmeans-standardize\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--root", help="Dataset root containing camera folders")
    group.add_argument("--video", help="Single-camera video file path")
    ap.add_argument(
        "--pattern",
        choices=["chessboard", "charuco", "aruco_grid", "circles", "acircles"],
        required=True,
        help="Type of calibration pattern",
    )
    ap.add_argument(
        "--rows",
        type=int,
        required=True,
        help="Pattern rows (inner corners or circle rows)",
    )
    ap.add_argument(
        "--cols",
        type=int,
        required=True,
        help="Pattern cols (inner corners or circle cols)",
    )
    ap.add_argument(
        "--aruco-dict",
        default="DICT_5X5_1000",
        help="ArUco dictionary name for aruco_grid/charuco",
    )
    ap.add_argument(
        "--square",
        type=float,
        default=0.0,
        help="Square size (meters) for Charuco/GridBoard",
    )
    ap.add_argument(
        "--marker",
        type=float,
        default=0.0,
        help="Marker size (meters) for Charuco/GridBoard",
    )
    ap.add_argument(
        "--separation",
        type=float,
        default=0.0,
        help="Marker separation (meters) for GridBoard",
    )
    ap.add_argument(
        "--expected-aruco-markers",
        type=int,
        default=0,
        help="Expected number of ArUco markers (for scoring normalisation). 0 = default",
    )
    ap.add_argument("--per-camera", type=int, default=80, help="Target images per camera")
    ap.add_argument(
        "--selector",
        choices=["kmeans", "greedy", "random"],
        default="kmeans",
        help="Selection algorithm. Note: random is quality-aware (top scores first).",
    )
    ap.add_argument(
        "--kmeans-standardize",
        action="store_true",
        help="Standardize features before k-means/greedy",
    )
    ap.add_argument(
        "--greedy-min-dist",
        type=float,
        default=0.6,
        help="Min feature-space distance for greedy diversity",
    )
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    ap.add_argument(
        "--require-all-cams",
        action="store_true",
        help="Select frames where all cams detect the pattern (by frame id)",
    )
    ap.add_argument(
        "--strict-all-cams",
        action="store_true",
        help="With --require-all-cams, error out if no common frames exist",
    )
    ap.add_argument("--jobs", type=int, default=0, help="Parallel workers (0=auto, 1=single)")
    ap.add_argument(
        "--max-size",
        type=int,
        default=0,
        help="Max long side in px for processing (0 = full res)",
    )
    ap.add_argument(
        "--video-step",
        type=int,
        default=10,
        help="Process every Nth frame from --video",
    )
    ap.add_argument(
        "--pairwise",
        action="store_true",
        help="Greedy selection to maximize camera-pair coverage over frame IDs",
    )
    ap.add_argument(
        "--min-sharpness",
        type=float,
        default=0.0,
        help="Minimum area-normalized Laplacian variance",
    )
    ap.add_argument("--min-corners", type=int, default=0, help="Minimum detected corners/points")
    ap.add_argument("--w-sharp", type=float, default=0.40, help="Weight for sharpness (clamped >=0)")
    ap.add_argument(
        "--w-exposure",
        type=float,
        default=0.30,
        help="Weight for exposure sanity (clamped >=0)",
    )
    ap.add_argument(
        "--w-corners",
        type=float,
        default=0.20,
        help="Weight for corner/marker coverage (clamped >=0)",
    )
    ap.add_argument(
        "--w-coverage",
        type=float,
        default=0.10,
        help="Weight for spatial coverage (convex hull area ratio). Larger values emphasise images where the pattern occupies more of the frame.",
    )
    ap.add_argument(
        "--relative-to",
        help="Write relative paths in YAMLs with respect to this directory",
    )
    ap.add_argument("--dump-metrics", help="Path to CSV file for per-image metrics (streaming)")
    ap.add_argument("--viz-out", help="Directory to save simple selection visualizations")
    ap.add_argument(
        "--cache-file",
        default=".selector_cache.pkl",
        help="Path to persistent cache pickle (default: .selector_cache.pkl)",
    )
    ap.add_argument("--resume", action="store_true", help="Resume from cache when possible")
    ap.add_argument("--out", required=True, help="Output directory for YAMLs")
    args = ap.parse_args(argv)
    if not getattr(args, "root", None) and not getattr(args, "video", None):
        _err("Provide either --root (images) or --video (mono video).")
        sys.exit(2)
    if args.pattern in ("charuco", "aruco_grid"):
        _ensure_aruco_available(f"--pattern {args.pattern}")
        try:
            _ = get_aruco_dict(args.aruco_dict)
        except Exception as ex:
            _err(str(ex))
            sys.exit(1)
    warn_list: List[str] = []
    if args.square < 0 or args.marker < 0 or args.separation < 0:
        warn_list.append("Negative physical dimensions were given; clamping to 0.")
    if args.expected_aruco_markers < 0:
        warn_list.append("--expected-aruco-markers < 0; ignoring.")
    if args.selector == "random":
        warn_list.append("--selector=random is quality-aware: selects randomly from top-scoring candidates.")
    if any(w < 0 for w in (args.w_sharp, args.w_exposure, args.w_corners, args.w_coverage)):
        warn_list.append("Negative weights provided; clamping to 0.")
    if args.greedy_min_dist <= 0 and args.selector == "greedy":
        warn_list.append("--greedy-min-dist <=0; may lead to low diversity.")
    for msg in warn_list:
        _warn(msg)
    if args.jobs <= 0:
        try:
            import multiprocessing as mp

            jobs = max(1, mp.cpu_count() - 2)
        except Exception:
            jobs = 1
    else:
        jobs = max(1, args.jobs)
    t0 = time.time()
    if getattr(args, "video", None):
        cams = {"mono": []}
        vcap = cv.VideoCapture(args.video)
        if not vcap.isOpened():
            _err(f"Cannot open video: {args.video}")
            sys.exit(1)
        fidx = 0
        step = max(1, int(args.video_step))
        while True:
            ok, _ = vcap.read()
            if not ok:
                break
            if fidx % step == 0:
                cams["mono"].append(f"{os.path.abspath(args.video)}#frame{fidx}")
            fidx += 1
        vcap.release()
        if not cams["mono"]:
            _err("No frames read from --video")
            sys.exit(1)
    else:
        cams = list_cameras(args.root)

    all_paths: List[str] = []
    for cam, paths in cams.items():
        all_paths.extend(paths)
    total_imgs = len(all_paths)
    if total_imgs == 0:
        _warn("No images found; exiting.")
        sys.exit(0)
    eval_cfg = dict(
        pattern=args.pattern,
        rows=args.rows,
        cols=args.cols,
        aruco_dict=args.aruco_dict,
        max_size=(args.max_size if args.max_size > 0 else None),
        min_sharpness=args.min_sharpness,
        min_corners=args.min_corners,
        square=max(0.0, args.square),
        marker=max(0.0, args.marker),
        separation=max(0.0, args.separation),
        expected_aruco_markers=(args.expected_aruco_markers if args.expected_aruco_markers > 0 else None),
        w_sharp=max(0.0, args.w_sharp),
        w_exposure=max(0.0, args.w_exposure),
        w_corners=max(0.0, args.w_corners),
        w_coverage=max(0.0, args.w_coverage),
    )

    cfg_hash = _config_hash(eval_cfg)
    cache_data: Optional[Dict] = None
    cache_items: Dict[str, Dict] = {}
    if args.cache_file and args.resume:
        cache_data = load_cache(args.cache_file)
        if cache_data and cache_data.get("config") == cfg_hash:
            cache_items = cache_data.get("items", {})
        elif cache_data:
            warn_list.append("Cache exists but config has changed; will re-evaluate images.")
    mw = MetricsWriter(
        args.dump_metrics,
        append=bool(args.resume and os.path.exists(args.dump_metrics or "")),
    )
    todo: List[str] = []
    for p in all_paths:
        mtime, size, content_hash = _fingerprint_file(p)
        item = cache_items.get(p)
        if not item or "content_hash" not in item:
            todo.append(p)
        else:
            if (
                (abs(item.get("mtime", 0.0) - mtime) > 1e-6)
                or (item.get("size", -1) != size)
                or (item.get("content_hash", "") != content_hash)
            ):
                todo.append(p)
    parallel_threshold = 50
    if todo:
        ev_kwargs = dict(
            pattern=eval_cfg["pattern"],
            rows=eval_cfg["rows"],
            cols=eval_cfg["cols"],
            aruco_name=eval_cfg["aruco_dict"],
            max_size=eval_cfg["max_size"],
            min_sharpness=eval_cfg["min_sharpness"],
            min_corners=eval_cfg["min_corners"],
            square=eval_cfg["square"],
            marker=eval_cfg["marker"],
            separation=eval_cfg["separation"],
            expected_aruco_markers=eval_cfg["expected_aruco_markers"],
            w_sharp=eval_cfg["w_sharp"],
            w_exposure=eval_cfg["w_exposure"],
            w_corners=eval_cfg["w_corners"],
            w_coverage=eval_cfg["w_coverage"],
        )

        tasks = [(p, ev_kwargs) for p in todo]
        desc = f"Evaluating {len(todo)}/{total_imgs} images ({jobs} jobs)"
        results = []
        if len(todo) < parallel_threshold or jobs == 1:
            for t in _pbar(tasks, total=len(tasks), desc=desc + " [sequential]"):
                results.append(_evaluate_workers(t))
        else:
            with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init) as ex:
                futures = [ex.submit(_evaluate_workers, t) for t in tasks]
                for fut in _pbar(as_completed(futures), total=len(futures), desc=desc):
                    results.append(fut.result())
        for path, mtime, size, content_hash, res_light in results:
            details = DetectResult.from_light(res_light).details
            mw.write(path, mtime, size, content_hash, details)
            cache_items[path] = {
                "mtime": mtime,
                "size": size,
                "content_hash": content_hash,
                "result": res_light,
            }
        if args.cache_file:
            data = {"version": _CACHE_VERSION, "config": cfg_hash, "items": cache_items}
            save_cache(args.cache_file, data)
    else:
        if args.dump_metrics:
            for p in _pbar(all_paths, total=len(all_paths), desc="Writing cached metrics"):
                item = cache_items.get(p)
                if not item:
                    continue
                mw.write(
                    p,
                    item.get("mtime", 0.0),
                    item.get("size", -1),
                    item.get("content_hash", ""),
                    DetectResult.from_light(item["result"]).details,
                )
    mw.close()
    selector = args.selector
    if args.require_all_cams:
        selected = select_joint_consistent_cached(
            cams,
            per_camera=args.per_camera,
            selector=selector,
            seed=args.seed,
            kmeans_standardize=args.kmeans_standardize,
            greedy_min_dist=args.greedy_min_dist,
            strict=args.strict_all_cams,
            cache_items=cache_items,
        )
    elif getattr(args, "pairwise", False):
        selected = select_pairwise_consistent_cached(
            cams,
            per_camera=args.per_camera,
            cache_items=cache_items,
        )
    else:
        selected = {
            cam: select_for_camera_cached(
                paths,
                per_camera=args.per_camera,
                selector=selector,
                seed=args.seed,
                kmeans_standardize=args.kmeans_standardize,
                greedy_min_dist=args.greedy_min_dist,
                cache_items=cache_items,
            )
            for cam, paths in cams.items()
        }

    any_selected = any(len(paths) > 0 for paths in selected.values())
    if not any_selected:
        _warn("No valid images selected for any camera. Check thresholds, pattern detection, or dataset.")
    os.makedirs(args.out, exist_ok=True)
    cam_to_yaml: Dict[str, str] = {}
    for cam, paths in selected.items():
        ypath = write_camera_yaml(args.out, cam, paths, args.relative_to)
        cam_to_yaml[cam] = ypath
    master_yaml = write_master_yaml(args.out, cam_to_yaml)
    if args.viz_out:
        ok = _ensure_matplotlib()
        if not ok:
            _warn("matplotlib not installed; skipping --viz-out visuals.")
        else:
            for cam, paths in cams.items():
                all_feats: List[np.ndarray] = []
                all_scores: List[float] = []
                for p in paths:
                    item = cache_items.get(p)
                    if not item:
                        continue
                    light = item.get("result")
                    res = DetectResult.from_light(light)
                    if res.found and res.features is not None:
                        all_feats.append(res.features)
                        all_scores.append(res.score)
                all_feats_arr = np.vstack(all_feats) if all_feats else np.empty((0, 3))
                all_scores_arr = np.array(all_scores) if all_scores else np.empty(0)
                sel_paths = selected.get(cam, [])
                sel_feats: List[np.ndarray] = []
                sel_scores: List[float] = []
                for p in sel_paths:
                    item = cache_items.get(p)
                    if not item:
                        continue
                    light = item.get("result")
                    if not light:
                        continue
                    res = DetectResult.from_light(light)
                    if res.found and res.features is not None:
                        sel_feats.append(res.features)
                        sel_scores.append(res.score)
                sel_feats_arr = np.vstack(sel_feats) if sel_feats else np.empty((0, 3))
                sel_scores_arr = np.array(sel_scores) if sel_scores else np.empty(0)
                viz_per_camera(
                    args.viz_out,
                    cam,
                    all_feats_arr,
                    all_scores_arr,
                    sel_feats_arr,
                    sel_scores_arr,
                )
    dt = time.time() - t0
    print("\nSelection summary:")
    for cam, paths in selected.items():
        print(f"  {cam}: {len(paths)} images selected")
    print(f"\nPer-camera YAMLs: {args.out}")
    print(f"Master YAML: {master_yaml}")
    print(f"Total time: {dt:.2f}s  |  Images: {total_imgs}  |  Jobs: {jobs}")


if __name__ == "__main__":
    main()
