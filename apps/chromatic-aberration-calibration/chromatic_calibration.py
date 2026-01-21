# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

'''
Camera calibration for chromatic aberration correction
The calibration is done of a photo of black discs on white background.
The calibration pattern can be found either in
opencv_extra/testdata/cv/cameracalibration/chromatic_aberration/chromatic_aberration_pattern_a3.png,
or can be replicated using the script for generating patterns:
https://github.com/opencv/opencv/blob/4.x/doc/pattern_tools/gen_pattern.py,
using the following invocation:

python doc/pattern_tools/gen_pattern.py \
  --output fc4_pattern_A3.svg \
  --type circles \
  --rows 26 --columns 37 \
  --units mm \
  --square_size 11 \
  --radius_rate 2.75 \
  --page_width 420 --page_height 297

And then converted to PNG:

inkscape fc4_pattern_A3.svg --export-type=png --export-dpi=300 \
  --export-background=white --export-background-opacity=1 \
  --export-filename=fc4_pattern_A3.png

Calibration image is split into b,g,r, and g is used as reference channel.
The centres of each circle in red and blue channels are found as centres of ellipses
and then calculated on a subpixel level. Each centre in red or blue channel is paired to
a respective centre in green channel. Then, a polynomial model of degree 11 is fit onto the image,
minimizing the difference between the displacements between centres in green and red/blue
and the actual delta computed with polynomial coefficients. The coefficients are then saved in yaml
format and can be used in this sample to correct images of the same camera, lens and settings.

usage:
    chromatic_calibration.py calibrate [-h] [--degree DEGREE] --coeffs_file YAML_FILE_PATH image [image ...]
    chromatic_calibration.py correct [-h] --coeffs_file YAML_FILE_PATH [-o OUTPUT] image
    chromatic_calibration.py full [-h] [--degree DEGREE] --coeffs_file YAML_FILE_PATH [-o OUTPUT] image

usage example:
    chromatic_calibration.py calibrate pattern_aberrated.png --coeffs_file calib_result.yaml

default values:
    --degree: 11
    -o, --output: corrected.png
'''

from __future__ import annotations

import argparse
import math
import pathlib
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.spatial import cKDTree


@dataclass
class Polynomial2D:
    coeffs_x: np.ndarray
    coeffs_y: np.ndarray
    degree: int
    height: int
    width: int

    def delta(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean_x, mean_y = self.width * 0.5, self.height * 0.5
        inv_std_x, inv_std_y = 1.0 / mean_x, 1.0 / mean_y
        x_n = (x - mean_x) * inv_std_x
        y_n = (y - mean_y) * inv_std_y
        terms = monomial_terms(x_n, y_n, self.degree)
        dx = terms @ self.coeffs_x
        dy = terms @ self.coeffs_y
        return dx.reshape(x.shape), dy.reshape(y.shape)



def validate_calibration_dict(data: dict) -> tuple[int, int, int]:
    required_keys = {
        "red_channel", "blue_channel", "image_width", "image_height"
    }
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"Missing keys in YAML: {', '.join(missing)}")

    width  = int(data["image_width"])
    height = int(data["image_height"])
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive integers")

    def _get_coeffs(channel: str, axis: str) -> np.ndarray:
        try:
            coeffs = np.asarray(data[channel][f"coeffs_{axis}"], dtype=float)
        except KeyError as e:
            raise ValueError(f"Missing {axis} coefficients for {channel}") from e
        if coeffs.ndim != 1:
            raise ValueError(f"{channel} {axis} coefficients must be a 1‑D list/array")
        if not np.all(np.isfinite(coeffs)):
            raise ValueError(f"{channel} {axis} coefficients contain NaN or Inf")
        return coeffs

    rx = _get_coeffs("red_channel",  "x")
    ry = _get_coeffs("red_channel",  "y")
    bx = _get_coeffs("blue_channel", "x")
    by = _get_coeffs("blue_channel", "y")

    for channel in ["red_channel", "blue_channel"]:
        try:
            rms = data[channel]["rms"]
        except KeyError as e:
            raise ValueError(f"Missing rms for {channel}") from e

    for name, cx, cy in [("red", rx, ry), ("blue", bx, by)]:
        if cx.size != cy.size:
            raise ValueError(
                f"{name} channel: coeffs_x ({cx.size}) and coeffs_y "
                f"({cy.size}) lengths differ"
            )

    if rx.size != bx.size:
        raise ValueError(
            f"Red and blue channels use different polynomial sizes "
            f"({rx.size} vs {bx.size})"
        )

    m = rx.size
    n_float = (math.sqrt(1 + 8*m) - 3) / 2
    degree  = int(round(n_float))
    expected_m = (degree + 1) * (degree + 2) // 2
    if expected_m != m:
        raise ValueError(
            f"Coefficient count {m} is not triangular (n != (deg+1)*(deg+2)/2); "
            f"nearest degree would be {degree} (needs {expected_m})"
        )

    return degree, height, width


def load_calib_result(path: str | None = None) -> dict[str, Any]:
    path = pathlib.Path(path)
    with path.open("r") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(fh)
        else:
            raise ValueError("YAML file expected as input for the calibration result")

    deg, height, width = validate_calibration_dict(data)

    red_data = data["red_channel"]
    blue_data = data["blue_channel"]

    poly_r = Polynomial2D(
        np.asarray(red_data["coeffs_x"]),
        np.asarray(red_data["coeffs_y"]),
        deg,
        height,
        width
    )
    poly_b = Polynomial2D(
        np.asarray(blue_data["coeffs_x"]),
        np.asarray(blue_data["coeffs_y"]),
        deg,
        height,
        width
    )

    return {
        "poly_red": poly_r,
        "poly_blue": poly_b,
        "image_height": height,
        "image_width": width,
    }


def repr_flow_seq(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq',
                                     data,
                                     flow_style=True)


yaml.SafeDumper.add_representer(list, repr_flow_seq)


def save_calib_result(calib, path: str | None = None) -> None:
    d = {
        "blue_channel": {
            "coeffs_x": calib["poly_blue"].coeffs_x.tolist(),
            "coeffs_y": calib["poly_blue"].coeffs_y.tolist(),
            "rms": calib["rms_red"]
        },
        "red_channel": {
            "coeffs_x": calib["poly_red"].coeffs_x.tolist(),
            "coeffs_y": calib["poly_red"].coeffs_y.tolist(),
            "rms": calib["rms_blue"]
        },
        "image_width": calib["image_width"],
        "image_height": calib["image_height"]
    }
    if path is not None:
        with open(path, "w") as fh:
            yaml.safe_dump(d,
                            fh,
                            version=(1, 2),
                            default_flow_style=False,
                            sort_keys=False)


def monomial_terms(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    x = x.flatten()
    y = y.flatten()
    terms = []
    cnt = 0
    for total in range(degree + 1):
        for i in range(total + 1):
            j = total - i
            terms.append((x ** i) * (y ** j))
            cnt += 1
    return np.vstack(terms).T


def detect_disk_centres(
    img: np.ndarray,
    *,
    min_area: int = 20,
    max_area: int | None = None,
    circularity_thresh: float = 0.7,
    morph_kernel: int = 3,
) -> np.ndarray:
    if img.ndim != 2:
        raise ValueError("detect_disk_centres expects a grayscale image")
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    centres = []

    for c in cnts:
        if len(c) < 5:
            continue
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        peri = cv2.arcLength(c, closed=True)
        circularity = 4 * np.pi * area / (peri * peri + 1e-12)
        if circularity < circularity_thresh:
            continue
        (cx, cy), (a, b), theta = cv2.fitEllipse(c)

        eps = 1e-6
        pts = c.reshape(-1, 2).astype(np.float64)
        ct, st = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        r = np.array([[ct, st], [-st, ct]])

        # translate points so that they are centered around mean, and rotate them
        p = (r @ (pts.T - np.array([[cx], [cy]]))).T
        # ellipse equation
        f = (p[:, 0] / (a / 2 + eps)) ** 2 + (p[:, 1] / (b / 2 + eps)) ** 2 - 1
        # gradients of ellipse equation
        j = np.column_stack(
            [2 * p[:, 0] / ((a / 2 + eps) ** 2), 2 * p[:, 1] / ((b / 2 + eps) ** 2)]
        )

        # solve least squares to get delta of centers
        delta, *_ = np.linalg.lstsq(j, -f, rcond=None)
        cx -= delta[0]
        cy -= delta[1]
        centres.append((cx, cy))

    if len(centres) == 0:
        raise RuntimeError("No valid disks detected, check function parameters")

    return np.asarray(centres, dtype=np.float32)


def pair_keypoints(
    ref: np.ndarray,
    target: np.ndarray,
    max_error: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tree = cKDTree(ref)
    dists, idx = tree.query(target, distance_upper_bound=max_error)
    mask = np.isfinite(dists)
    if not np.any(mask):
        raise RuntimeError("No valid keypoint matches were created")
    target_valid = target[mask]
    ref_valid = ref[idx[mask]]
    disp = ref_valid - target_valid
    return target_valid[:, 0], target_valid[:, 1], disp


def fit_channel(
    x: np.ndarray,
    y: np.ndarray,
    disp: np.ndarray,
    degree: int,
    height: int,
    width: int,
    method: str = "L-BFGS-B",
) -> tuple[np.ndarray, np.ndarray, float]:
    mean_x, mean_y = width * 0.5, height * 0.5
    inv_std_x, inv_std_y = 1.0 / mean_x, 1.0 / mean_y
    x = (x - mean_x) * inv_std_x
    y = (y - mean_y) * inv_std_y

    terms = monomial_terms(x, y, degree)
    m = terms.shape[1]

    def objective(c: np.ndarray) -> float:
        cx = c[:m]
        cy = c[m:]
        pred_x = terms @ cx
        pred_y = terms @ cy
        err = np.hstack([pred_x - disp[:, 0], pred_y - disp[:, 1]])
        if np.any(np.isnan(err)) or np.any(np.isinf(err)):
            return 1e12
        return np.sum(err ** 2)

    cx_ls, *_ = np.linalg.lstsq(terms, disp[:, 0], rcond=None)
    cy_ls, *_ = np.linalg.lstsq(terms, disp[:, 1], rcond=None)
    c0 = np.hstack([cx_ls, cy_ls])

    res = minimize(objective, c0, method=method, options={
                    "maxiter": 500,
                    "maxfun": 5000,
                    "maxls": 50,
                    "ftol": 1e-9,
               })

    coeffs_x = res.x[:m]
    coeffs_y = res.x[m:]
    rms = math.sqrt(res.fun / disp.shape[0])
    return coeffs_x, coeffs_y, rms


def fit_polynomials(
    x_r: np.ndarray,
    y_r: np.ndarray,
    disp_r: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
    disp_b: np.ndarray,
    degree: int,
    height: int,
    width: int
) -> tuple[Polynomial2D, Polynomial2D, float, float]:
    crx, cry, rms_r = fit_channel(x_r, y_r, disp_r, degree, height, width)
    cbx, cby, rms_b = fit_channel(x_b, y_b, disp_b, degree, height, width)
    poly_r = Polynomial2D(crx, cry, degree, height, width)
    poly_b = Polynomial2D(cbx, cby, degree, height, width)
    return poly_r, poly_b, rms_r, rms_b

def calibrate(
    imgs: list[np.ndarray],
    degree: int = 11,
):
    xr_all, yr_all, dr_all = [], [], []
    xb_all, yb_all, db_all = [], [], []
    h0, w0 = None, None

    for i, img in enumerate(imgs):
        if img is None or img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Expected a BGR color image")

        h, w = img.shape[:2]
        b, g, r = cv2.split(img)

        pts_g = detect_disk_centres(g)
        pts_r = detect_disk_centres(r)
        pts_b = detect_disk_centres(b)

        xr, yr, disp_r = pair_keypoints(pts_g, pts_r)
        xb, yb, disp_b = pair_keypoints(pts_g, pts_b)
        if h0 is None:
            h0, w0 = h, w
        else:
            if (h, w) != (h0, w0):
                raise ValueError(
                    f"All calibration images must have the same resolution; "
                    f"got {(h,w)} vs {(h0,w0)} at image #{i}"
                )

        xr_all.append(xr)
        yr_all.append(yr)
        dr_all.append(disp_r)
        xb_all.append(xb)
        yb_all.append(yb)
        db_all.append(disp_b)

    xr = np.concatenate(xr_all, axis=0)
    yr = np.concatenate(yr_all, axis=0)
    disp_r = np.concatenate(dr_all, axis=0)

    xb = np.concatenate(xb_all, axis=0)
    yb = np.concatenate(yb_all, axis=0)
    disp_b = np.concatenate(db_all, axis=0)

    poly_r, poly_b, rms_r, rms_b = fit_polynomials(
        xr, yr, disp_r,
        xb, yb, disp_b,
        degree, h0, w0
    )

    print(f"Calibrated polynomial with degree {degree} on {len(imgs)} images, "
            f"RMS red: {rms_r:.3f} px; RMS blue: {rms_b:.3f} px")

    return {
        "poly_red": poly_r,
        "poly_blue": poly_b,
        "image_width": w0,
        "image_height": h0,
        "rms_red": rms_r,
        "rms_blue": rms_b,
    }

def calibrate_multi_degree(
    imgs: list[np.ndarray],
    k0: int,
    k1: int,
) -> dict[int, tuple[Polynomial2D, Polynomial2D, float, float]]:
    """
    Returns a dict mapping degree → (poly_r, poly_b, rms_r, rms_b).
    """
    xr_all, yr_all, dr_all = [], [], []
    xb_all, yb_all, db_all = [], [], []
    h0, w0 = None, None

    for i, img in enumerate(imgs):
        if img is None or img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Expected a BGR color image")

        h, w = img.shape[:2]
        b, g, r = cv2.split(img)

        pts_g = detect_disk_centres(g)
        pts_r = detect_disk_centres(r)
        pts_b = detect_disk_centres(b)

        xr, yr, disp_r = pair_keypoints(pts_g, pts_r)
        xb, yb, disp_b = pair_keypoints(pts_g, pts_b)
        if h0 is None:
            h0, w0 = h, w
        else:
            if (h, w) != (h0, w0):
                raise ValueError(
                    f"All calibration images must have the same resolution; "
                    f"got {(h,w)} vs {(h0,w0)} at image #{i}"
                )

        xr_all.append(xr)
        yr_all.append(yr)
        dr_all.append(disp_r)
        xb_all.append(xb)
        yb_all.append(yb)
        db_all.append(disp_b)

    xr = np.concatenate(xr_all, axis=0)
    yr = np.concatenate(yr_all, axis=0)
    disp_r = np.concatenate(dr_all, axis=0)

    xb = np.concatenate(xb_all, axis=0)
    yb = np.concatenate(yb_all, axis=0)
    disp_b = np.concatenate(db_all, axis=0)

    results = {}
    for deg in range(k0, k1+1):
        print(deg)

        poly_r, poly_b, rms_r, rms_b = fit_polynomials(
            xr,
            yr,
            disp_r,
            xb,
            yb,
            disp_b,
            deg,
            h0,
            w0
        )
        print(f"Calibrated polynomial with degree {deg},               RMS red: {rms_r:.3f} px; RMS blue: {rms_b:.3f} px")
        results[deg] = (poly_r, poly_b, rms_r, rms_b)
    return results


def build_remap(
    h: int,
    w: int,
    poly: Polynomial2D,
) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dx, dy = poly.delta(x, y)
    map_x = (x - dx).astype(np.float32)
    map_y = (y - dy).astype(np.float32)
    return map_x, map_y


def correct_image(
    img: np.ndarray,
    calib: dict[str, Any],
) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("correct_image expects a BGR colour image")

    h, w = img.shape[:2]
    b, g, r = cv2.split(img)
    map_x_r, map_y_r = build_remap(h, w, calib["poly_red"])
    map_x_b, map_y_b = build_remap(h, w, calib["poly_blue"])

    r_corr = cv2.remap(r, map_x_r, map_y_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    b_corr = cv2.remap(b, map_x_b, map_y_b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    map_x_g, map_y_g = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    g_corr = cv2.remap(g, map_x_g, map_y_g,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)

    corrected = cv2.merge((b_corr, g_corr, r_corr))
    return corrected

def detect_disk_contours(
    img: np.ndarray,
    *,
    min_area: int = 20,
    max_area: int | None = None,
    circularity_thresh: float = 0.7,
    morph_kernel: int = 3,
) -> list[np.ndarray]:
    """
    Find all external contours of “discs” in a binary mask of `img` and return
    their raw point coordinates as a list of (N_i,2) float32 arrays.
    """
    if img.ndim != 2:
        raise ValueError("detect_disk_contours expects a grayscale image")
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel,)*2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = []
    for c in cnts:
        if len(c) < 5:
            continue
        area = cv2.contourArea(c)
        if area < min_area or (max_area is not None and area > max_area):
            continue
        peri = cv2.arcLength(c, True)
        circ = 4 * math.pi * area / (peri*peri + 1e-12)
        if circ < circularity_thresh:
            continue
        pts = c.reshape(-1, 2).astype(np.float32)
        contours.append(pts)
    if not contours:
        raise RuntimeError("No valid disk contours found")
    return contours

def warp_and_compare(contours_src: list[np.ndarray],
                     poly_src: Polynomial2D,
                     pts_ref: np.ndarray) -> np.ndarray:
    """
    Warp src-channel contours through poly_src.delta,
    then compute for each warped point its distance to the nearest
    green contour point in pts_ref.
    """
    pts = np.vstack(contours_src)
    xs, ys = pts[:,0], pts[:,1]
    dx, dy = poly_src.delta(xs, ys)
    warped = np.column_stack([xs - dx, ys - dy])

    tree = cKDTree(pts_ref)
    dists, _ = tree.query(warped, k=1)
    return dists


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chromatic aberration calibration and correction tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("calibrate", help="Calibrate from calibration target image")
    sc.add_argument("image", nargs="+", help="One or more images of black‑disk calibration target")
    sc.add_argument("--degree", type=int, default=11, help="Polynomial degree")
    sc.add_argument("--coeffs_file", required=True, help="Save coefficients to YAML file")

    sr = sub.add_parser("correct", help="Correct a photograph using saved coefficients")
    sr.add_argument("image", help="Input image to be corrected")
    sr.add_argument("--coeffs_file", required=True,
                    help="Calibration coefficient file (.json/.yaml)")
    sr.add_argument("-o", "--output", default="corrected.png", help="Output filename")

    sf = sub.add_parser("full",help="Calibrate from calibration target image and \
                        correct the calibration target")
    sf.add_argument("image", nargs="+", help="One or more images of black‑disk calibration target")
    sf.add_argument("--degree", type=int, default=11, help="Polynomial degree")
    sf.add_argument("--coeffs_file", required=True, help="Save coefficients to YAML file")
    sf.add_argument("-o", "--output", default="corrected.png", help="Output filename")

    ss = sub.add_parser("scan", help="Sweep degree range and report errors")
    ss.add_argument("image", nargs="+", help="Calibration image path")
    ss.add_argument("--degree_range", nargs=2, type=int, metavar=("k0","k1"),
                    required=True, help="Inclusive degree range to scan")
    ss.add_argument("--method", default="POWELL", help="Optimizer method")

    return p.parse_args()


def cmd_calibrate(parsed_args: argparse.Namespace) -> None:
    paths = parsed_args.image if isinstance(parsed_args.image, list) else [parsed_args.image]
    imgs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(p)
        imgs.append(im)

    calib = calibrate(imgs, degree=parsed_args.degree)
    save_calib_result(calib, path=parsed_args.coeffs_file)
    print("Saved coefficients to", parsed_args.coeffs_file)


def cmd_correct(parsed_args: argparse.Namespace) -> None:
    path = parsed_args.image

    fs = cv2.FileStorage(parsed_args.coeffs_file, cv2.FileStorage_READ)
    if not fs.isOpened():
        print(f"Could not calibration coefficients from {parsed_args.coeffs_file}")
        return
    coeff_mat, calib_size, degree = cv2.loadChromaticAberrationParams(fs.root())

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not read image {path}")
        return

    fixed = cv2.correctChromaticAberration(img, coeff_mat, calib_size, degree)

    cv2.imwrite(parsed_args.output, fixed)
    print(f"Corrected image written to {parsed_args.output}")


def cmd_full(parsed_args: argparse.Namespace) -> None:
    paths = parsed_args.image if isinstance(parsed_args.image, list) else [parsed_args.image]
    imgs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(p)
        imgs.append(im)

    calib = calibrate(imgs, degree=parsed_args.degree)
    img_for_correction = imgs[0]
    save_calib_result(calib, path=parsed_args.coeffs_file)
    print("Saved coefficients to", parsed_args.coeffs_file)

    fs = cv2.FileStorage(parsed_args.coeffs_file, cv2.FileStorage_READ)
    if not fs.isOpened():
        print(f"Could not calibration coefficients from {parsed_args.coeffs_file}")
        return
    coeff_mat, calib_size, degree = cv2.loadChromaticAberrationParams(fs.root())

    fixed = cv2.correctChromaticAberration(img_for_correction, coeff_mat, calib_size, degree)
    cv2.imwrite(parsed_args.output, fixed)
    print(f"Corrected image written to {parsed_args.output}")


def cmd_scan(parsed_args: argparse.Namespace) -> None:
    paths = parsed_args.image if isinstance(parsed_args.image, list) else [parsed_args.image]
    imgs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(p)
        imgs.append(im)

    k0, k1 = parsed_args.degree_range
    results = calibrate_multi_degree(imgs, k0, k1)

    all_contours_b = []
    all_contours_g = []
    all_contours_r = []

    for img in imgs:
        b, g, r = cv2.split(img)
        all_contours_b.extend(detect_disk_contours(b))
        all_contours_g.extend(detect_disk_contours(g))
        all_contours_r.extend(detect_disk_contours(r))

    pts_g = np.vstack(all_contours_g)

    print(f"Reference degree: {k1}\n")
    header = "deg |   max_r   mean_r   std_r   |   max_b   mean_b   std_b"
    print(header)
    print("-" * len(header))

    for deg in sorted(results):
        if deg == k1:
            continue
        pr, pb, _, _ = results[deg]

        d_r = warp_and_compare(all_contours_r, pr, pts_g)
        d_b = warp_and_compare(all_contours_b, pb, pts_g)

        s = {
            'max_r': d_r.max(), 'mean_r': d_r.mean(), 'std_r': d_r.std(),
            'max_b': d_b.max(), 'mean_b': d_b.mean(), 'std_b': d_b.std()
        }

        print(f"{deg:3d} | "
              f"{s['max_r']:8.3f} {s['mean_r']:8.3f} {s['std_r']:8.3f} | "
              f"{s['max_b']:8.3f} {s['mean_b']:8.3f} {s['std_b']:8.3f}")


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "calibrate":
        cmd_calibrate(args)
    elif args.cmd == "correct":
        cmd_correct(args)
    elif args.cmd == "full":
        cmd_full(args)
    elif args.cmd == "scan":
        cmd_scan(args)
