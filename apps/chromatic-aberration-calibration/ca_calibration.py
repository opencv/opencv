#!/usr/bin/env python

'''
Camera calibration for chromatic aberration correction
The calibration is done of a photo of black discs on white background.
Calibration image is split into b,g,r, and g is used as reference channel.
The centres of each circle in red and blue channels are found as centres of ellipses and then calculated on a subpixel level.
Each centre in red or blue channel is paired to a respective centre in green channel.
Then, a polynomial model of degree 11 is fit onto the image, minimizing the difference between the displacements between centres in green and red/blue and the actual delta computed with polynomial coefficients.
The coefficients are then saved in yaml format and can be used in this sample to correct images of the same camera, lens and settings.

usage:
    ca_calibration.py calibrate [-h] [--degree DEGREE] --coeffs_file YAML_FILE_PATH image
    ca_calibration.py correct [-h] --coeffs_file YAML_FILE_PATH [-o OUTPUT] image
    ca_calibration.py full [-h] [--degree DEGREE] --coeffs_file YAML_FILE_PATH [-o OUTPUT] image

usage example:
    ca_calibration.py calibrate pattern_aberrated.png --coeffs_file calib_result.yaml

default values:
    --degree: 11
    -o, --output: corrected.png
'''

from __future__ import annotations

import argparse
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, Tuple

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

    def delta(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_x, mean_y = self.width * 0.5, self.height * 0.5
        inv_std_x, inv_std_y = 1.0 / mean_x, 1.0 / mean_y
        x_n = (x - mean_x) * inv_std_x
        y_n = (y - mean_y) * inv_std_y
        terms = monomial_terms(x_n, y_n, self.degree)
        dx = terms @ self.coeffs_x
        dy = terms @ self.coeffs_y
        return dx.reshape(x.shape), dy.reshape(y.shape)



def validate_calibration_dict(data: dict) -> Tuple[int, int, int]:
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


def load_calib_result(path: str | None = None) -> Dict:
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
        R = np.array([[ct, st], [-st, ct]])

        # translate points so that they are centered around mean, and rotate them
        p = (R @ (pts.T - np.array([[cx], [cy]]))).T
        # ellipse equation
        f = (p[:, 0] / (a / 2 + eps)) ** 2 + (p[:, 1] / (b / 2 + eps)) ** 2 - 1
        # gradients of ellipse equation
        J = np.column_stack(
            [2 * p[:, 0] / ((a / 2 + eps) ** 2), 2 * p[:, 1] / ((b / 2 + eps) ** 2)]
        )

        # solve least squares to get delta of centers
        delta, *_ = np.linalg.lstsq(J, -f, rcond=None)
        delta_img = R.T @ delta 
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tree = cKDTree(ref)
    dists, idx = tree.query(target, distance_upper_bound=max_error)
    mask = np.isfinite(dists)
    if not np.any(mask):
        raise RuntimeError(
            f"No valid keypoint matches were created"
        )
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
) -> Tuple[np.ndarray, np.ndarray, float]:
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
                   "maxfun": 5000,   # allow more total function+gradient calls
                    "maxls": 50,
                   "ftol": 1e-9,
                #    "disp": True,
               })


    # if not res.success:
    #     raise RuntimeError(f"Optimiser failed: {res.message}")

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
) -> Tuple[Polynomial2D, Polynomial2D, float, float]:
    crx, cry, rms_r = fit_channel(x_r, y_r, disp_r, degree, height, width)
    cbx, cby, rms_b = fit_channel(x_b, y_b, disp_b, degree, height, width)
    poly_r = Polynomial2D(crx, cry, degree, height, width)
    poly_b = Polynomial2D(cbx, cby, degree, height, width)
    return poly_r, poly_b, rms_r, rms_b


def calibrate_from_image(
    img: np.ndarray,
    degree: int = 11,
):
    h, w = img.shape[:2]
    b, g, r = cv2.split(img)

    pts_g = detect_disk_centres(g)
    pts_r = detect_disk_centres(r)
    pts_b = detect_disk_centres(b)

    xr, yr, disp_r = pair_keypoints(pts_g, pts_r)
    xb, yb, disp_b = pair_keypoints(pts_g, pts_b)

    poly_r, poly_b, rms_r, rms_b = fit_polynomials(
        xr,
        yr,
        disp_r,
        xb,
        yb,
        disp_b,
        degree,
        h,
        w
    )

    print(f"Calibrated polynomial with degree {degree}, RMS red: {rms_r:.3f} px; RMS blue: {rms_b:.3f} px")

    return {
        "poly_red": poly_r,
        "poly_blue": poly_b,
        "image_width": w,
        "image_height": h,
        "rms_red": rms_r,
        "rms_blue": rms_b
    }

def calibrate_multi_degree(
    img: np.ndarray,
    K0: int,
    K1: int,
) -> Dict[int, Tuple[Polynomial2D, Polynomial2D, float, float]]:
    """
    Returns a dict mapping degree → (poly_r, poly_b, rms_r, rms_b).
    """
    h, w = img.shape[:2]
    b, g, r = cv2.split(img)

    pts_g = detect_disk_centres(g)
    pts_r = detect_disk_centres(r)
    pts_b = detect_disk_centres(b)

    xr, yr, disp_r = pair_keypoints(pts_g, pts_r)
    xb, yb, disp_b = pair_keypoints(pts_g, pts_b)

    results = {}
    for deg in range(K0, K1+1):
        print(deg)

        poly_r, poly_b, rms_r, rms_b = fit_polynomials(
            xr,
            yr,
            disp_r,
            xb,
            yb,
            disp_b,
            deg,
            h,
            w
        )
        print(f"Calibrated polynomial with degree {deg}, RMS red: {rms_r:.3f} px; RMS blue: {rms_b:.3f} px")
        results[deg] = (poly_r, poly_b, rms_r, rms_b)
    return results

def degree_error_stats(
    results: Dict[int, Tuple[Polynomial2D, Polynomial2D, float, float]],
) -> Dict[int, Dict[str, float]]:
    """
    Given the dict from calibrate_multi_degree, returns for each degree k < K1:
      {
        'max_r': ...,
        'mean_r': ...,
        'std_r':  ...,
        'max_b': ...,
        'mean_b': ...,
        'std_b':  ...
      }
    using degree=K1 as reference.
    """
    degs = sorted(results)
    K1 = degs[-1]
    h, w = results[K1][0].height, results[K1][0].width

    # precompute reference maps (in pixel coords)
    pr_ref, pb_ref, _, _ = results[K1]
    rx_ref, ry_ref = build_remap(h, w, pr_ref)
    bx_ref, by_ref = build_remap(h, w, pb_ref)

    stats = {}
    for deg in degs[:-1]:
        pr, pb, _, _ = results[deg]
        rx_k, ry_k = build_remap(h, w, pr)
        bx_k, by_k = build_remap(h, w, pb)

        # compute pixel‐wise displacement error
        err_r = np.sqrt((rx_k - rx_ref)**2 + (ry_k - ry_ref)**2)
        err_b = np.sqrt((bx_k - bx_ref)**2 + (by_k - by_ref)**2)

        stats[deg] = {
            'max_r': float(err_r.max()),
            'mean_r': float(err_r.mean()),
            'std_r':  float(err_r.std()),
            'max_b': float(err_b.max()),
            'mean_b': float(err_b.mean()),
            'std_b':  float(err_b.std()),
        }
    return stats



def build_remap(
    h: int,
    w: int,
    poly: Polynomial2D,
) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dx, dy = poly.delta(X, Y)
    map_x = (X - dx).astype(np.float32)
    map_y = (Y - dy).astype(np.float32)
    return map_x, map_y


def correct_image(
    img: np.ndarray,
    calib: Dict,
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chromatic aberration calibration and correction tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("calibrate", help="Calibrate from calibration target image")
    sc.add_argument("image", help="Image of black‑disk calibration target")
    sc.add_argument("--degree", type=int, default=11, help="Polynomial degree")
    sc.add_argument("--coeffs_file", required=True, help="Save coefficients to YAML file")

    sr = sub.add_parser("correct", help="Correct a photograph using saved coefficients")
    sr.add_argument("image", help="Input image to be corrected")
    sr.add_argument("--coeffs_file", required=True, help="Calibration coefficient file (.json/.yaml)")
    sr.add_argument("-o", "--output", default="corrected.png", help="Output filename")

    sf = sub.add_parser("full", help="Calibrate from calibration target image and correct the calibration target")
    sf.add_argument("image", help="Image of black‑disk calibration target")
    sf.add_argument("--degree", type=int, default=11, help="Polynomial degree")
    sf.add_argument("--coeffs_file", required=True, help="Save coefficients to YAML file")
    sf.add_argument("-o", "--output", default="corrected.png", help="Output filename")

    ss = sub.add_parser("scan", help="Sweep degree range and report errors")
    ss.add_argument("image", help="Calibration image path")
    ss.add_argument("--degree_range", nargs=2, type=int, metavar=("K0","K1"),
                    required=True, help="Inclusive degree range to scan")
    ss.add_argument("--method", default="POWELL", help="Optimizer method")

    return p.parse_args()


def cmd_calibrate(args: argparse.Namespace) -> None:
    calib = calibrate_from_image(
        cv2.imread(args.image, cv2.IMREAD_COLOR),
        degree=args.degree
    )
    save_calib_result(calib, path=args.coeffs_file)
    print("Saved coefficients to", args.coeffs_file)

    # img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    # b, g, r = cv2.split(img)

    # pts_g = detect_disk_centres(g)
    # pts_r = detect_disk_centres(r)
    # pts_b = detect_disk_centres(b)

    # xr, yr, disp_r = pair_keypoints(pts_g, pts_r, max_error=30.0)
    # xb, yb, disp_b = pair_keypoints(pts_g, pts_b, max_error=30.0)

    # mag_r = np.hypot(disp_r[:,0], disp_r[:,1])
    # thr_r = mag_r.mean() + 2.0 * mag_r.std()
    # out_r = mag_r > thr_r

    # mag_b = np.hypot(disp_b[:,0], disp_b[:,1])
    # thr_b = mag_b.mean() + 2.0 * mag_b.std()
    # out_b = mag_b > thr_b

    # print(f"Red matches: {len(mag_r)}, outliers: {out_r.sum()} (thr={thr_r:.2f}px)")
    # print(f"Blue matches: {len(mag_b)}, outliers: {out_b.sum()} (thr={thr_b:.2f}px)")

    # vis = img.copy()
    # for x, y, (dx, dy) in zip(xr, yr, disp_r):
    #     end = (int(x+dx), int(y+dy))
    #     cv2.arrowedLine(vis, (int(x), int(y)), end,
    #                     color=(0,0,255), thickness=1, tipLength=0.2)
    # for x, y, (dx, dy) in zip(xb, yb, disp_b):
    #     end = (int(x+dx), int(y+dy))
    #     cv2.arrowedLine(vis, (int(x), int(y)), end,
    #                     color=(255,0,0), thickness=1, tipLength=0.2)

    # for idx in np.where(out_r)[0]:
    #     x, y = xr[idx], yr[idx]
    #     dx, dy = disp_r[idx]
    #     cv2.arrowedLine(vis, (int(x), int(y)),
    #                     (int(x+dx), int(y+dy)),
    #                     color=(255,0,255), thickness=2, tipLength=0.3)
    # for idx in np.where(out_b)[0]:
    #     x, y = xb[idx], yb[idx]
    #     dx, dy = disp_b[idx]
    #     cv2.arrowedLine(vis, (int(x), int(y)),
    #                     (int(x+dx), int(y+dy)),
    #                     color=(255,255,0), thickness=2, tipLength=0.3)

    # out_path = 'pairing_outliers.png'
    # cv2.imwrite(out_path, vis)
    # print(f"Written visualization to {out_path}")



def cmd_correct(args: argparse.Namespace) -> None:
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)
    calib = load_calib_result(args.coeffs_file)
    fixed = correct_image(img, calib)
    cv2.imwrite(args.output, fixed)
    print(f"Corrected image written to {args.output}")

def cmd_full(args: argparse.Namespace) -> None:
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)
    calib = calibrate_from_image(img, degree=args.degree)
    save_calib_result(calib, path=args.coeffs_file)
    print("Saved coefficients to", args.coeffs_file)
    fixed = correct_image(img, calib)
    cv2.imwrite(args.output, fixed)
    print(f"Corrected image written to {args.output}")

def cmd_scan(args: argparse.Namespace) -> None:
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    K0, K1 = args.degree_range
    results = calibrate_multi_degree(img, K0, K1, method=args.method)
    stats = degree_error_stats(results)
    print(f"Reference degree: {K1}\n")
    header = "deg |   max_r   mean_r   std_r   |   max_b   mean_b   std_b"
    print(header)
    print("-" * len(header))
    for deg in sorted(stats):
        s = stats[deg]
        print(f"{deg:3d} | {s['max_r']:8.3f} {s['mean_r']:8.3f} {s['std_r']:8.3f} | "
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
