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
    ca_calibration.py calibrate [-h] [--degree DEGREE] --yaml YAML image
    ca_calibration.py correct [-h] --coeff COEFF [-o OUTPUT] image
    ca_calibration.py full [-h] [--degree DEGREE] --yaml YAML [-o OUTPUT] image

usage example:
    ca_calibration.py calibrate pattern_aberrated.png --yaml poly.yaml

default values:
    --degree: 11
    -o, --output: corrected.png
'''

from __future__ import annotations

import argparse
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, Tuple, List, TYPE_CHECKING, Optional

import cv2
import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.spatial import cKDTree

try:
    from cv2.typing import MatLike
except ModuleNotFoundError:
    MatLike = np.ndarray

__all__ = [
    "CalibrationResult",
    "detect_circles",
    "fit_polynomials",
]

@dataclass
class Polynomial2D:
    coeffs_x: np.ndarray
    coeffs_y: np.ndarray
    degree: int
    mean_x: float
    mean_y: float
    std_x: float
    std_y: float

    def delta(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_n = (x - self.mean_x) / self.std_x
        y_n = (y - self.mean_y) / self.std_y
        terms = _monomial_terms(x_n, y_n, self.degree)
        dx = terms @ self.coeffs_x
        dy = terms @ self.coeffs_y
        return dx.reshape(x.shape), dy.reshape(y.shape)

# Register a representer that forces flow-style for every list
def _repr_flow_seq(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq',
                                     data,
                                     flow_style=True)

yaml.SafeDumper.add_representer(list, _repr_flow_seq)

@dataclass
class CalibrationResult:
    degree: int
    poly_red: Polynomial2D
    poly_blue: Polynomial2D
    mean_x_red: float
    std_x_red: float  
    mean_y_red: float
    std_y_red: float
    mean_x_blue: float
    std_x_blue: float
    mean_y_blue: float
    std_y_blue: float
    image_width: int
    image_height: int
    rms_red: Optional[float] = None
    rms_blue: Optional[float] = None

    def to_dict(self) -> Dict:
        d = {
            "degree": self.degree,
            "image_resolution": {
                "width": int(self.image_width),
                "height": int(self.image_height)
            },
            "red_channel": {
                "coeffs_x": self.poly_red.coeffs_x.tolist(),
                "coeffs_y": self.poly_red.coeffs_y.tolist(),
                "mean_x": float(self.mean_x_red),
                "std_x": float(self.std_x_red),
                "mean_y": float(self.mean_y_red),
                "std_y": float(self.std_y_red)
            },
            "blue_channel": {
                "coeffs_x": self.poly_blue.coeffs_x.tolist(),
                "coeffs_y": self.poly_blue.coeffs_y.tolist(),
                "mean_x": float(self.mean_x_blue),
                "std_x": float(self.std_x_blue),
                "mean_y": float(self.mean_y_blue),
                "std_y": float(self.std_y_blue)
            }
        }
        if self.rms_red is not None:
            d["red_channel"]["rms"] = float(self.rms_red)
            d["blue_channel"]["rms"] = float(self.rms_blue)
        return d

    @classmethod
    def from_file(cls, path: str | pathlib.Path):
        path = pathlib.Path(path)
        with path.open("r") as fh:
            if path.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.safe_load(fh)
            else:
                raise ValueError("YAML file expected as input for CalibrationResult")
        
        deg = data["degree"]
        red_data = data["red_channel"]
        blue_data = data["blue_channel"]
        resolution = data["image_resolution"]
        
        poly_r = Polynomial2D(
            np.asarray(red_data["coeffs_x"]),
            np.asarray(red_data["coeffs_y"]),
            deg,
            red_data["mean_x"], red_data["std_x"], red_data["mean_y"], red_data["std_y"]
        )
        poly_b = Polynomial2D(
            np.asarray(blue_data["coeffs_x"]),
            np.asarray(blue_data["coeffs_y"]),
            deg,
            blue_data["mean_x"], blue_data["std_x"], blue_data["mean_y"], blue_data["std_y"]
        )
        
        return cls(
            degree=deg,
            poly_red=poly_r,
            poly_blue=poly_b,
            mean_x_red=red_data["mean_x"],
            std_x_red=red_data["std_x"],
            mean_y_red=red_data["mean_y"],
            std_y_red=red_data["std_y"],
            mean_x_blue=blue_data["mean_x"],
            std_x_blue=blue_data["std_x"],
            mean_y_blue=blue_data["mean_y"],
            std_y_blue=blue_data["std_y"],
            image_width=resolution["width"],
            image_height=resolution["height"],
            rms_red=red_data.get("rms"),
            rms_blue=blue_data.get("rms"),
        )

    def save(self, path: str | None = None):
        d = self.to_dict()
        if path is not None:
            with open(path, "w") as fh:
                fh.write("%YAML:1.0\n")
                yaml.safe_dump(d, 
                               fh, 
                               default_flow_style=False,
                               sort_keys=False)



def _monomial_terms(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    x = x.flatten()
    y = y.flatten()
    terms: List[np.ndarray] = []
    cnt = 0
    for total in range(degree + 1):
        for i in range(total + 1):
            j = total - i
            terms.append((x ** i) * (y ** j))
            cnt += 1
    return np.vstack(terms).T


def _detect_disk_centres(
    img: MatLike,
    *,
    min_area: int = 50,
    max_area: int | None = None,
    circularity_thresh: float = 0.75,
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

    centres: list[tuple[float, float]] = []

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
        delta, *_ = np.linalg.lstsq(J, f, rcond=None)
        cx -= delta[0]
        cy -= delta[1]
        centres.append((cx, cy))

    if len(centres) == 0:
        raise RuntimeError("No valid disks detected, check function parameters")

    return np.asarray(centres, dtype=np.float32)


def _pair_keypoints(
    ref: np.ndarray,
    target: np.ndarray,
    max_error: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tree = cKDTree(ref)
    dists, idx = tree.query(target, distance_upper_bound=max_error)
    mask = np.isfinite(dists)
    target_valid = target[mask]
    ref_valid = ref[idx[mask]]
    disp = ref_valid - target_valid
    return target_valid[:, 0], target_valid[:, 1], disp


def _fit_channel(
    x: np.ndarray,
    y: np.ndarray,
    disp: np.ndarray,
    degree: int,
    method: str = "L-BFGS-B",
) -> Tuple[np.ndarray, np.ndarray, float]:
    mean_x, std_x, mean_y, std_y = x.mean(), x.std(), y.mean(), y.std()
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    terms = _monomial_terms(x, y, degree)
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

    res = minimize(objective, c0, method=method)
    coeffs_x = res.x[:m]
    coeffs_y = res.x[m:]
    rms = math.sqrt(res.fun / disp.size)
    return coeffs_x, coeffs_y, rms, mean_x, mean_y, std_x, std_y


def fit_polynomials(
    x_r: np.ndarray,
    y_r: np.ndarray,
    disp_r: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
    disp_b: np.ndarray,
    degree: int,
) -> Tuple[Polynomial2D, Polynomial2D, float, float]:
    crx, cry, rms_r, mean_x_red, mean_y_red, std_x_red, std_y_red = _fit_channel(x_r, y_r, disp_r, degree)
    cbx, cby, rms_b, mean_x_blue, mean_y_blue, std_x_blue, std_y_blue = _fit_channel(x_b, y_b, disp_b, degree)
    poly_r = Polynomial2D(crx, cry, 11, mean_x_red, mean_y_red, std_x_red, std_y_red)
    poly_b = Polynomial2D(cbx, cby, 11, mean_x_blue, mean_y_blue, std_x_blue, std_y_blue)
    return poly_r, poly_b, rms_r, rms_b, (mean_x_red, mean_y_red, std_x_red, std_y_red), (mean_x_blue, mean_y_blue, std_x_blue, std_y_blue)


def _calibrate_from_image(
    img: np.ndarray,
    degree: int = 11,
) -> CalibrationResult:
    h, w = img.shape[:2]
    b, g, r = cv2.split(img)

    pts_g = _detect_disk_centres(g)
    pts_r = _detect_disk_centres(r)
    pts_b = _detect_disk_centres(b)

    xr, yr, disp_r = _pair_keypoints(pts_g, pts_r)
    xb, yb, disp_b = _pair_keypoints(pts_g, pts_b)

    poly_r, poly_b, rms_r, rms_b, stats_r, stats_b = fit_polynomials(
        xr,
        yr,
        disp_r,
        xb,
        yb,
        disp_b,
        degree,
    )

    return CalibrationResult(
        degree=degree,
        poly_red=poly_r,
        poly_blue=poly_b,
        mean_x_red=stats_r[0],
        std_x_red=stats_r[1],
        mean_y_red=stats_r[2],
        std_y_red=stats_r[3],
        mean_x_blue=stats_b[0],
        std_x_blue=stats_b[1],
        mean_y_blue=stats_b[2],
        std_y_blue=stats_b[3],
        image_width=w,
        image_height=h,
        rms_red=rms_r,
        rms_blue=rms_b,
    )


def _build_remap(
    h: int,
    w: int,
    poly: Polynomial2D,
) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dx, dy = poly.delta(X, Y)
    map_x = (X - dx).astype(np.float32)
    map_y = (Y - dy).astype(np.float32)
    return map_x, map_y


def _correct_image(
    img: np.ndarray,
    calib: CalibrationResult,
) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("correct_image expects a BGR colour image")

    h, w = img.shape[:2]
    b, g, r = cv2.split(img)
    map_x_r, map_y_r = _build_remap(h, w, calib.poly_red)
    map_x_b, map_y_b = _build_remap(h, w, calib.poly_blue)

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chromatic‑aberration calibration and correction tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("calibrate", help="Calibrate from calibration target image")
    sc.add_argument("image", help="Image of black‑disk calibration target")
    sc.add_argument("--degree", type=int, default=11, help="Polynomial degree")
    sc.add_argument("--coeffs", required=True, help="Save coefficients to YAML file")

    sr = sub.add_parser("correct", help="Correct a photograph using saved coefficients")
    sr.add_argument("image", help="Input image to be corrected")
    sr.add_argument("--coeffs", required=True, help="Calibration coefficient file (.json/.yaml)")
    sr.add_argument("-o", "--output", default="corrected.png", help="Output filename")

    sf = sub.add_parser("full", help="Calibrate from calibration target image and correct the calibration target")
    sf.add_argument("image", help="Image of black‑disk calibration target")
    sf.add_argument("--degree", type=int, default=11, help="Polynomial degree")
    sf.add_argument("--coeffs", required=True, help="Save coefficients to YAML file")
    sf.add_argument("-o", "--output", default="corrected.png", help="Output filename")

    return p.parse_args()


def _cmd_calibrate(args: argparse.Namespace) -> None:
    calib = _calibrate_from_image(cv2.imread(args.image, cv2.IMREAD_COLOR), degree=args.degree)
    print(
        f"Calibrated polynomial degree {calib.degree}: RMS red={calib.rms_red:.3f} px, "
        f"blue={calib.rms_blue:.3f} px"
    )
    calib.save(path=args.coeffs)
    print("Saved coefficients to", args.coeffs)


def _cmd_correct(args: argparse.Namespace) -> None:
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)
    calib = CalibrationResult.from_file(args.coeffs)
    fixed = _correct_image(img, calib)
    cv2.imwrite(args.output, fixed)
    print(f"Corrected image written to {args.output}")

def _cmd_full(args: argparse.Namespace) -> None:
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)
    calib = _calibrate_from_image(img, degree=args.degree)
    print(
        f"Calibrated polynomial degree {calib.degree}: RMS red={calib.rms_red:.3f} px, "
        f"blue={calib.rms_blue:.3f} px"
    )
    calib.save(path=args.coeffs)
    print("Saved coefficients to", args.coeffs)
    fixed = _correct_image(img, calib)
    cv2.imwrite(args.output, fixed)
    print(f"Corrected image written to {args.output}")


if __name__ == "__main__":
    args = _parse_args()
    if args.cmd == "calibrate":
        _cmd_calibrate(args)
    elif args.cmd == "correct":
        _cmd_correct(args)
    elif args.cmd == "full":
        _cmd_full(args)
