#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level
# directory of this distribution and at http://opencv.org/license.html.
#
# Auto white balance using FC4: https://github.com/yuanming-hu/fc4
#
# Given an RGB image, the FC4 model predicts scene illuminant (R,G,B). We then apply
# the illuminant to the image, applying the correction in the linear RGB space.
#
# Yuanming Hu, Baoyuan Wang, and Stephen Lin. “FC⁴: Fully Convolutional Color
# Constancy with Confidence-Weighted Pooling.” CVPR, 2017, pp. 4085–4094.

import argparse
import sys
import numpy as np
import cv2 as cv

from common import *


def extract_illuminant(out: np.ndarray) -> np.ndarray:
    flat = out.astype(np.float32).reshape(-1)
    assert flat.size >= 3, "ONNX output has fewer than 3 values"
    return flat[:3]

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    a = 0.055
    low  = rgb / 12.92
    high = np.power((rgb + a) / (1.0 + a), 2.4)
    return np.where(rgb <= 0.04045, low, high).astype(np.float32)

def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    a = 0.055
    low  = lin * 12.92
    high = (1.0 + a) * np.power(lin, 1.0/2.4) - a
    return np.where(lin <= 0.0031308, low, high).astype(np.float32)

def correct(bgr8u: np.ndarray, illum_rgb_linear: np.ndarray) -> np.ndarray:
    assert bgr8u.dtype == np.uint8 and bgr8u.ndim == 3 and bgr8u.shape[2] == 3
    rgb = cv.cvtColor(bgr8u, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lin = srgb_to_linear(rgb)

    s3 = np.sqrt(3.0).astype(np.float32)
    corr = (illum_rgb_linear.astype(np.float32) * s3) + 1e-10  # [R,G,B]
    corrected = lin / corr.reshape(1, 1, 3)

    max_val = corrected.max() + 1e-10
    normalized = corrected / max_val

    srgb = np.clip(linear_to_srgb(normalized), 0.0, 1.0)
    out_rgb8 = (srgb * 255.0 + 0.5).astype(np.uint8)
    return cv.cvtColor(out_rgb8, cv.COLOR_RGB2BGR)

def annotate(img_bgr: np.ndarray, title: str) -> None:
    fs = max(0.5, min(img_bgr.shape[1], img_bgr.shape[0]) / 800.0)
    th = max(1, int(round(fs * 2)))
    cv.putText(img_bgr, title, (10, 30), cv.FONT_HERSHEY_SIMPLEX, fs, (0,255,0), th)



def main():
    backends = ("default", "openvino", "opencv", "vkcom", "cuda")
    targets = ("cpu", "opencl", "opencl_fp16", "ncs2_vpu", "hddl_vpu", "vulkan", "cuda", "cuda_fp16")

    p = argparse.ArgumentParser(
        description="FC4 Color Constancy (ONNX): " \
        "predicts illuminant and applies Von Kries white balance."
    )
    p.add_argument("--model", required=True, help="Path to ONNX model file")
    p.add_argument("--input", required=True, help="Path to input image")
    p.add_argument("--scale", type=float, default=1.0, help="Input scaling factor (e.g., 1/255)")
    p.add_argument("--mean", type=float, nargs=3, default=0.0,
                   help="Mean to subtract (B G R)")
    p.add_argument("--rgb", action="store_true", help="Swap BGR->RGB for model input")
    p.add_argument('--backend', default="default", type=str, choices=backends,
            help="Choose one of computation backends: "
            "default: automatically (by default), "
            "openvino: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
            "opencv: OpenCV implementation, "
            "vkcom: VKCOM, "
            "cuda: CUDA, "
            "webnn: WebNN")
    p.add_argument('--target', default="cpu", type=str, choices=targets,
            help="Choose one of target computation devices: "
            "cpu: CPU target (by default), "
            "opencl: OpenCL, "
            "opencl_fp16: OpenCL fp16 (half-float precision), "
            "ncs2_vpu: NCS2 VPU, "
            "hddl_vpu: HDDL VPU, "
            "vulkan: Vulkan, "
            "cuda: CUDA, "
            "cuda_fp16: CUDA fp16 (half-float preprocess)")
    args = p.parse_args()

    try:
        net = cv.dnn.readNetFromONNX(args.model)
        net.setPreferableBackend(get_backend_id(args.backend))
        net.setPreferableTarget(get_target_id(args.target))
    except cv.error as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    img = cv.imread(args.input, cv.IMREAD_COLOR)
    if img is None:
        print(f"Cannot load image: {args.input}", file=sys.stderr)
        sys.exit(1)

    blob = cv.dnn.blobFromImage(
        img, scalefactor=args.scale, size=(img.shape[1], img.shape[0]),
        mean=args.mean, swapRB=args.rgb, crop=False, ddepth=cv.CV_32F
    )
    net.setInput(blob)

    try:
        out = net.forward()
    except cv.error as e:
        print(f"Forward error: {e}", file=sys.stderr)
        sys.exit(1)

    illum = extract_illuminant(out)
    corrected = correct(img, illum)

    orig_vis = img.copy()
    corr_vis = corrected.copy()
    annotate(orig_vis, "Original")
    annotate(corr_vis, "FC4-corrected")
    stacked = np.hstack([orig_vis, corr_vis])
    cv.imshow("Original and Corrected Images", stacked)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
