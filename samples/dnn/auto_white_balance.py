#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level
# directory of this distribution and at http://opencv.org/license.html.

'''
Auto white balance using FC4: https://github.com/yuanming-hu/fc4

Color constancy is a method to make colors of objects render correctly on a photo.
White balance aims to make white objects appear white on an image and not a shade of any
other color, independent of the actual light setting. White balance correction creates
a neutral looking coloring of the objects, and generally makes colors look more similar
to their 'true' colors under different light conditions.

Given an RGB image, the FC4 model predicts scene illuminant (R,G,B). We then apply
the illuminant to the image, applying the correction in the linear RGB space.
The transformation between linear and sRGB spaces is done as described in the sRGB standard,
which is a nonlinear Gamma correction with exponent 2.4 and extra handling of very small values.
This sample is written for 8bit images. The FC4 model accepts RGB images with applied Gamma scaling.

The training of the FC4 model was done on the Gehler-Shi dataset. The dataset includes
568 images and ground truth corrections, as well as ground truth illuminants. The linear
RGB images from the dataset were used with Gamma correction of 2.2 applied.

The model is a pretrained fold 0 of a training pipeline on the Gehler-Shi dataset, from the PyTorch
implementation of the FC4 algorithm by Mateo Rizzo. The model was converted from a .pth file to onnx
using torch.onnx.export. The model can be downloaded in the following link:
https://raw.githubusercontent.com/MykhailoTrushch/opencv/d6ab21353a87e4c527e38e464384c7ee78e96e22/samples/dnn/models/fc4_fold_0.onnx

Copyright (c) 2017 Yuanming Hu, Baoyuan Wang, Stephen Lin
Copyright (c) 2021 Matteo Rizzo

Licensed under the MIT license.

References:

Yuanming Hu, Baoyuan Wang, and Stephen Lin. “FC⁴: Fully Convolutional Color
Constancy with Confidence-Weighted Pooling.” CVPR, 2017, pp. 4085–4094.

Implementations of FC4:
https://github.com/yuanming-hu/fc4/
https://github.com/matteo-rizzo/fc4-pytorch

Lilong Shi and Brian Funt, "Re-processed Version of the Gehler Color
Constancy Dataset of 568 Images," accessed from http://www.cs.sfu.ca/~colour/data/

“IEC 61966-2-1:1999 – Multimedia Systems and Equipment – Colour Measurement and Management –
Part 2-1: Colour Management – Default RGB Colour Space – sRGB.” IEC Standard, 1999.
'''

import argparse
import sys
import numpy as np
import cv2 as cv

from common import *


# Normalization constant for 8bit values
NORMALIZE_FACTOR = 1.0 / 255.0

# sRGB to linear conversion constants (or vice versa):
# SRGB_THRESHOLD / LINEAR_THRESHOLD: breakpoints between linear and gamma regions
# SRGB_SLOPE: slope of the linear segment near black
# SRGB_ALPHA: offset to ensure continuity at the threshold
# SRGB_EXP: gamma exponent
SRGB_THRESHOLD = 0.04045
SRGB_ALPHA     = 0.055
SRGB_SLOPE     = 12.92
SRGB_EXP       = 2.4
LINEAR_THRESHOLD = 0.0031308
EPS = 1e-10

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    low  = rgb / SRGB_SLOPE
    high = np.power((rgb + SRGB_ALPHA) / (1.0 + SRGB_ALPHA), SRGB_EXP, dtype=np.float32)
    return np.where(rgb <= SRGB_THRESHOLD, low, high).astype(np.float32)

def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    low  = lin * SRGB_SLOPE
    high = (1.0 + SRGB_ALPHA) * np.power(lin, 1.0 / SRGB_EXP, dtype=np.float32) - SRGB_ALPHA
    return np.where(lin <= LINEAR_THRESHOLD, low, high).astype(np.float32)

def correct(bgr8u: np.ndarray, illum_rgb_linear: np.ndarray) -> np.ndarray:
    assert bgr8u.dtype == np.uint8 and bgr8u.ndim == 3 and bgr8u.shape[2] == 3

    bgr = bgr8u.astype(np.float32) * NORMALIZE_FACTOR
    lin = srgb_to_linear(bgr)
    e_r = max(float(illum_rgb_linear[0]), EPS)
    e_g = max(float(illum_rgb_linear[1]), EPS)
    e_b = max(float(illum_rgb_linear[2]), EPS)
    s3 = np.float32(np.sqrt(3.0))
    corr_bgr = np.array([e_b * s3 + EPS,
                         e_g * s3 + EPS,
                         e_r * s3 + EPS],
                        dtype=np.float32)

    corrected = lin / corr_bgr.reshape(1, 1, 3)

    max_val = float(corrected.max()) + EPS
    corrected /= max_val
    corrected = np.clip(corrected, 0.0, 1.0)

    srgb = linear_to_srgb(corrected)

    out_bgr8 = (srgb * 255.0 + 0.5).astype(np.uint8)
    return out_bgr8

def annotate(img_bgr: np.ndarray, title: str) -> None:
    fs = max(0.5, min(img_bgr.shape[1], img_bgr.shape[0]) / 800.0)
    th = max(1, int(round(fs * 2)))
    cv.putText(img_bgr, title, (10, 30), cv.FONT_HERSHEY_SIMPLEX, fs, (0,255,0), th)

def get_args_parser(func_args):
    backends = ("default", "openvino", "opencv", "vkcom", "cuda", "webnn")
    targets = ("cpu", "opencl", "opencl_fp16", "ncs2_vpu", "hddl_vpu", "vulkan",
               "cuda", "cuda_fp16")

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--zoo',
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                   help='An optional path to file with preprocessing parameters.')
    p.add_argument("--input", help="Path to input image", default="castle.png")
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

    args, _ = p.parse_known_args()
    add_preproc_args(args.zoo, p, 'auto_white_balance', prefix="", alias="fc4")
    p = argparse.ArgumentParser(
        parents=[p],
        description="FC4 Color Constancy (ONNX): " \
        "predicts illuminant and applies white balance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    return p.parse_args(func_args)



def main(func_args=None):
    args = get_args_parser(func_args)
    args.model = findModel(args.model, args.sha1)

    try:
        net = cv.dnn.readNetFromONNX(args.model)
        net.setPreferableBackend(get_backend_id(args.backend))
        net.setPreferableTarget(get_target_id(args.target))
    except cv.error as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    img = cv.imread(findFile(args.input), cv.IMREAD_COLOR)
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

    illum = out.astype(np.float32).reshape(-1)
    if out.size != 3:
        print("Error: model output of size not equal to 3 (should output 3 illuminants in RGB order)")
        sys.exit(-1)

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
