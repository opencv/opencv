"""
This file is part of OpenCV project.
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution and at http://opencv.org/license.html.

Copyright (C) 2025, Bigvision LLC.

MODNet Alpha Matting with OpenCV DNN

This sample demonstrates human portrait alpha matting using MODNet model.
MODNet is a trimap-free portrait matting method that can produce high-quality
alpha mattes for portrait images in real-time.

Reference:
    Github: https://github.com/ZHKKKe/MODNet

To download the MODNet model, run:
    python download_models.py modnet

Usage:
    python alpha_matting.py --input=image.jpg
"""

import cv2 as cv
import numpy as np
import argparse
import os
from common import *


def get_args_parser(func_args):
    backends = ("default", "openvino", "opencv", "vkcom", "cuda")
    targets = (
        "cpu",
        "opencl",
        "opencl_fp16",
        "ncs2_vpu",
        "hddl_vpu",
        "vulkan",
        "cuda",
        "cuda_fp16",
    )

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--zoo",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.yml"),
        help="An optional path to file with preprocessing parameters.",
    )
    parser.add_argument(
        "--input",
        default="messi5.jpg",
        help="Path to input image or video file. Defaults to messi5.jpg in samples/data.",
    )
    parser.add_argument(
        "--backend",
        default="default",
        type=str,
        choices=backends,
        help="Choose one of computation backends: "
        "default: automatically (by default), "
        "openvino: Intel's Deep Learning Inference Engine, "
        "opencv: OpenCV implementation, "
        "vkcom: VKCOM, "
        "cuda: CUDA",
    )
    parser.add_argument(
        "--target",
        default="cpu",
        type=str,
        choices=targets,
        help="Choose one of target computation devices: "
        "cpu: CPU target (by default), "
        "opencl: OpenCL, "
        "opencl_fp16: OpenCL fp16 (half-float precision), "
        "ncs2_vpu: NCS2 VPU, "
        "hddl_vpu: HDDL VPU, "
        "vulkan: Vulkan, "
        "cuda: CUDA, "
        "cuda_fp16: CUDA fp16 (half-float precision)",
    )

    args, _ = parser.parse_known_args()
    add_preproc_args(args.zoo, parser, "alpha_matting", "modnet")
    parser = argparse.ArgumentParser(
        parents=[parser],
        description="""
        To run:
            python alpha_matting.py --input=path/to/your/input/image

        Model path can also be specified using --model argument
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    return parser.parse_args(func_args)


def postprocess_output(image, alpha_output):
    """Process model output to create alpha mask."""
    h, w = image.shape[:2]

    alpha = alpha_output[0, 0] if alpha_output.ndim == 4 else alpha_output[0]
    alpha = cv.resize(alpha, (w, h))
    alpha = np.clip(alpha, 0, 1)

    alpha_mask = (alpha * 255).astype(np.uint8)

    return alpha_mask


def loadModel(args, engine):
    net = cv.dnn.readNetFromONNX(args.model, engine)
    net.setPreferableBackend(get_backend_id(args.backend))
    net.setPreferableTarget(get_target_id(args.target))
    return net


def draw_label(img, text, color):
    h, w = img.shape[:2]
    font_scale = max(h, w) / 1000.0
    thickness = 1
    text_size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = 10
    y = text_size[1] + 10
    cv.putText(img, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def apply_modnet(model, image):
    out = model.forward()
    alpha_mask = postprocess_output(image, out)
    alpha_3ch = cv.merge([alpha_mask / 255.0, alpha_mask / 255.0, alpha_mask / 255.0])
    composite = (image.astype(np.float32) * alpha_3ch).astype(np.uint8)
    return alpha_mask, composite


def main(func_args=None):
    args = get_args_parser(func_args)
    engine = cv.dnn.ENGINE_AUTO
    if args.backend != "default" or args.target != "cpu":
        engine = cv.dnn.ENGINE_CLASSIC

    image = cv.imread(cv.samples.findFile(args.input))
    if image is None:
        print("Failed to load the input image")
        exit(-1)

    cv.namedWindow("Input", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("Alpha Mask", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("Composite", cv.WINDOW_AUTOSIZE)
    cv.moveWindow("Alpha Mask", 200, 50)
    cv.moveWindow("Composite", 400, 50)

    args.model = findModel(args.model, args.sha1)
    net = loadModel(args, engine)

    inp = cv.dnn.blobFromImage(
        image, args.scale, (args.width, args.height), args.mean, swapRB=args.rgb
    )
    net.setInput(inp)
    alpha_mask, composite = apply_modnet(net, image)

    t, _ = net.getPerfProfile()
    label = "Inference time: %.2f ms" % (t * 1000.0 / cv.getTickFrequency())

    draw_label(image, label, (0, 255, 0))
    draw_label(alpha_mask, label, (255, 255, 255))
    draw_label(composite, label, (0, 255, 0))
    cv.imshow("Input", image)
    cv.imshow("Alpha Mask", alpha_mask)
    cv.imshow("Composite", composite)

    print("Press any key to exit")
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
