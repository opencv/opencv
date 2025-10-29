"""
This file is part of OpenCV project.
It is subject to the license terms in the LICENSE file found in the top-level directory
of this distribution and at http://opencv.org/license.html.

Copyright (C) 2025, Bigvision LLC.


This sample demonstrates super-resolution using the SeeMoreDetails model.
The model upscales images by 4x while enhancing details and reducing noise.
Supports image inputs only.

SeeMoreDetails Repo: https://github.com/eduardzamfir/seemoredetails
"""

import cv2 as cv
import argparse
import numpy as np
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
        "--input", help="Path to input image file.", default="chicky_512.png", required=False
    )
    parser.add_argument(
        "--backend",
        default="default",
        type=str,
        choices=backends,
        help="Choose one of computation backends: "
        "default: automatically (by default), "
        "openvino: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
        "opencv: OpenCV implementation, "
        "vkcom: VKCOM, "
        "cuda: CUDA, "
        "webnn: WebNN",
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
        "cuda_fp16: CUDA fp16 (half-float preprocess)",
    )

    args, _ = parser.parse_known_args()

    model_name = "seemoredetails"
    add_preproc_args(args.zoo, parser, "super_resolution", model_name)

    parser = argparse.ArgumentParser(
        parents=[parser],
        description="""
        To run:
            Default image:
                python super_resolution.py
            Image processing:
                python super_resolution.py --input=path/to/your/input/image.jpg

        The model performs 4x super-resolution on input images.
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    return parser.parse_args(func_args)

def load_model(args):
    """Load the super-resolution model"""
    try:
        model_path = findModel(args.model, args.sha1)
        net = cv.dnn.readNetFromONNX(model_path)
        net.setPreferableBackend(get_backend_id(args.backend))
        net.setPreferableTarget(get_target_id(args.target))
        return net
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def postprocess_output(output, args, original_shape=None):
    """Postprocess model output to displayable image"""
    output = np.squeeze(output, axis=0)
    output = np.clip(output, 0, 1)
    output = np.transpose(output, (1, 2, 0))
    output = (output * 255).astype(np.uint8)

    output = cv.cvtColor(output, cv.COLOR_RGB2BGR)

    if original_shape is not None:
        target_height, target_width = original_shape
        upscaled_height, upscaled_width = target_height * 4, target_width * 4
        output = cv.resize(output, (upscaled_width, upscaled_height))

    return output

def apply_super_resolution(net, image, args):
    """Apply super-resolution to a single image"""
    original_shape = image.shape[:2]

    blob = cv.dnn.blobFromImage(
        image,
        scalefactor=args.scale,
        size=(args.width, args.height),
        mean=args.mean,
        swapRB=args.rgb,
        crop=False,
    )

    net.setInput(blob)
    output = net.forward()

    result = postprocess_output(output, args, original_shape)

    t, _ = net.getPerfProfile()
    label = "Inference time: %.2f ms" % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(result, label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return result

def main(func_args=None):
    args = get_args_parser(func_args)

    net = load_model(args)
    if net is None:
        print("Failed to load model.")
        return -1

    input_path = cv.samples.findFile(args.input)
    image = cv.imread(input_path)
    if image is None:
        print(f"Cannot load image: {input_path}")
        return -1

    print(f"Processing image: {input_path}")
    result = apply_super_resolution(net, image, args)

    cv.namedWindow("Input", cv.WINDOW_NORMAL)
    cv.namedWindow("Super-Resolution Result", cv.WINDOW_NORMAL)
    cv.imshow("Input", image)
    cv.imshow("Super-Resolution Result", result)
    print("Press 'q' to quit...")
    while True:
        key = cv.waitKey(0) & 0xFF
        if key == ord("q"):
            break
    cv.destroyAllWindows()
    return 0

if __name__ == "__main__":
    main()
