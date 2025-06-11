#!/usr/bin/env python
'''
This file is part of OpenCV project.
It is subject to the license terms in the LICENSE file found in the top-level directory
of this distribution and at http://opencv.org/license.html.

This sample deblurs the given blurry image.

Copyright (C) 2025, Bigvision LLC.

How to use:
    Sample command to run:
        `python deblurring.py`

    You can download NAFNet deblurring model using
        `python download_models.py NAFNet`

    References:
      Github: https://github.com/megvii-research/NAFNet
      PyTorch model: https://drive.google.com/file/d/14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X/view

      PyTorch model was converted to ONNX and then ONNX model was further quantized using block quantization from [opencv_zoo](https://github.com/opencv/opencv_zoo/blob/main/tools/quantize/block_quantize.py)

    Set environment variable OPENCV_DOWNLOAD_CACHE_DIR to point to the directory where models are downloaded. Also, point OPENCV_SAMPLES_DATA_PATH to opencv/samples/data.
'''

import argparse
import cv2 as cv
import numpy as np
from common import *

def help():
    print(
        '''
        Use this script for image deblurring using OpenCV.

        Firstly, download required models i.e. NAFNet using `download_models.py` (if not already done). Set environment variable OPENCV_DOWNLOAD_CACHE_DIR to specify where models should be downloaded. Also, point OPENCV_SAMPLES_DATA_PATH to opencv/samples/data.

        To run:
        Example: python deblurring.py [--input=<image_name>]

        Deblurring model path can also be specified using --model argument.
        '''
    )

def get_args_parser():
    backends = ("default", "openvino", "opencv", "vkcom", "cuda")
    targets = ("cpu", "opencl", "opencl_fp16", "ncs2_vpu", "hddl_vpu", "vulkan", "cuda", "cuda_fp16")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                        help='An optional path to file with preprocessing parameters.')
    parser.add_argument('--input', '-i', default="licenseplate_motion.jpg", help='Path to image file.', required=False)
    parser.add_argument('--backend', default="default", type=str, choices=backends,
            help="Choose one of computation backends: "
            "default: automatically (by default), "
            "openvino: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
            "opencv: OpenCV implementation, "
            "vkcom: VKCOM, "
            "cuda: CUDA, "
            "webnn: WebNN")
    parser.add_argument('--target', default="cpu", type=str, choices=targets,
            help="Choose one of target computation devices: "
            "cpu: CPU target (by default), "
            "opencl: OpenCL, "
            "opencl_fp16: OpenCL fp16 (half-float precision), "
            "ncs2_vpu: NCS2 VPU, "
            "hddl_vpu: HDDL VPU, "
            "vulkan: Vulkan, "
            "cuda: CUDA, "
            "cuda_fp16: CUDA fp16 (half-float preprocess)")
    args, _ = parser.parse_known_args()
    add_preproc_args(args.zoo, parser, 'deblurring', prefix="", alias="NAFNet")
    parser = argparse.ArgumentParser(parents=[parser],
                                        description='Image deblurring using OpenCV.',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser.parse_args()

def main():
    if hasattr(args, 'help'):
        help()
        exit(1)

    args.model = findModel(args.model, args.sha1)

    engine = cv.dnn.ENGINE_AUTO

    if args.backend != "default" or args.target != "cpu":
        engine = cv.dnn.ENGINE_CLASSIC

    net = cv.dnn.readNetFromONNX(args.model, engine)
    net.setPreferableBackend(get_backend_id(args.backend))
    net.setPreferableTarget(get_target_id(args.target))

    input_image = cv.imread(findFile(args.input))
    image = input_image.copy()
    height, width = image.shape[:2]

    image_blob = cv.dnn.blobFromImage(image, args.scale, (width, height), args.mean, args.rgb, False)
    net.setInput(image_blob)
    out = net.forward()

    # Postprocessing
    output = out[0]
    output = np.transpose(output, (1, 2, 0))
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    out_image = cv.cvtColor(output, cv.COLOR_RGB2BGR)

    cv.imshow("input image: ", input_image)
    cv.imshow("output image: ", out_image)
    cv.waitKey(0)

if __name__ == '__main__':
    args = get_args_parser()
    main()
