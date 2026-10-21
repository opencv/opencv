#!/usr/bin/env python
'''
This sample uses the RAFT model to calculate optical flow.

RAFT Original Paper: https://arxiv.org/pdf/2003.12039.pdf
RAFT Repo: https://github.com/princeton-vl/RAFT

Download the .onnx model from here https://github.com/opencv/opencv_zoo/raw/281d232cd99cd920853106d853c440edd35eb442/models/optical_flow_estimation_raft/optical_flow_estimation_raft_2023aug.onnx.

Note: the legacy FlowNet v2 Caffe pipeline (--proto/.caffemodel) has been removed together
with the Caffe importer. Please provide a single ONNX model.
'''

import argparse
import os.path
import numpy as np
import cv2 as cv
from common import get_engine_id, get_backend_id, get_target_id, print_dnn_config


class OpticalFlow(object):
    def __init__(self, model, height, width, proto="", engine=0, backend=0, target=0):
        if proto:
            raise cv.error("Caffe support has been removed. Please provide a single ONNX model path (e.g. RAFT).")
        self.net = cv.dnn.readNetFromONNX(model, engine=engine)
        self.net.setPreferableBackend(backend)
        self.net.setPreferableTarget(target)
        print_dnn_config(self.net)
        self.height = height
        self.width = width

    def compute_flow(self, first_img, second_img):
        inp0 = cv.dnn.blobFromImage(first_img, size=(self.width, self.height))
        inp1 = cv.dnn.blobFromImage(second_img, size=(self.width, self.height))
        self.net.setInputsNames(["img0", "img1"])
        self.net.setInput(inp0, "img0")
        self.net.setInput(inp1, "img1")

        flow = self.net.forward()
        output = self.motion_to_color(flow)
        return output

    def motion_to_color(self, flow):
        arr = np.arange(0, 255, dtype=np.uint8)
        colormap = cv.applyColorMap(arr, cv.COLORMAP_HSV)
        colormap = colormap.squeeze(1)

        flow = flow.squeeze(0)
        fx, fy = flow[0, ...], flow[1, ...]
        rad = np.sqrt(fx**2 + fy**2)
        maxrad = rad.max() if rad.max() != 0 else 1

        ncols = arr.size
        rad = rad[..., np.newaxis] / maxrad
        a = np.arctan2(-fy / maxrad, -fx / maxrad) / np.pi
        fk = (a + 1) / 2.0 * (ncols - 1)
        k0 = fk.astype(np.int32)
        k1 = (k0 + 1) % ncols
        f = fk[..., np.newaxis] - k0[..., np.newaxis]

        col0 = colormap[k0] / 255.0
        col1 = colormap[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        col = np.where(rad <= 1, 1 - rad * (1 - col), col * 0.75)
        output = (255.0 * col).astype(np.uint8)
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use this script to calculate optical flow',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', '-i', required=True, help='Path to input video file. Skip this argument to capture frames from a camera.')
    parser.add_argument('--height', default=320, type=int, help='Input height')
    parser.add_argument('--width', default=448, type=int, help='Input width')
    parser.add_argument('--model', '-m', required=True, help='Path to a single ONNX model (e.g. RAFT).')
    parser.add_argument('--backend', default='default', type=str,
                        choices=['default', 'openvino', 'opencv', 'vkcom', 'cuda'],
                        help="Choose one of computation backends: "
                             "default: automatically (by default), "
                             "openvino: Intel's Deep Learning Deployment Toolkit (has been deprecated), "
                             "opencv: OpenCV's built-in engine, "
                             "vkcom: Vulkan-based backend, "
                             "cuda: CUDA backend")
    parser.add_argument('--target', default='cpu', type=str,
                        choices=['cpu', 'opencl', 'opencl_fp16', 'ncs2_vpu', 'hddl_vpu', 'vulkan', 'cuda', 'cuda_fp16'],
                        help='Choose one of target devices: '
                             'cpu: CPU (by default), '
                             'opencl: OpenCL, '
                             'opencl_fp16: OpenCL FP16 (half-precision), '
                             'ncs2_vpu: NCS2 VPU, '
                             'hddl_vpu: HDDL VPU, '
                             'vulkan: Vulkan, '
                             'cuda: CUDA, '
                             'cuda_fp16: CUDA FP16')
    parser.add_argument('--engine', default='auto', type=str,
                        choices=['auto', 'opencv', 'ort'],
                        help='Choose DNN engine: '
                             'auto: automatically select engine (default), '
                             'opencv: OpenCV built-in DNN engine, '
                             'ort: ONNX Runtime (requires build with WITH_ONNXRUNTIME=ON)')
    args, _ = parser.parse_known_args()

    if not os.path.isfile(args.model):
        raise OSError("Model does not exist")

    winName = 'Calculation optical flow in OpenCV'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)
    cap = cv.VideoCapture(args.input if args.input else 0)
    hasFrame, first_frame = cap.read()

    engine = get_engine_id(args.engine)
    backend = get_backend_id(args.backend)
    target = get_target_id(args.target)
    opt_flow = OpticalFlow(args.model, 360, 480, engine=engine, backend=backend, target=target)

    while cv.waitKey(1) < 0:
        hasFrame, second_frame = cap.read()
        if not hasFrame:
            break
        flow = opt_flow.compute_flow(first_frame, second_frame)
        first_frame = second_frame
        cv.imshow(winName, flow)
