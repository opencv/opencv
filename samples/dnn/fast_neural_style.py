from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from common import get_engine_id, get_backend_id, get_target_id, print_dnn_config

parser = argparse.ArgumentParser(
        description='This script is used to run style transfer models from '
                    'https://github.com/onnx/models/tree/main/vision/style_transfer/fast_neural_style using OpenCV')
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--model', help='Path to .onnx model')
parser.add_argument('--width', default=-1, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=-1, type=int, help='Resize input to specific height.')
parser.add_argument('--median_filter', default=0, type=int, help='Kernel size of postprocessing blurring.')
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
args = parser.parse_args()

engine = get_engine_id(args.engine)
net = cv.dnn.readNetFromONNX(cv.samples.findFile(args.model), engine=engine)
net.setPreferableBackend(get_backend_id(args.backend))
net.setPreferableTarget(get_target_id(args.target))
print_dnn_config(net)

if args.input:
    cap = cv.VideoCapture(args.input)
else:
    cap = cv.VideoCapture(0)

cv.namedWindow('Styled image', cv.WINDOW_NORMAL)
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    inWidth = args.width if args.width != -1 else frame.shape[1]
    inHeight = args.height if args.height != -1 else frame.shape[0]
    inp = cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight),
                               swapRB=True, crop=False)

    net.setInput(inp)
    t0 = cv.getTickCount()
    out = net.forward()
    t = (cv.getTickCount() - t0) / cv.getTickFrequency()

    out = out.reshape(3, out.shape[2], out.shape[3])
    out = out.transpose(1, 2, 0)

    print('%.2f ms' % (t * 1000.0))

    if args.median_filter:
        out = cv.medianBlur(out, args.median_filter)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    cv.imshow('Styled image', out)
