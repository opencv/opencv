#!/usr/bin/env python
'''
This sample using FlowNet v2 model to calculate optical flow.
Original paper: https://arxiv.org/abs/1612.01925
Original repo:  https://github.com/lmb-freiburg/flownet2
To get original model:
    wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/resources/binaries/flownet2/flownet2-models.tar.gz
    tar xvzf flownet2-models.tar.gz       

To run model in OpenCV modify Flownetv2.prototxt.

change ChannelNorm layers to series of layers:
Remove:
    layer {
    name: "ChannelNorm_1"
    type: "ChannelNorm"
    bottom: "input"
    top: "output"
    }
Add:
    layer {
    name: "Sqr"
    type: "Power"
    bottom: "input"
    top: "sqr"
    power_param {
        power: 2
    }
    }
    layer{
        type: "Reshape"
        name: "reshape"
        bottom: "sqr"
        reshape_param{
            shape{
                dim: 1
                dim: 1
                dim: C # number of channels in input layer
                dim: -1
            }
            axis: 0
            num_axes: -1
        }
        top: "reshape"
    }
    layer {
    type: "Pooling"
    pooling_param{
        pool: AVE
        pad_h: 0
        pad_w: 0
        kernel_w: 1
        kernel_h: C
        stride_h: 1
        stride_w: 1
    }
    name: "AvePool"
    bottom: "reshape"
    top: "ave_pool"
    }
    layer {
    name: "Mul"
    type: "Power"
    bottom: "ave_pool"
    top: "mul"
    power_param {
        scale: C
        power: 0.5
    }
    }
    layer{
        type: "Reshape"
        name: "reshape_back"
        bottom: "mul"
        reshape_param{
            shape{
                dim: 1
                dim: 1
                dim: H # input height
                dim: -1
            }
            axis: 0
            num_axes: -1
        }
        top: "output"
    }
'''

import argparse
import os.path
import numpy as np
import cv2 as cv


class OpticalFlow(object):
    def __init__(self, proto, model, height, width, backend, target):
        self.net = cv.dnn.readNet(proto, model)
        self.net.setPreferableBackend(backend)
        self.net.setPreferableTarget(target)
        self.height = height
        self.width = width

    def compute_flow(self, first_img, second_img):
        inp0 = cv.dnn.blobFromImage(first_img, 1.0, (self.width, self.height))
        inp1 = cv.dnn.blobFromImage(second_img, 1.0, (self.width, self.height))

        self.net.setInput(inp0, "img0")
        self.net.setInput(inp1, "img1")

        flow = self.net.forward()
        flow = flow.squeeze(0)
        mag, ang = cv.cartToPolar(flow[0, ...], flow[1, ...])

        hsv = np.zeros((self.height, self.width, first_img.shape[-1]), dtype=np.float32)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / (2 * np.pi)
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
        return rgb

  
if __name__ == '__main__':
    backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv.dnn.DNN_BACKEND_OPENCV)
    targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD)

    parser = argparse.ArgumentParser(description='Use this script to calculate optical flow using FlowNetv2',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', '-i', required=True, help='Path to input video file. Skip this argument to capture frames from a camera.')
    parser.add_argument('--height', default=320, help='Input height')
    parser.add_argument('--width',  default=448, help='Input width')
    parser.add_argument('--proto', '-p', default='FlowNet2_deploy.prototxt', help='Path to prototxt.')
    parser.add_argument('--model', '-m', default='FlowNet2_weights.caffemodel', help='Path to caffemodel.')
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help="Choose one of computation backends: "
                             "%d: automatically (by default), "
                             "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                             "%d: OpenCV implementation" % backends)
    parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                        help='Choose one of target computation devices: '
                             '%d: CPU target (by default), '
                             '%d: OpenCL, '
                             '%d: OpenCL fp16 (half-float precision), '
                             '%d: VPU' % targets)
    args, _ = parser.parse_known_args()

    if not os.path.isfile(args.model) or not os.path.isfile(args.proto):
        raise OSError("Prototxt or caffemodel not exist")

    winName = 'Calculation optical flow in OpenCV'
    cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
    cap = cv.VideoCapture(args.input if args.input else 0)
    hasFrame, first_frame = cap.read()
    opt_flow = OpticalFlow(args.proto, args.model, args.height, args.width, args.backend, args.target)
    while cv.waitKey(1) < 0:
        hasFrame, second_frame = cap.read()
        if not hasFrame:
            break
        flow = opt_flow.compute_flow(first_frame, second_frame)
        first_frame = second_frame
        cv.imshow(winName, flow)
