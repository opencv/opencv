#!/usr/bin/env python
'''
This sample using FlowNet v2 model to calculate optical flow.

You can download the converted .caffemodel model from https://drive.google.com/open?id=16qvE9VNmU39NttpZwZs81Ga8VYQJDaWZ
or convert .h5 model to .caffemodel yourself.
Download .prototxt from https://drive.google.com/open?id=19bo6SWU2p8ZKvjXqMKiCPdK8mghwDy9b
or modify original Flownetv2.prototxt - change ChannelNorm layers to series of layers as in link above.

Original paper: https://arxiv.org/abs/1612.01925
Original repo:  https://github.com/lmb-freiburg/flownet2
To get original model:
    wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/resources/binaries/flownet2/flownet2-models.tar.gz
'''

import argparse
import os.path
import numpy as np
import cv2 as cv


class OpticalFlow(object):
    def __init__(self, proto, model, height, width):
        self.net = cv.dnn.readNet(proto, model)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.height = height
        self.width = width

    def compute_flow(self, first_img, second_img):
        inp0 = cv.dnn.blobFromImage(first_img, size=(self.width, self.height))
        inp1 = cv.dnn.blobFromImage(second_img, size=(self.width, self.height))

        self.net.setInput(inp0, "img0")
        self.net.setInput(inp1, "img1")

        flow = self.net.forward()
        flow = flow.squeeze(0)
        mag, ang = cv.cartToPolar(flow[0, ...], flow[1, ...])

        hsv = np.zeros((self.height, self.width, first_img.shape[-1]), dtype=np.float32)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / (2 * np.pi)
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        return bgr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use this script to calculate optical flow using FlowNetv2',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', '-i', required=True, help='Path to input video file. Skip this argument to capture frames from a camera.')
    parser.add_argument('--height', default=320, help='Input height')
    parser.add_argument('--width',  default=448, help='Input width')
    parser.add_argument('--proto', '-p', default='FlowNet2_deploy.prototxt', help='Path to prototxt.')
    parser.add_argument('--model', '-m', default='FlowNet2_weights.caffemodel', help='Path to caffemodel.')
    args, _ = parser.parse_known_args()

    if not os.path.isfile(args.model) or not os.path.isfile(args.proto):
        raise OSError("Prototxt or caffemodel not exist")

    winName = 'Calculation optical flow in OpenCV'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)
    cap = cv.VideoCapture(args.input if args.input else 0)
    hasFrame, first_frame = cap.read()
    opt_flow = OpticalFlow(args.proto, args.model, args.height, args.width)
    while cv.waitKey(1) < 0:
        hasFrame, second_frame = cap.read()
        if not hasFrame:
            break
        flow = opt_flow.compute_flow(first_frame, second_frame)
        first_frame = second_frame
        cv.imshow(winName, flow)
