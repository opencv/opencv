#!/usr/bin/env python
'''
This sample using FlowNet v2 model to calculate optical flow.
Original paper: https://arxiv.org/abs/1612.01925.
Original repo:  https://github.com/lmb-freiburg/flownet2.

Download the converted .caffemodel model from https://drive.google.com/open?id=16qvE9VNmU39NttpZwZs81Ga8VYQJDaWZ
and .prototxt from https://drive.google.com/file/d/1RyNIUsan1ZOh2hpYIH36A-jofAvJlT6a/view?usp=sharing.
Otherwise download original model from https://lmb.informatik.uni-freiburg.de/resources/binaries/flownet2/flownet2-models.tar.gz,
convert .h5 model to .caffemodel and modify original .prototxt using .prototxt from link above.
'''

import argparse
import os.path
import numpy as np
import cv2 as cv


class OpticalFlow(object):
    def __init__(self, proto, model, height, width):
        self.net = cv.dnn.readNetFromCaffe(proto, model)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.height = height
        self.width = width

    def compute_flow(self, first_img, second_img):
        inp0 = cv.dnn.blobFromImage(first_img, size=(self.width, self.height))
        inp1 = cv.dnn.blobFromImage(second_img, size=(self.width, self.height))
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
        k0 = fk.astype(np.int)
        k1 = (k0 + 1) % ncols
        f = fk[..., np.newaxis] - k0[..., np.newaxis]

        col0 = colormap[k0] / 255.0
        col1 = colormap[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        col = np.where(rad <= 1, 1 - rad * (1 - col), col * 0.75)
        output = (255.0 * col).astype(np.uint8)
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use this script to calculate optical flow using FlowNetv2',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', '-i', required=True, help='Path to input video file. Skip this argument to capture frames from a camera.')
    parser.add_argument('--height', default=320, type=int, help='Input height')
    parser.add_argument('--width', default=448, type=int, help='Input width')
    parser.add_argument('--proto', '-p', default='FlowNet2_deploy_anysize.prototxt', help='Path to prototxt.')
    parser.add_argument('--model', '-m', default='FlowNet2_weights.caffemodel', help='Path to caffemodel.')
    args, _ = parser.parse_known_args()

    if not os.path.isfile(args.model) or not os.path.isfile(args.proto):
        raise OSError("Prototxt or caffemodel not exist")

    winName = 'Calculation optical flow in OpenCV'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)
    cap = cv.VideoCapture(args.input if args.input else 0)
    hasFrame, first_frame = cap.read()

    divisor = 64.
    var = {}
    var['ADAPTED_WIDTH'] = int(np.ceil(args.width/divisor) * divisor)
    var['ADAPTED_HEIGHT'] = int(np.ceil(args.height/divisor) * divisor)
    var['SCALE_WIDTH'] = args.width / float(var['ADAPTED_WIDTH'])
    var['SCALE_HEIGHT'] = args.height / float(var['ADAPTED_HEIGHT'])

    config = ''
    proto = open(args.proto).readlines()
    for line in proto:
        for key, value in var.items():
            tag = "$%s$" % key
            line = line.replace(tag, str(value))
        config += line

    caffemodel = open(args.model, 'rb').read()

    opt_flow = OpticalFlow(bytearray(config.encode()), caffemodel, var['ADAPTED_HEIGHT'], var['ADAPTED_WIDTH'])
    while cv.waitKey(1) < 0:
        hasFrame, second_frame = cap.read()
        if not hasFrame:
            break
        flow = opt_flow.compute_flow(first_frame, second_frame)
        first_frame = second_frame
        cv.imshow(winName, flow)
