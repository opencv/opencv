from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

parser = argparse.ArgumentParser(
        description='This script is used to run style transfer models from '
                    'https://github.com/onnx/models/tree/main/vision/style_transfer/fast_neural_style using OpenCV')
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--model', help='Path to .onnx model')
parser.add_argument('--width', default=-1, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=-1, type=int, help='Resize input to specific height.')
parser.add_argument('--median_filter', default=0, type=int, help='Kernel size of postprocessing blurring.')
args = parser.parse_args()

net = cv.dnn.readNet(cv.samples.findFile(args.model))
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

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
    out = net.forward()

    out = out.reshape(3, out.shape[2], out.shape[3])
    out = out.transpose(1, 2, 0)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    print(t / freq, 'ms')

    if args.median_filter:
        out = cv.medianBlur(out, args.median_filter)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    cv.imshow('Styled image', out)
