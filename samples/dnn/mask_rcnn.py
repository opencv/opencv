'''
Mask R-CNN
This is an example of using Mask R-CNN for object detection and instance segmentation.

The model is the ONNX Model Zoo build of Mask R-CNN, converted from
facebookresearch/maskrcnn-benchmark. Download it with:

    python download_models.py mask_rcnn

It takes a BGR image of shape (3, height, width) with no batch dimension, resized so
the short side is 800 and zero-padded to a multiple of 32, and returns four tensors:
boxes (nbox, 4), labels (nbox), scores (nbox) and masks (nbox, 1, 28, 28).
'''
import cv2 as cv
import argparse
import numpy as np
import math
import sys
import os

from common import *

def help():
    print(
        '''
        Firstly, download the model using `download_models.py mask_rcnn`. Set the environment
        variable OPENCV_DOWNLOAD_CACHE_DIR to specify where models should be downloaded. Also,
        point OPENCV_SAMPLES_DATA_PATH to opencv/samples/data.

        To run:
            python mask_rcnn.py --input=path/to/your/input/image/or/video (don't pass --input to use device camera)
        '''
    )

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                    help='An optional path to file with preprocessing parameters.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument('--thr', type=float, default=0.5, help='Confidence threshold')
parser.add_argument('--colors', help='Optional path to a text file with colors for an every class. '
                                     'An every color is represented with three values from 0 to 255 in BGR channels order.')
args, _ = parser.parse_known_args()
add_preproc_args(args.zoo, parser, 'mask_rcnn')
parser = argparse.ArgumentParser(parents=[parser],
                                 description='Use this script to run Mask-RCNN object detection '
                                             'and instance segmentation.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
args = parser.parse_args()

if hasattr(args, 'help'):
    help()
    exit(1)

args.model = findModel(args.model, args.sha1)
if args.labels is not None:
    args.labels = findFile(args.labels)

np.random.seed(324)

# Load names of classes. Label 0 is background, so the model's labels are 1-based.
classes = None
if args.labels:
    with open(args.labels, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

colors = None
if args.colors:
    with open(args.colors, 'rt') as f:
        colors = [np.array(color.split(' '), np.uint8) for color in f.read().rstrip('\n').split('\n')]

legend = None
def showLegend(classes):
    global legend
    if not classes is None and legend is None:
        blockHeight = 30
        assert(len(classes) == len(colors))

        legend = np.zeros((blockHeight * len(colors), 200, 3), np.uint8)
        for i in range(len(classes)):
            block = legend[i * blockHeight:(i + 1) * blockHeight]
            block[:,:] = colors[i]
            cv.putText(block, classes[i], (0, blockHeight//2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        cv.namedWindow('Legend', cv.WINDOW_NORMAL)
        cv.imshow('Legend', legend)
        classes = None


def drawBox(frame, classId, conf, left, top, right, bottom):
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

    label = '%.2f' % conf
    if classes and 0 <= classId < len(classes):
        label = '%s: %s' % (classes[classId], label)

    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


def preprocess(frame, mean):
    ratio = 800.0 / min(frame.shape[0], frame.shape[1])
    resized = cv.resize(frame, None, fx=ratio, fy=ratio, interpolation=cv.INTER_LINEAR)

    blob = resized.astype(np.float32) - np.array(mean, np.float32)
    blob = blob.transpose(2, 0, 1)

    paddedH = int(math.ceil(blob.shape[1] / 32) * 32)
    paddedW = int(math.ceil(blob.shape[2] / 32) * 32)
    padded = np.zeros((3, paddedH, paddedW), np.float32)
    padded[:, :blob.shape[1], :blob.shape[2]] = blob
    return padded, ratio


net = cv.dnn.readNetFromONNX(args.model)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
outNames = net.getUnconnectedOutLayersNames()

winName = 'Mask-RCNN in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

cap = cv.VideoCapture(cv.samples.findFileOrKeep(args.input) if args.input else 0)
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameH, frameW = frame.shape[0], frame.shape[1]

    blob, ratio = preprocess(frame, args.mean)
    net.setInput(blob)

    t0 = cv.getTickCount()
    boxes, labels, scores, masks = net.forward(outNames)
    t = (cv.getTickCount() - t0) / cv.getTickFrequency()

    boxes = boxes.reshape(-1, 4) / ratio
    labels = labels.reshape(-1)
    scores = scores.reshape(-1)

    if not colors:
        numClasses = len(classes) if classes else 80
        colors = [np.array([0, 0, 0], np.uint8)]
        for i in range(1, numClasses + 1):
            colors.append((colors[i - 1] + np.random.randint(0, 256, [3], np.uint8)) / 2)
        del colors[0]

    boxesToDraw = []
    for i in range(len(scores)):
        if scores[i] <= args.thr:
            continue

        # Model labels are 1-based; index 0 is background.
        classId = int(labels[i]) - 1
        left, top, right, bottom = [int(v) for v in boxes[i]]

        left = max(0, min(left, frameW - 1))
        top = max(0, min(top, frameH - 1))
        right = max(0, min(right, frameW - 1))
        bottom = max(0, min(bottom, frameH - 1))
        if right <= left or bottom <= top:
            continue

        boxesToDraw.append([frame, classId, scores[i], left, top, right, bottom])

        classMask = cv.resize(masks[i][0], (right - left + 1, bottom - top + 1))
        mask = (classMask > 0.5)

        roi = frame[top:bottom+1, left:right+1][mask]
        frame[top:bottom+1, left:right+1][mask] = (0.7 * colors[classId % len(colors)] + 0.3 * roi).astype(np.uint8)

    for box in boxesToDraw:
        drawBox(*box)

    label = 'Inference time: %.2f ms' % (t * 1000.0)
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    showLegend(classes)

    cv.imshow(winName, frame)
