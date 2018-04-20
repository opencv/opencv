import cv2 as cv
import argparse
import numpy as np
import sys

backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_HALIDE, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE)
targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL)

parser = argparse.ArgumentParser(description='Use this script to run semantic segmentation deep learning networks using OpenCV.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument('--model', required=True,
                    help='Path to a binary file of model contains trained weights. '
                         'It could be a file with extensions .caffemodel (Caffe), '
                         '.pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet)')
parser.add_argument('--config',
                    help='Path to a text file of model contains network configuration. '
                         'It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet)')
parser.add_argument('--framework', choices=['caffe', 'tensorflow', 'torch', 'darknet'],
                    help='Optional name of an origin framework of the model. '
                         'Detect it automatically if it does not set.')
parser.add_argument('--classes', help='Optional path to a text file with names of classes.')
parser.add_argument('--colors', help='Optional path to a text file with colors for an every class. '
                                     'An every color is represented with three values from 0 to 255 in BGR channels order.')
parser.add_argument('--mean', nargs='+', type=float, default=[0, 0, 0],
                    help='Preprocess input image by subtracting mean values. '
                         'Mean values should be in BGR order.')
parser.add_argument('--scale', type=float, default=1.0,
                    help='Preprocess input image by multiplying on a scale factor.')
parser.add_argument('--width', type=int, required=True,
                    help='Preprocess input image by resizing to a specific width.')
parser.add_argument('--height', type=int, required=True,
                    help='Preprocess input image by resizing to a specific height.')
parser.add_argument('--rgb', action='store_true',
                    help='Indicate that model works with RGB input images instead BGR ones.')
parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                    help="Choose one of computation backends: "
                         "%d: default C++ backend, "
                         "%d: Halide language (http://halide-lang.org/), "
                         "%d: Intel's Deep Learning Inference Engine (https://software.seek.intel.com/deep-learning-deployment)" % backends)
parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                    help='Choose one of target computation devices: '
                         '%d: CPU target (by default), '
                         '%d: OpenCL' % targets)
args = parser.parse_args()

np.random.seed(324)

# Load names of classes
classes = None
if args.classes:
    with open(args.classes, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

# Load colors
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
            cv.putText(block, classes[i], (0, blockHeight/2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        cv.namedWindow('Legend', cv.WINDOW_NORMAL)
        cv.imshow('Legend', legend)
        classes = None

# Load a network
net = cv.dnn.readNet(args.model, args.config, args.framework)
net.setPreferableBackend(args.backend)
net.setPreferableTarget(args.target)

winName = 'Deep learning image classification in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

cap = cv.VideoCapture(args.input if args.input else 0)
legend = None
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, args.scale, (args.width, args.height), args.mean, args.rgb, crop=False)

    # Run a model
    net.setInput(blob)
    score = net.forward()

    numClasses = score.shape[1]
    height = score.shape[2]
    width = score.shape[3]

    # Draw segmentation
    if not colors:
        # Generate colors
        colors = [np.array([0, 0, 0], np.uint8)]
        for i in range(1, numClasses):
            colors.append((colors[i - 1] + np.random.randint(0, 256, [3], np.uint8)) / 2)

    classIds = np.argmax(score[0], axis=0)
    segm = np.stack([colors[idx] for idx in classIds.flatten()])
    segm = segm.reshape(height, width, 3)

    segm = cv.resize(segm, (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_NEAREST)
    frame = (0.1 * frame + 0.9 * segm).astype(np.uint8)

    # Put efficiency information.
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    showLegend(classes)

    cv.imshow(winName, frame)
