import cv2 as cv
import argparse
import numpy as np
import sys

from common import *

backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv.dnn.DNN_BACKEND_OPENCV,
            cv.dnn.DNN_BACKEND_VKCOM, cv.dnn.DNN_BACKEND_CUDA)
targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD, cv.dnn.DNN_TARGET_HDDL,
           cv.dnn.DNN_TARGET_VULKAN, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                    help='An optional path to file with preprocessing parameters.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument('--colors', help='Optional path to a text file with colors for an every class. '
                                     'An every color is represented with three values from 0 to 255 in BGR channels order.')
parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                    help="Choose one of computation backends: "
                         "%d: automatically (by default), "
                         "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                         "%d: OpenCV implementation, "
                         "%d: VKCOM, "
                         "%d: CUDA"% backends)
parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                    help='Choose one of target computation devices: '
                         '%d: CPU target (by default), '
                         '%d: OpenCL, '
                         '%d: OpenCL fp16 (half-float precision), '
                         '%d: NCS2 VPU, '
                         '%d: HDDL VPU, '
                         '%d: Vulkan, '
                         '%d: CUDA, '
                         '%d: CUDA fp16 (half-float preprocess)'% targets)
args, _ = parser.parse_known_args()
add_preproc_args(args.zoo, parser, 'segmentation')
parser = argparse.ArgumentParser(parents=[parser],
                                 description='Use this script to run semantic segmentation deep learning networks using OpenCV.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
args = parser.parse_args()

args.model = findFile(args.model)
args.classes = findFile(args.classes)

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
            cv.putText(block, classes[i], (0, blockHeight//2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        cv.namedWindow('Legend', cv.WINDOW_NORMAL)
        cv.imshow('Legend', legend)
        classes = None

# Load a network
net = cv.dnn.readNetFromONNX(args.model)
net.setPreferableBackend(args.backend)
net.setPreferableTarget(args.target)

winName = 'Deep learning semantic segmentation in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

cap = cv.VideoCapture(cv.samples.findFile(args.input) if args.input else 0)
legend = None
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    cv.imshow("Original Image", frame)
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    # Create a 4D blob from a frame.
    inpWidth = args.width if args.width else frameWidth
    inpHeight = args.height if args.height else frameHeight

    blob = cv.dnn.blobFromImage(frame, args.scale, (inpWidth, inpHeight), args.mean, args.rgb, crop=False)
    net.setInput(blob)

    if args.alias == 'u2netp':
        output = net.forward(net.getUnconnectedOutLayersNames())
        pred = output[0][0, 0, :, :]
        frame = (pred * 255).astype(np.uint8)
    else:
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

        segm = cv.resize(segm, (frameWidth, frameHeight), interpolation=cv.INTER_NEAREST)
        frame = (0.1 * frame + 0.9 * segm).astype(np.uint8)

        showLegend(classes)

    # Put efficiency information.
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    cv.imshow(winName, frame)