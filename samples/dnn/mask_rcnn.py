import cv2 as cv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description=
        'Use this script to run Mask-RCNN object detection and semantic '
        'segmentation network from TensorFlow Object Detection API.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument('--model', required=True, help='Path to a .pb file with weights.')
parser.add_argument('--config', required=True, help='Path to a .pxtxt file contains network configuration.')
parser.add_argument('--classes', help='Optional path to a text file with names of classes.')
parser.add_argument('--colors', help='Optional path to a text file with colors for an every class. '
                                     'An every color is represented with three values from 0 to 255 in BGR channels order.')
parser.add_argument('--width', type=int, default=800,
                    help='Preprocess input image by resizing to a specific width.')
parser.add_argument('--height', type=int, default=800,
                    help='Preprocess input image by resizing to a specific height.')
parser.add_argument('--thr', type=float, default=0.5, help='Confidence threshold')
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


def drawBox(frame, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

    label = '%.2f' % conf

    # Print a label of class.
    if classes:
        assert(classId < len(classes))
        label = '%s: %s' % (classes[classId], label)

    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


# Load a network
net = cv.dnn.readNet(args.model, args.config)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

winName = 'Mask-RCNN in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

cap = cv.VideoCapture(args.input if args.input else 0)
legend = None
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameH = frame.shape[0]
    frameW = frame.shape[1]

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, size=(args.width, args.height), swapRB=True, crop=False)

    # Run a model
    net.setInput(blob)

    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

    numClasses = masks.shape[1]
    numDetections = boxes.shape[2]

    # Draw segmentation
    if not colors:
        # Generate colors
        colors = [np.array([0, 0, 0], np.uint8)]
        for i in range(1, numClasses + 1):
            colors.append((colors[i - 1] + np.random.randint(0, 256, [3], np.uint8)) / 2)
        del colors[0]

    boxesToDraw = []
    for i in range(numDetections):
        box = boxes[0, 0, i]
        mask = masks[i]
        score = box[2]
        if score > args.thr:
            classId = int(box[1])
            left = int(frameW * box[3])
            top = int(frameH * box[4])
            right = int(frameW * box[5])
            bottom = int(frameH * box[6])

            left = max(0, min(left, frameW - 1))
            top = max(0, min(top, frameH - 1))
            right = max(0, min(right, frameW - 1))
            bottom = max(0, min(bottom, frameH - 1))

            boxesToDraw.append([frame, classId, score, left, top, right, bottom])

            classMask = mask[classId]
            classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
            mask = (classMask > 0.5)

            roi = frame[top:bottom+1, left:right+1][mask]
            frame[top:bottom+1, left:right+1][mask] = (0.7 * colors[classId] + 0.3 * roi).astype(np.uint8)

    for box in boxesToDraw:
        drawBox(*box)

    # Put efficiency information.
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    showLegend(classes)

    cv.imshow(winName, frame)
