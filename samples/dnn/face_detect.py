import argparse

import cv2 as cv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, help='Path to the input image.')
parser.add_argument('--model', '-m', type=str, help='Path to the model. Download the model at https://github.com/ShiqiYu/libfacedetection.train/tree/master/tasks/task1/onnx.')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
args = parser.parse_args()


# Read the input image
img = cv.imread(args.input)

# Instantiate face detector
detector = cv.FaceDetector.create(
    args.model,
    "",
    (img.shape[1], img.shape[0]),
    args.score_threshold,
    args.nms_threshold,
    args.top_k
)

# Forward
results = detector.detect(img)
print(results[1].shape) # tuple (1, faces); faces: (num_faces, 15)

# Visualize results
for idx, face in enumerate(results[1]):
    print('{}: [{:.0f}, {:.0f}] [{:.0f}, {:.0f}] {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

    coords = face[:-1].astype(np.int32)
    cv.rectangle(img, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)
    cv.circle(img, (coords[4], coords[5]), 2, (255, 0, 0), 2)
    cv.circle(img, (coords[6], coords[7]), 2, (0, 0, 255), 2)
    cv.circle(img, (coords[8], coords[9]), 2, (0, 255, 0), 2)
    cv.circle(img, (coords[10], coords[11]), 2, (255, 0, 255), 2)
    cv.circle(img, (coords[12], coords[13]), 2, (0, 255, 255), 2)

cv.imwrite(args.input.split('/')[-1], img)