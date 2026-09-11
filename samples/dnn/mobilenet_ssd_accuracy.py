from __future__ import print_function
# Script to evaluate MobileNet-SSD object detection model with OpenCV. Example:
#
# python mobilenet_ssd_accuracy.py \
#   --model=ssd_mobilenet_v1_coco_2017_11_17_2026jul.onnx \
#   --images=val2017 \
#   --annotations=annotations/instances_val2017.json
#
# Tested on COCO 2017 object detection dataset, http://cocodataset.org/#download
import os
import cv2 as cv
import json
import argparse

parser = argparse.ArgumentParser(
    description='Evaluate MobileNet-SSD model using OpenCV. '
                'COCO evaluation framework is required: http://cocodataset.org')
parser.add_argument('--model', required=True,
                    help='Path to ssd_mobilenet_v1_coco .onnx model. '
                         'Download it from https://huggingface.co/opencv/opencv_contribution/resolve/main/'
                         'ssd_mobilenet_v1_coco_2017_11_17/ssd_mobilenet_v1_coco_2017_11_17_2026jul.onnx?download=true')
parser.add_argument('--images', help='Path to COCO validation images directory.', required=True)
parser.add_argument('--annotations', help='Path to COCO annotations file.', required=True)
args = parser.parse_args()

### Get OpenCV predictions #####################################################
net = cv.dnn.readNetFromONNX(cv.samples.findFile(args.model))
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

detections = []
for imgName in os.listdir(args.images):
    inp = cv.imread(cv.samples.findFile(os.path.join(args.images, imgName)))
    rows = inp.shape[0]
    cols = inp.shape[1]
    resized = cv.resize(inp, (300, 300))

    # image_tensor input is NHWC uint8 RGB
    blob = cv.cvtColor(resized, cv.COLOR_BGR2RGB).reshape(1, 300, 300, 3)
    net.setInput(blob)
    boxes, scores, classes, num = net.forward(
        ['detection_boxes:0', 'detection_scores:0', 'detection_classes:0', 'num_detections:0'])

    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    classes = classes.reshape(-1)
    for i in range(int(num.reshape(-1)[0])):
        y = boxes[i][0] * rows
        x = boxes[i][1] * cols
        h = boxes[i][2] * rows - y
        w = boxes[i][3] * cols - x
        detections.append({
          "image_id": int(imgName.rstrip('0')[:imgName.rfind('.')]),
          "category_id": int(classes[i]),
          "bbox": [x, y, w, h],
          "score": float(scores[i])
        })

with open('cv_result.json', 'wt') as f:
    json.dump(detections, f)

### Evaluation part ############################################################

# %matplotlib inline
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print('Running demo for *%s* results.'%(annType))

#initialize COCO ground truth api
cocoGt=COCO(args.annotations)

#initialize COCO detections api
resFile = 'cv_result.json'
print(resFile)
cocoDt=cocoGt.loadRes(resFile)

cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
