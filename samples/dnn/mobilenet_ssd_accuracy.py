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
    inp = cv.resize(inp, (300, 300))

    net.setInput(cv.dnn.blobFromImage(inp, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), True))
    out = net.forward()

    for i in range(out.shape[2]):
        score = float(out[0, 0, i, 2])
        classId = int(out[0, 0, i, 1])

        x = out[0, 0, i, 3] * cols
        y = out[0, 0, i, 4] * rows
        w = out[0, 0, i, 5] * cols - x
        h = out[0, 0, i, 6] * rows - y
        detections.append({
          "image_id": int(imgName.rstrip('0')[:imgName.rfind('.')]),
          "category_id": classId,
          "bbox": [x, y, w, h],
          "score": score
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
