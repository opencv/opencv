from __future__ import print_function
# Script to evaluate MobileNet-SSD object detection model trained in TensorFlow
# using both TensorFlow and OpenCV. Example:
#
# python mobilenet_ssd_accuracy.py \
#   --weights=frozen_inference_graph.pb \
#   --prototxt=ssd_mobilenet_v1_coco.pbtxt \
#   --images=val2017 \
#   --annotations=annotations/instances_val2017.json
#
# Tested on COCO 2017 object detection dataset, http://cocodataset.org/#download
import os
import cv2 as cv
import json
import argparse

parser = argparse.ArgumentParser(
    description='Evaluate MobileNet-SSD model using both TensorFlow and OpenCV. '
                'COCO evaluation framework is required: http://cocodataset.org')
parser.add_argument('--weights', required=True,
                    help='Path to frozen_inference_graph.pb of MobileNet-SSD model. '
                         'Download it from http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz')
parser.add_argument('--prototxt', help='Path to ssd_mobilenet_v1_coco.pbtxt from opencv_extra.', required=True)
parser.add_argument('--images', help='Path to COCO validation images directory.', required=True)
parser.add_argument('--annotations', help='Path to COCO annotations file.', required=True)
args = parser.parse_args()

### Get OpenCV predictions #####################################################
net = cv.dnn.readNetFromTensorflow(args.weights, args.prototxt)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV);

detections = []
for imgName in os.listdir(args.images):
    inp = cv.imread(os.path.join(args.images, imgName))
    rows = inp.shape[0]
    cols = inp.shape[1]
    inp = cv.resize(inp, (300, 300))

    net.setInput(cv.dnn.blobFromImage(inp, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), True))
    out = net.forward()

    for i in range(out.shape[2]):
        score = float(out[0, 0, i, 2])
        # Confidence threshold is in prototxt.
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

### Get TensorFlow predictions #################################################
import tensorflow as tf

with tf.gfile.FastGFile(args.weights) as f:
    # Load the model
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    detections = []
    for imgName in os.listdir(args.images):
        inp = cv.imread(os.path.join(args.images, imgName))
        rows = inp.shape[0]
        cols = inp.shape[1]
        inp = cv.resize(inp, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.01:
                x = bbox[1] * cols
                y = bbox[0] * rows
                w = bbox[3] * cols - x
                h = bbox[2] * rows - y
                detections.append({
                  "image_id": int(imgName.rstrip('0')[:imgName.rfind('.')]),
                  "category_id": classId,
                  "bbox": [x, y, w, h],
                  "score": score
                })

with open('tf_result.json', 'wt') as f:
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
for resFile in ['tf_result.json', 'cv_result.json']:
    print(resFile)
    cocoDt=cocoGt.loadRes(resFile)

    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
