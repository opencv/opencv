#!/usr/bin/env python

'''
face detection using haar cascades

USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>]
                  [--dnn_config <config_path>] [--dnn_model <model_path>]
                  [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

# local modules
from common import clock, draw_str
from video import create_capture

import numpy as np
import cv2 as cv


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade=', 'dnn_config=', 'dnn_model='])  # FIXIT: nested-cascade -> nested_cascade
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye.xml")

    cascade = cv.CascadeClassifier(cascade_fn)
    nested = cv.CascadeClassifier(nested_fn)

    def detect(img, cascade):
        rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                         flags=cv.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            return []
        rects[:,2:] += rects[:,:2]  # x, y, w, h ==> x, y, x+w, y+h
        return rects

    dnn_config_fn = args.get('--dnn_config', "../data/dnn_face_detector/opencv_face_detector.pbtxt")
    dnn_model_fn  = args.get('--dnn_model', "../data/dnn_face_detector/opencv_face_detector_uint8.pb")

    net = cv.dnn.readNet(dnn_model_fn, dnn_config_fn)

    # see dnn/object_detection.py for details
    layerNames = net.getLayerNames()
    lastLayer = net.getLayer(net.getLayerId(layerNames[-1]))
    assert lastLayer.type == 'DetectionOutput', lastLayer.type

    def detect_faces_dnn(frame):
        confThreshold = 0.5
        nmsThreshold = 0.4

        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123), swapRB=False, crop=False)
        net.setInput(blob)
        outs = net.forward(['detection_out'])

        confidences = []
        boxes = []
        for out in outs:
            for detection in out[0, 0]:
                confidence = detection[2]
                if confidence > confThreshold:
                    left = int(detection[3] * frameWidth)
                    top = int(detection[4] * frameHeight)
                    right = int(detection[5] * frameWidth)
                    bottom = int(detection[6] * frameHeight)
                    width = right - left + 1
                    height = bottom - top + 1
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        result = []
        for i in indices:
            box = boxes[i[0]]
            result.append((box[0], box[1], box[0] + box[2], box[1] + box[3]))  # x1, y1, x2, y2
        return result


    cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')

    useDNN = True
    while True:
        ret, img = cam.read()

        t0 = clock()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        t1 = clock()
        if useDNN:
            rects = detect_faces_dnn(img)
        else:
            rects = detect(gray, cascade)
        dt1 = clock() - t1
        sub_rects = []
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                subrects = np.array(subrects)
                if subrects.size > 0:
                    subrects[:,:] += (x1, y1, x1, y1)  # roi -> global
                    sub_rects.append(subrects)
        dt2 = clock() - t0
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        for subrects in sub_rects:
            draw_rects(vis, subrects, (255, 0, 0))

        draw_str(vis, (20, 20), ('DNN face detector (resized 300x300)' if useDNN else 'Cascade face detector ({}x{})'.format(img.shape[1], img.shape[0])))
        draw_str(vis, (20, 40), 'Face detection time: %.1f ms (total %.1f ms)' % (dt1*1000, dt2*1000))
        draw_str(vis, (20, 60), "Press 'Space' to switch between DNN / Cascade classifier face detection")
        cv.imshow('facedetect', vis)

        c = cv.waitKey(1)
        if c == 27:  # Esc
            break
        if c == 32:  # Space
            useDNN = not useDNN

    cv.destroyAllWindows()
