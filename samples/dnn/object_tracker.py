#!/usr/bin/env python
'''
Tracker demo

For usage download models by following links
For DaSiamRPN:
    network:     https://www.dropbox.com/s/rr1lk9355vzolqv/dasiamrpn_model.onnx?dl=0
    kernel_r1:   https://www.dropbox.com/s/999cqx5zrfi7w4p/dasiamrpn_kernel_r1.onnx?dl=0
    kernel_cls1: https://www.dropbox.com/s/qvmtszx5h339a0w/dasiamrpn_kernel_cls1.onnx?dl=0
For NanoTrack:
    nanotrack_backbone: https://github.com/HonglinChu/SiamTrackers/blob/master/NanoTrack/models/nanotrackv2/nanotrack_backbone_sim.onnx
    nanotrack_headneck: https://github.com/HonglinChu/SiamTrackers/blob/master/NanoTrack/models/nanotrackv2/nanotrack_head_sim.onnx
For VitTrack:
    vitTracker: https://github.com/opencv/opencv_zoo/raw/fef72f8fa7c52eaf116d3df358d24e6e959ada0e/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx
USAGE:
    tracker.py [alias] [-h] [--input INPUT_VIDEO]
                    [--tracker_algo TRACKER_ALGO mil, dasiamrpn, nanotrack, vittrack]
                    [--dasiamrpn_net DASIAMRPN_NET]
                    [--dasiamrpn_kernel_r1 DASIAMRPN_KERNEL_R1]
                    [--dasiamrpn_kernel_cls1 DASIAMRPN_KERNEL_CLS1]
                    [--dasiamrpn_backend DASIAMRPN_BACKEND]
                    [--dasiamrpn_target DASIAMRPN_TARGET]
                    [--nanotrack_backbone NANOTRACK_BACKBONE]
                    [--nanotrack_headneck NANOTRACK_TARGET]
                    [--vittrack_net VITTRACK_MODEL]
'''

import cv2 as cv
import argparse
from common import *

def createTracker():
    if args.alias == 'dasiamrpn':
        params = cv.TrackerDaSiamRPN_Params()
        params.model = findModel(args.dasiamrpn_net, "")
        params.kernel_cls1 = findModel(args.kernel_cls1, "")
        params.kernel_r1 = findModel(args.kernel_r1, "")
        tracker = cv.TrackerDaSiamRPN_create(params)
    elif args.alias == 'nano':
        params = cv.TrackerNano_Params()
        params.backbone = findModel(args.backbone, "")
        params.neckhead = findModel(args.headneck, "")
        tracker = cv.TrackerNano_create(params)
    elif args.alias == 'vit':
        params = cv.TrackerVit_Params()
        params.net = findModel(args.vit_net, "")
        tracker = cv.TrackerVit_create(params)
    else:
        print("Tracker {} is not recognized. Please use one of three available: mil, dasiamrpn, nanotrack.".format(args.alias))
        exit(-1)
    return tracker

def initializeTracker(image, tracker):
    while True:
        print('==> Select object ROI for tracker ...')
        bbox = cv.selectROI('tracking', image)
        print('ROI: {}'.format(bbox))
        if bbox[2] <= 0 or bbox[3] <= 0:
            print("ROI selection cancelled. Exiting...")
            exit(-1)

        try:
            tracker.init(image, bbox)
        except Exception as e:
            print('Unable to initialize tracker with requested bounding box. Is there any object?')
            print(e)
            print('Try again ...')
            continue

        return

def run():
    tracker = createTracker()
    videoPath = args.input
    print('Using video: {}'.format(videoPath))
    camera = cv.VideoCapture(cv.samples.findFileOrKeep(videoPath))
    if not camera.isOpened():
        print("Can't open video stream: {}".format(videoPath))
        exit(-1)

    ok, image = camera.read()
    if not ok:
        print("Can't read first frame")
        exit(-1)
    assert image is not None

    cv.namedWindow('tracking')
    initializeTracker(image, tracker)

    print("==> Tracking is started. Press 'SPACE' to re-initialize tracker or 'ESC' for exit...")

    while camera.isOpened():
        ok, image = camera.read()
        if not ok:
            print("Can't read frame")
            break

        ok, newbox = tracker.update(image)
        if ok:
            cv.rectangle(image, newbox, (200, 0, 0), thickness=2)

        cv.imshow("tracking", image)
        k = cv.waitKey(100)
        if k == 32:  # SPACE
            initializeTracker(image, tracker)
        if k == 27:  # ESC
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    parser = argparse.ArgumentParser(description="Run tracker")
    parser.add_argument("alias", type=str, nargs='?', help="Path to video source")
    parser.add_argument("--input", type=str, default="vtest.avi", help="Path to video source")
    parser.add_argument("--dasiamrpn_net", type=str, default="dasiamrpn_model.onnx", help="Path to onnx model of DaSiamRPN net")
    parser.add_argument("--kernel_r1", type=str, default="dasiamrpn_kernel_r1.onnx", help="Path to onnx model of DaSiamRPN kernel_r1")
    parser.add_argument("--kernel_cls1", type=str, default="dasiamrpn_kernel_cls1.onnx", help="Path to onnx model of DaSiamRPN kernel_cls1")
    parser.add_argument("--backbone", type=str, default="nanotrack_backbone_sim.onnx", help="Path to onnx model of NanoTrack backBone")
    parser.add_argument("--headneck", type=str, default="nanotrack_head_sim.onnx", help="Path to onnx model of NanoTrack headNeck")
    parser.add_argument("--vit_net", type=str, default="vitTracker.onnx", help="Path to onnx model of  vittrack")

    args = parser.parse_args()
    run()
    cv.destroyAllWindows()
