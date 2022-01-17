#!/usr/bin/env python
'''
Tracker demo

For usage download models by following links
For GOTURN:
    goturn.prototxt and goturn.caffemodel: https://github.com/opencv/opencv_extra/tree/c4219d5eb3105ed8e634278fad312a1a8d2c182d/testdata/tracking
For DaSiamRPN:
    network:     https://www.dropbox.com/s/rr1lk9355vzolqv/dasiamrpn_model.onnx?dl=0
    kernel_r1:   https://www.dropbox.com/s/999cqx5zrfi7w4p/dasiamrpn_kernel_r1.onnx?dl=0
    kernel_cls1: https://www.dropbox.com/s/qvmtszx5h339a0w/dasiamrpn_kernel_cls1.onnx?dl=0

USAGE:
    tracker.py [-h] [--input INPUT] [--tracker_algo TRACKER_ALGO]
                    [--goturn GOTURN] [--goturn_model GOTURN_MODEL]
                    [--dasiamrpn_net DASIAMRPN_NET]
                    [--dasiamrpn_kernel_r1 DASIAMRPN_KERNEL_R1]
                    [--dasiamrpn_kernel_cls1 DASIAMRPN_KERNEL_CLS1]
                    [--dasiamrpn_backend DASIAMRPN_BACKEND]
                    [--dasiamrpn_target DASIAMRPN_TARGET]
'''

# Python 2/3 compatibility
from __future__ import print_function

import sys

import numpy as np
import cv2 as cv
import argparse

from video import create_capture, presets

class App(object):

    def __init__(self, args):
        self.args = args
        self.trackerAlgorithm = args.tracker_algo
        self.tracker = self.createTracker()

    def createTracker(self):
        if self.trackerAlgorithm == 'mil':
            tracker = cv.TrackerMIL_create()
        elif self.trackerAlgorithm == 'goturn':
            params = cv.TrackerGOTURN_Params()
            params.modelTxt = self.args.goturn
            params.modelBin = self.args.goturn_model
            tracker = cv.TrackerGOTURN_create(params)
        elif self.trackerAlgorithm == 'dasiamrpn':
            params = cv.TrackerDaSiamRPN_Params()
            params.model = self.args.dasiamrpn_net
            params.kernel_cls1 = self.args.dasiamrpn_kernel_cls1
            params.kernel_r1 = self.args.dasiamrpn_kernel_r1
            tracker = cv.TrackerDaSiamRPN_create(params)
        else:
            sys.exit("Tracker {} is not recognized. Please use one of three available: mil, goturn, dasiamrpn.".format(self.trackerAlgorithm))
        return tracker

    def initializeTracker(self, image):
        while True:
            print('==> Select object ROI for tracker ...')
            bbox = cv.selectROI('tracking', image)
            print('ROI: {}'.format(bbox))
            if bbox[2] <= 0 or bbox[3] <= 0:
                sys.exit("ROI selection cancelled. Exiting...")

            try:
                self.tracker.init(image, bbox)
            except Exception as e:
                print('Unable to initialize tracker with requested bounding box. Is there any object?')
                print(e)
                print('Try again ...')
                continue

            return

    def run(self):
        videoPath = self.args.input
        print('Using video: {}'.format(videoPath))
        camera = create_capture(cv.samples.findFileOrKeep(videoPath), presets['cube'])
        if not camera.isOpened():
            sys.exit("Can't open video stream: {}".format(videoPath))

        ok, image = camera.read()
        if not ok:
            sys.exit("Can't read first frame")
        assert image is not None

        cv.namedWindow('tracking')
        self.initializeTracker(image)

        print("==> Tracking is started. Press 'SPACE' to re-initialize tracker or 'ESC' for exit...")

        while camera.isOpened():
            ok, image = camera.read()
            if not ok:
                print("Can't read frame")
                break

            ok, newbox = self.tracker.update(image)
            #print(ok, newbox)

            if ok:
                cv.rectangle(image, newbox, (200,0,0))

            cv.imshow("tracking", image)
            k = cv.waitKey(1)
            if k == 32:  # SPACE
                self.initializeTracker(image)
            if k == 27:  # ESC
                break

        print('Done')


if __name__ == '__main__':
    print(__doc__)
    parser = argparse.ArgumentParser(description="Run tracker")
    parser.add_argument("--input", type=str, default="vtest.avi", help="Path to video source")
    parser.add_argument("--tracker_algo", type=str, default="mil", help="One of available tracking algorithms: mil, goturn, dasiamrpn")
    parser.add_argument("--goturn", type=str, default="goturn.prototxt", help="Path to GOTURN architecture")
    parser.add_argument("--goturn_model", type=str, default="goturn.caffemodel", help="Path to GOTERN model")
    parser.add_argument("--dasiamrpn_net", type=str, default="dasiamrpn_model.onnx", help="Path to onnx model of DaSiamRPN net")
    parser.add_argument("--dasiamrpn_kernel_r1", type=str, default="dasiamrpn_kernel_r1.onnx", help="Path to onnx model of DaSiamRPN kernel_r1")
    parser.add_argument("--dasiamrpn_kernel_cls1", type=str, default="dasiamrpn_kernel_cls1.onnx", help="Path to onnx model of DaSiamRPN kernel_cls1")

    args = parser.parse_args()
    App(args).run()
    cv.destroyAllWindows()
