#!/usr/bin/env python

'''
Tracker demo

USAGE:
    tracker.py [<video_source> <tracker_algorithm>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import sys

import numpy as np
import cv2 as cv

from video import create_capture, presets

class App(object):

    def initializeTracker(self, image, trackerAlgorithm):
        while True:
            if trackerAlgorithm == 'mil':
                tracker = cv.TrackerMIL_create()
            elif trackerAlgorithm == 'goturn':
                tracker = cv.TrackerGOTURN_create()
            elif trackerAlgorithm == 'dasiamrpn':
                tracker = cv.TrackerDaSiamRPN_create()
            else:
                sys.exit("Tracker {} is not recognized. Please use one of three available: mil, goturn, dasiamrpn.".format(trackerAlgorithm))

            print('==> Select object ROI for tracker ...')
            bbox = cv.selectROI('tracking', image)
            print('ROI: {}'.format(bbox))

            try:
                tracker.init(image, bbox)
            except Exception as e:
                print('Unable to initialize tracker with requested bounding box. Is there any object?')
                print(e)
                print('Try again ...')
                continue

            return tracker

    def run(self):
        videoPath = sys.argv[1] if len(sys.argv) >= 2 else 'vtest.avi'
        trackerAlgorithm = sys.argv[2] if len(sys.argv) > 2 else 'mil'
        camera = create_capture(videoPath, presets['cube'])
        if not camera.isOpened():
            sys.exit("Can't open video stream: {}".format(videoPath))

        ok, image = camera.read()
        if not ok:
            sys.exit("Can't read first frame")
        assert image is not None

        cv.namedWindow('tracking')
        tracker = self.initializeTracker(image, trackerAlgorithm)

        print("==> Tracking is started. Press 'SPACE' to re-initialize tracker or 'ESC' for exit...")

        while camera.isOpened():
            ok, image = camera.read()
            if not ok:
                print("Can't read frame")
                break

            ok, newbox = tracker.update(image)
            #print(ok, newbox)

            if ok:
                cv.rectangle(image, newbox, (200,0,0))

            cv.imshow("tracking", image)
            k = cv.waitKey(1)
            if k == 32:  # SPACE
                tracker = self.initializeTracker(image)
            if k == 27:  # ESC
                break

        print('Done')


if __name__ == '__main__':
    print(__doc__)
    App().run()
    cv.destroyAllWindows()
