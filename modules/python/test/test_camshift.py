#!/usr/bin/env python

'''
Camshift tracker
================

This is a demo that shows mean-shift based tracking
You select a color objects such as your face and it tracks it.
This reads from video camera (0 by default, or the camera number the user enters)

http://www.robinhewitt.com/research/track/camshift.html

'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2
from tst_scene_render import TestSceneRender

from tests_common import NewOpenCVTests, intersectionRate

class camshift_test(NewOpenCVTests):

    framesNum = 300
    frame = None
    selection = None
    drag_start = None
    show_backproj = False
    track_window = None
    render = None
    errors = 0

    def prepareRender(self):

        self.render = TestSceneRender(self.get_sample('samples/python2/data/pca_test1.jpg'), deformation = True)

    def runTracker(self):

        framesCounter = 0
        self.selection = True

        xmin, ymin, xmax, ymax = self.render.getCurrentRect()

        self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)

        while True:
            framesCounter += 1
            self.frame = self.render.getNextFrame()
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

            if self.selection:
                x0, y0, x1, y1 = self.render.getCurrentRect() + 50
                x0 -= 100
                y0 -= 100

                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]
                hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                self.hist = hist.reshape(-1)
                self.selection = False

            if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
                self.selection = None
                prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                prob &= mask
                term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
                track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)

            trackingRect = np.array(self.track_window)
            trackingRect[2] += trackingRect[0]
            trackingRect[3] += trackingRect[1]

            if intersectionRate(self.render.getCurrentRect(), trackingRect) < 0.4:
                self.errors += 1

            if framesCounter > self.framesNum:
                break

        self.assertLess(float(self.errors) / self.framesNum, 0.4)

    def test_camshift(self):
        self.prepareRender()
        self.runTracker()