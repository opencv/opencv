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

def intersectionRate(s1, s2):

    x1, y1, x2, y2 = s1
    s1 = [[x1, y1], [x2,y1], [x2, y2], [x1, y2] ]

    x1, y1, x2, y2 = s2
    s2 = [[x1, y1], [x2,y1], [x2, y2], [x1, y2] ]

    area, intersection = cv2.intersectConvexConvex(np.array(s1), np.array(s2))
    return 2 * area / (cv2.contourArea(np.array(s1)) + cv2.contourArea(np.array(s2)))


from tests_common import NewOpenCVTests

class camshift_test(NewOpenCVTests):

    frame = None
    selection = None
    drag_start = None
    show_backproj = False
    track_window = None
    render = None

    def prepareRender(self):

        self.render = TestSceneRender(self.get_sample('samples/data/pca_test1.jpg'))

    def runTracker(self):

        framesCounter = 0
        self.selection = True

        xmin, ymin, xmax, ymax = self.render.getCurrentRect()

        self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)

        while True:
            framesCounter += 1
            self.frame = self.render.getNextFrame()
            vis = self.frame.copy()
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

                vis_roi = vis[y0:y1, x0:x1]
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

                self.selection = False

            if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
                self.selection = None
                prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                prob &= mask
                term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
                track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)

                if self.show_backproj:
                    vis[:] = prob[...,np.newaxis]

            trackingRect = np.array(self.track_window)
            trackingRect[2] += trackingRect[0]
            trackingRect[3] += trackingRect[1]

            self.assertGreater(intersectionRate((self.render.getCurrentRect()), trackingRect), 0.5)

            if framesCounter > 300:
                break

    def test_camshift(self):
        self.prepareRender()
        self.runTracker()