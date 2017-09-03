#!/usr/bin/env python

'''
Lucas-Kanade homography tracker test
===============================
Uses goodFeaturesToTrack for track initialization and back-tracking for match verification
between frames. Finds homography between reference and current views.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

#local modules
from tst_scene_render import TestSceneRender
from tests_common import NewOpenCVTests, isPointInRect

lk_params = dict( winSize  = (19, 19),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.01,
                       minDistance = 8,
                       blockSize = 19 )

def checkedTrace(img0, img1, p0, back_threshold = 1.0):
    p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    status = d < back_threshold
    return p1, status

class lk_homography_test(NewOpenCVTests):

    render = None
    framesCounter = 0
    frame = frame0 = None
    p0 = None
    p1 = None
    gray0 = gray1 = None
    numFeaturesInRectOnStart = 0

    def test_lk_homography(self):
        self.render = TestSceneRender(self.get_sample('samples/data/graf1.png'),
            self.get_sample('samples/data/box.png'), noise = 0.1, speed = 1.0)

        frame = self.render.getNextFrame()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame0 = frame.copy()
        self.p0 = cv2.goodFeaturesToTrack(frame_gray, **feature_params)

        isForegroundHomographyFound = False

        if self.p0 is not None:
            self.p1 = self.p0
            self.gray0 = frame_gray
            self.gray1 = frame_gray
            currRect = self.render.getCurrentRect()
            for (x,y) in self.p0[:,0]:
                if isPointInRect((x,y), currRect):
                    self.numFeaturesInRectOnStart += 1

        while self.framesCounter < 200:
            self.framesCounter += 1
            frame = self.render.getNextFrame()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.p0 is not None:
                p2, trace_status = checkedTrace(self.gray1, frame_gray, self.p1)

                self.p1 = p2[trace_status].copy()
                self.p0 = self.p0[trace_status].copy()
                self.gray1 = frame_gray

                if len(self.p0) < 4:
                    self.p0 = None
                    continue
                _H, status = cv2.findHomography(self.p0, self.p1, cv2.RANSAC, 5.0)

                goodPointsInRect = 0
                goodPointsOutsideRect = 0
                for (_x0, _y0), (x1, y1), good in zip(self.p0[:,0], self.p1[:,0], status[:,0]):
                    if good:
                        if isPointInRect((x1,y1), self.render.getCurrentRect()):
                            goodPointsInRect += 1
                        else: goodPointsOutsideRect += 1

                if goodPointsOutsideRect < goodPointsInRect:
                    isForegroundHomographyFound = True
                    self.assertGreater(float(goodPointsInRect) / (self.numFeaturesInRectOnStart + 1), 0.6)
            else:
                self.p0 = cv2.goodFeaturesToTrack(frame_gray, **feature_params)

        self.assertEqual(isForegroundHomographyFound, True)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
