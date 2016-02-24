#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

#local modules
from tst_scene_render import TestSceneRender
from tests_common import NewOpenCVTests, intersectionRate, isPointInRect

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

def getRectFromPoints(points):

    distances = []
    for point in points:
        distances.append(cv2.norm(point, cv2.NORM_L2))

    x0, y0 = points[np.argmin(distances)]
    x1, y1 = points[np.argmax(distances)]

    return np.array([x0, y0, x1, y1])


class lk_track_test(NewOpenCVTests):

    track_len = 10
    detect_interval = 5
    tracks = []
    frame_idx = 0
    render = None

    def test_lk_track(self):

        self.render = TestSceneRender(self.get_sample('samples/data/graf1.png'), self.get_sample('samples/data/box.png'))
        self.runTracker()

    def runTracker(self):
        foregroundPointsNum = 0

        while True:
            frame = self.render.getNextFrame()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1][0] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append([(x, y), self.frame_idx])
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                self.tracks = new_tracks

            if self.frame_idx % self.detect_interval == 0:
                goodTracksCount = 0
                for tr in self.tracks:
                    oldRect = self.render.getRectInTime(self.render.timeStep * tr[0][1])
                    newRect = self.render.getRectInTime(self.render.timeStep * tr[-1][1])
                    if isPointInRect(tr[0][0], oldRect) and isPointInRect(tr[-1][0], newRect):
                        goodTracksCount += 1

                if self.frame_idx == self.detect_interval:
                    foregroundPointsNum = goodTracksCount

                fgIndex = float(foregroundPointsNum) / (foregroundPointsNum + 1)
                fgRate = float(goodTracksCount) / (len(self.tracks) + 1)

                if self.frame_idx > 0:
                    self.assertGreater(fgIndex, 0.9)
                    self.assertGreater(fgRate, 0.2)

                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1][0]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([[(x, y), self.frame_idx]])

            self.frame_idx += 1
            self.prev_gray = frame_gray

            if self.frame_idx > 300:
                break