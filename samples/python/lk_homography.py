#!/usr/bin/env python

'''
Lucas-Kanade homography tracker
===============================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames. Finds homography between reference and current views.

Usage
-----
lk_homography.py [<video_source>]


Keys
----
ESC   - exit
SPACE - start tracking
r     - toggle RANSAC
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

import video
from common import draw_str
from video import presets

lk_params = dict( winSize  = (19, 19),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.01,
                       minDistance = 8,
                       blockSize = 19 )

def checkedTrace(img0, img1, p0, back_threshold = 1.0):
    p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    status = d < back_threshold
    return p1, status

green = (0, 255, 0)
red = (0, 0, 255)

class App:
    def __init__(self, video_src):
        self.cam = self.cam = video.create_capture(video_src, presets['book'])
        self.p0 = None
        self.use_ransac = True

    def run(self):
        while True:
            _ret, frame = self.cam.read()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()
            if self.p0 is not None:
                p2, trace_status = checkedTrace(self.gray1, frame_gray, self.p1)

                self.p1 = p2[trace_status].copy()
                self.p0 = self.p0[trace_status].copy()
                self.gray1 = frame_gray

                if len(self.p0) < 4:
                    self.p0 = None
                    continue
                H, status = cv.findHomography(self.p0, self.p1, (0, cv.RANSAC)[self.use_ransac], 10.0)
                h, w = frame.shape[:2]
                overlay = cv.warpPerspective(self.frame0, H, (w, h))
                vis = cv.addWeighted(vis, 0.5, overlay, 0.5, 0.0)

                for (x0, y0), (x1, y1), good in zip(self.p0[:,0], self.p1[:,0], status[:,0]):
                    if good:
                        cv.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 128, 0))
                    cv.circle(vis, (int(x1), int(y1)), 2, (red, green)[good], -1)
                draw_str(vis, (20, 20), 'track count: %d' % len(self.p1))
                if self.use_ransac:
                    draw_str(vis, (20, 40), 'RANSAC')
            else:
                p = cv.goodFeaturesToTrack(frame_gray, **feature_params)
                if p is not None:
                    for x, y in p[:,0]:
                        cv.circle(vis, (int(x), int(y)), 2, green, -1)
                    draw_str(vis, (20, 20), 'feature count: %d' % len(p))

            cv.imshow('lk_homography', vis)

            ch = cv.waitKey(1)
            if ch == 27:
                break
            if ch == ord(' '):
                self.frame0 = frame.copy()
                self.p0 = cv.goodFeaturesToTrack(frame_gray, **feature_params)
                if self.p0 is not None:
                    self.p1 = self.p0
                    self.gray0 = frame_gray
                    self.gray1 = frame_gray
            if ch == ord('r'):
                self.use_ransac = not self.use_ransac



def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    App(video_src).run()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
