#!/usr/bin/env python

'''
Camshift tracker
================
This is a demo that shows mean-shift based tracking
You select a color objects such as your face and it tracks it.
This reads from video camera ( 0 by default, or the camera
    number the user enters)
http://www.robinhewitt.com/research/track/camshift.html
Usage:
------
    camshift.py [<video source>]
    To initialize tracking, select the object with mouse
Keys:
-----
    ESC   - exit
    b     - toggle back-projected probability visualization
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2

# local module
import video


class App(object):

    def __init__(self, video_src):
        self.cam = video.create_capture(video_src)
        ret, self.frame = self.cam.read()
        cv2.namedWindow('camshift')
        cv2.setMouseCallback('camshift', self.onmouse)

        self.selection = None
        self.drag_start = None
        self.show_backproj = False
        self.track_window = None
        self.expand_ratio = 0.2

    def onmouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.track_window = None
        if event == cv2.EVENT_LBUTTONUP:
            self.drag_start = None
            self.track_window = self.selection
        if self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)

    def show_hist(self, hist):
        bin_count = hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count * bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(hist[i])
            cv2.rectangle(img, (i * bin_w + 2, 255), ((i + 1) * bin_w -
                                                      2, 255 - h), (int(180.0 * i / bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('hist', img)

    def expand_window(self, last_track):
        x, y, w, h = last_track
        row, col = self.frame.shape[:2]
        n_x0 = np.maximum(0, x - int(w * self.expand_ratio) - 1)
        n_y0 = np.maximum(0, y - int(h * self.expand_ratio) - 1)
        n_w = np.minimum(col, w + int(w * self.expand_ratio * 2) + 1)
        n_h = np.minimum(row, h + int(h * self.expand_ratio * 2) + 1)
        return (n_x0, n_y0, n_w, n_h)

    def run(self):
        last_track = (0, 0, self.frame.shape[0], self.frame.shape[1])
        while True:
            ret, self.frame = self.cam.read()
            vis = self.frame.copy()
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(
                hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

            if self.selection:
                x0, y0, w, h = self.selection
                hsv_roi = hsv[y0:y0 + h, x0:x0 + w]
                mask_roi = mask[y0:y0 + h, x0:x0 + w]
                hist = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                hist = hist.reshape(-1)
                self.show_hist(hist)

                vis_roi = vis[y0:y0 + h, x0:x0 + w]
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

            if self.track_window:
                # lost the target, expand last valid track window
                if self.track_window[2] <= 0 or self.track_window[3] <= 0:
                    self.track_window = self.expand_window(last_track)
                last_track = self.track_window
                self.selection = None
                prob = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
                prob &= mask
                term_crit = (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                track_box, self.track_window = cv2.CamShift(
                    prob, self.track_window, term_crit)

                if self.show_backproj:
                    vis[:] = prob[..., np.newaxis]
                try:
                    cv2.ellipse(vis, track_box, (0, 0, 255), 2)
                except:
                    print(track_box)

            cv2.imshow('camshift', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    print(__doc__)
    App(video_src).run()
