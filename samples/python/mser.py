#!/usr/bin/env python

'''
MSER detector demo
==================

Usage:
------
    mser.py [<video source>]

Keys:
-----
    ESC   - exit

'''

import numpy as np
import cv2 as cv
import video
import sys

if __name__ == '__main__':
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    cam = video.create_capture(video_src)
    mser = cv.MSER_create()

    while True:
        ret, img = cam.read()
        if ret == 0:
            break
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        vis = img.copy()

        regions, _ = mser.detectRegions(gray)
        hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        cv.polylines(vis, hulls, 1, (0, 255, 0))

        cv.imshow('img', vis)
        if cv.waitKey(5) == 27:
            break
    cv.destroyAllWindows()
