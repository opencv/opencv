#!/usr/bin/env python

'''
Robust line fitting.
==================

Example of using cv.fitLine function for fitting line
to points in presence of outliers.

Usage
-----
fitline.py

Switch through different M-estimator functions and see,
how well the robust functions fit the line even
in case of ~50% of outliers.

Keys
----
SPACE - generate random points
f     - change distance function
ESC   - exit
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

import numpy as np
import cv2 as cv

# built-in modules
import itertools as it

# local modules
from common import draw_str


w, h = 512, 256

def toint(p):
    return tuple(map(int, p))

def sample_line(p1, p2, n, noise=0.0):
    p1 = np.float32(p1)
    t = np.random.rand(n,1)
    return p1 + (p2-p1)*t + np.random.normal(size=(n, 2))*noise

dist_func_names = it.cycle('DIST_L2 DIST_L1 DIST_L12 DIST_FAIR DIST_WELSCH DIST_HUBER'.split())

if PY3:
    cur_func_name = next(dist_func_names)
else:
    cur_func_name = dist_func_names.next()

def update(_=None):
    noise = cv.getTrackbarPos('noise', 'fit line')
    n = cv.getTrackbarPos('point n', 'fit line')
    r = cv.getTrackbarPos('outlier %', 'fit line') / 100.0
    outn = int(n*r)

    p0, p1 = (90, 80), (w-90, h-80)
    img = np.zeros((h, w, 3), np.uint8)
    cv.line(img, toint(p0), toint(p1), (0, 255, 0))

    if n > 0:
        line_points = sample_line(p0, p1, n-outn, noise)
        outliers = np.random.rand(outn, 2) * (w, h)
        points = np.vstack([line_points, outliers])
        for p in line_points:
            cv.circle(img, toint(p), 2, (255, 255, 255), -1)
        for p in outliers:
            cv.circle(img, toint(p), 2, (64, 64, 255), -1)
        func = getattr(cv, cur_func_name)
        vx, vy, cx, cy = cv.fitLine(np.float32(points), func, 0, 0.01, 0.01)
        cv.line(img, (int(cx-vx*w), int(cy-vy*w)), (int(cx+vx*w), int(cy+vy*w)), (0, 0, 255))

    draw_str(img, (20, 20), cur_func_name)
    cv.imshow('fit line', img)

def main():
    cv.namedWindow('fit line')
    cv.createTrackbar('noise', 'fit line', 3, 50, update)
    cv.createTrackbar('point n', 'fit line', 100, 500, update)
    cv.createTrackbar('outlier %', 'fit line', 30, 100, update)
    while True:
        update()
        ch = cv.waitKey(0)
        if ch == ord('f'):
            global cur_func_name
            if PY3:
                cur_func_name = next(dist_func_names)
            else:
                cur_func_name = dist_func_names.next()
        if ch == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
