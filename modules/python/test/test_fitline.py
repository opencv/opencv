#!/usr/bin/env python

'''
Robust line fitting.
==================

Example of using cv2.fitLine function for fitting line
to points in presence of outliers.

Switch through different M-estimator functions and see,
how well the robust functions fit the line even
in case of ~50% of outliers.

'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

import numpy as np
import cv2

from tests_common import NewOpenCVTests

w, h = 512, 256

def toint(p):
    return tuple(map(int, p))

def sample_line(p1, p2, n, noise=0.0):
    np.random.seed(10)
    p1 = np.float32(p1)
    t = np.random.rand(n,1)
    return p1 + (p2-p1)*t + np.random.normal(size=(n, 2))*noise

dist_func_names = ['DIST_L2', 'DIST_L1', 'DIST_L12', 'DIST_FAIR', 'DIST_WELSCH', 'DIST_HUBER']

class fitline_test(NewOpenCVTests):

    def test_fitline(self):

        noise = 5
        n = 200
        r = 5 / 100.0
        outn = int(n*r)

        p0, p1 = (90, 80), (w-90, h-80)
        line_points = sample_line(p0, p1, n-outn, noise)
        outliers = np.random.rand(outn, 2) * (w, h)
        points = np.vstack([line_points, outliers])

        lines = []

        for name in dist_func_names:
            func = getattr(cv2, name)
            vx, vy, cx, cy = cv2.fitLine(np.float32(points), func, 0, 0.01, 0.01)
            line = [float(vx), float(vy), float(cx), float(cy)]
            lines.append(line)

        eps = 0.05

        refVec =  (np.float32(p1) - p0) / cv2.norm(np.float32(p1) - p0)

        for i in range(len(lines)):
            self.assertLessEqual(cv2.norm(refVec - lines[i][0:2], cv2.NORM_L2), eps)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
