#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class Imgproc_Tests(NewOpenCVTests):

    def test_python_986(self):
        cntls = []
        img = np.zeros((100,100,3), dtype=np.uint8)
        color = (0,0,0)
        cnts = np.array(cntls, dtype=np.int32).reshape((1, -1, 2))
        try:
            cv.fillPoly(img, cnts, color)
            assert False
        except:
            assert True

    # See https://github.com/opencv/opencv/issues/29938
    def test_calcBackProject_3d(self):
        # A 3D histogram reaches the native layer as a 2D multi-channel Mat.
        # calcBackProject() has to restore the third dimension before the
        # lookup, otherwise the last channel of the image is ignored.
        channels = [0, 1, 2]
        ranges = [0, 256, 0, 256, 0, 256]

        roi = np.array([[[10, 20, 30]]], dtype=np.uint8)
        hist = cv.calcHist([roi], channels, None, [8, 8, 8], ranges)
        self.assertEqual(hist.shape, (8, 8, 8))

        img = np.array([[[10, 20, 30], [10, 20, 130]]], dtype=np.uint8)
        bp = cv.calcBackProject([img], channels, hist, ranges, 255.0)

        self.assertEqual(bp[0, 0], 255)  # present in the histogram
        self.assertEqual(bp[0, 1], 0)    # differs in the 3rd channel only
