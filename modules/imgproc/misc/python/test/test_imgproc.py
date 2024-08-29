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

    def test_filter2d(self):
        img = self.get_sample('samples/data/lena.jpg', 1)
        eps = 0.001
        # compare 2 ways of computing 3x3 blur using the same function
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype='float32')
        img_blur0 = cv.filter2D(img, cv.CV_32F, kernel*(1./9))
        img_blur1 = cv.filter2Dp(img, kernel, ddepth=cv.CV_32F, scale=1./9)
        self.assertLess(cv.norm(img_blur0 - img_blur1, cv.NORM_INF), eps)
