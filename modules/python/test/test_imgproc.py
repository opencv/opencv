#!/usr/bin/env python

'''
Test for disctrete fourier transform (dft)
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import numpy as np
import sys

from tests_common import NewOpenCVTests

class imgproc_test(NewOpenCVTests):
    def test_filter2d(self):
        img = self.get_sample('samples/data/lena.jpg', 1)
        eps = 0.001
        # compare 2 ways of computing 3x3 blur using the same function
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype='float32')
        img_blur0 = cv.filter2D(img, cv.CV_32F, kernel*(1./9))
        img_blur1 = cv.filter2Dp(img, kernel, ddepth=cv.CV_32F, scale=1./9)
        self.assertLess(cv.norm(img_blur0 - img_blur1, cv.NORM_INF), eps)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
