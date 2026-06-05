#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import cv2 as cv
try:
    from PIL import Image
except ImportError:
    Image = None

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

    def test_resize_pillow(self):
        if Image is None:
            self.skipTest("Pillow is not available")

        r = np.random.randint(0, 255, size=(128, 147, 3), dtype="uint8")
        target_size=[(128,128), (129,129), (160,160)]
        for ts in target_size:

            pil_result = np.array(Image.fromarray(r).resize(ts, Image.NEAREST))
            ocv_result = cv.resize(r, dsize=ts, interpolation=cv.INTER_NEAREST_EXACT)
            status = np.all(pil_result == ocv_result)
            print(ts, status)
            self.assertTrue(status, "resize result differs from Pillow for target size %s" % (ts,))
