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

    def test_resize_batch_matches_independent_calls(self):
        rng = np.random.RandomState(12345)
        # heterogeneous batch: different sizes, channel counts, and dtypes on purpose,
        # since resizeBatch makes no homogeneity assumption -- each element is resized
        # exactly as an independent cv.resize call would.
        images = [
            rng.randint(0, 255, size=(64, 96, 3), dtype=np.uint8),
            rng.randint(0, 255, size=(50, 50), dtype=np.uint8),
            rng.uniform(0, 1, size=(80, 40, 4)).astype(np.float32),
        ]
        dsize = (32, 24)

        for interpolation in (cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_AREA):
            expected = [cv.resize(img, dsize, interpolation=interpolation) for img in images]
            actual = cv.resizeBatch(images, dsize, interpolation=interpolation)

            self.assertEqual(len(actual), len(expected))
            for e, a in zip(expected, actual):
                self.assertEqual(e.shape, a.shape)
                self.assertEqual(e.dtype, a.dtype)
                np.testing.assert_array_equal(e, a)

    def test_resize_batch_empty(self):
        result = cv.resizeBatch([], (32, 32))
        self.assertEqual(len(result), 0)

    def test_resize_batch_with_scale_factors(self):
        rng = np.random.RandomState(42)
        images = [rng.randint(0, 255, size=(40, 60, 3), dtype=np.uint8) for _ in range(4)]

        expected = [cv.resize(img, None, fx=0.5, fy=0.25, interpolation=cv.INTER_LINEAR) for img in images]
        actual = cv.resizeBatch(images, None, fx=0.5, fy=0.25, interpolation=cv.INTER_LINEAR)

        for e, a in zip(expected, actual):
            np.testing.assert_array_equal(e, a)

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
