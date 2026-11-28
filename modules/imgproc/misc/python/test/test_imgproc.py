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

    def test_warp_reject_nd_input(self):
        # https://github.com/opencv/opencv/issues/29589
        batch = np.zeros((4, 3, 5, 3), dtype=np.uint8)
        map_x, map_y = np.meshgrid(np.arange(5, dtype=np.float32), np.arange(3, dtype=np.float32))
        affine = np.eye(2, 3, dtype=np.float32)
        perspective = np.eye(3, dtype=np.float32)

        with self.assertRaises(cv.error):
            cv.remap(batch, map_x, map_y, cv.INTER_NEAREST)
        with self.assertRaises(cv.error):
            cv.warpAffine(batch, affine, (5, 3), flags=cv.INTER_NEAREST)
        with self.assertRaises(cv.error):
            cv.warpPerspective(batch, perspective, (5, 3), flags=cv.INTER_NEAREST)

        # a single HWC image is still accepted
        res = cv.remap(batch[0], map_x, map_y, cv.INTER_NEAREST)
        self.assertEqual(res.shape, (3, 5, 3))

    def test_resize_params(self):
        img = np.random.randint(0, 256, size=(37, 51, 3), dtype=np.uint8)

        # The default parameters are classic resize, given a size or a scale.
        params = cv.ResizeParams((24, 17))
        self.assertEqual(params.coordMode, cv.ResizeCoord_PIXEL_CENTER)
        np.testing.assert_array_equal(cv.resize(img, params), cv.resize(img, (24, 17)))
        np.testing.assert_array_equal(cv.resize(img, cv.ResizeParams((0, 0), 0.5, 0.5)),
                                      cv.resize(img, None, fx=0.5, fy=0.5))

        # A 4-D array is a batch: every (n, c) plane is resized to dsize.
        batch = np.random.rand(2, 3, 16, 16).astype(np.float32)
        params = cv.ResizeParams((32, 8), interpolation=cv.INTER_NEAREST)
        params.coordMode = cv.ResizeCoord_ASYMMETRIC
        params.nearestMode = cv.ResizeNearest_FLOOR
        out = cv.resize(batch, params)
        self.assertEqual(out.shape, (2, 3, 8, 32))
        # asymmetric + floor takes src[floor(dst * in / out)] on each axis
        rows = np.arange(8) * 2
        cols = np.arange(32) // 2
        np.testing.assert_array_equal(out, batch[:, :, rows][:, :, :, cols])

        # Each plane of a batch comes out exactly as it does on its own.
        params = cv.ResizeParams((12, 9), interpolation=cv.INTER_CUBIC)
        params.coordMode = cv.ResizeCoord_HALF_PIXEL
        params.cubicCoeffA = -0.5
        params.excludeOutside = True
        out = cv.resize(batch, params)
        self.assertEqual(out.shape, (2, 3, 9, 12))
        for n in range(2):
            for c in range(3):
                np.testing.assert_array_equal(out[n, c], cv.resize(batch[n, c], params))
