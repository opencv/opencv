#!/usr/bin/env python

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests


# Plaidml is an optional backend

pkgs = [
        cv.gapi.core.ocl.kernels(),
        cv.gapi.core.cpu.kernels(),
        cv.gapi.core.fluid.kernels()
        # cv.gapi.core.plaidml.kernels()
      ]


class gapi_imgproc_test(NewOpenCVTests):

    def test_good_features_to_track(self):
        # TODO: Extend to use any type and size here
        sz = (1280, 720)
        in1 = np.random.randint(0, 100, sz).astype(np.uint8)

        # NB: goodFeaturesToTrack configuration
        max_corners         = 50
        quality_lvl         = 0.01
        min_distance        = 10
        block_sz            = 3
        use_harris_detector = True
        k                   = 0.04
        mask                = None

        # OpenCV
        expected = cv.goodFeaturesToTrack(in1, max_corners, quality_lvl,
                                          min_distance, mask=mask,
                                          blockSize=block_sz, useHarrisDetector=use_harris_detector, k=k)

        # G-API
        g_in = cv.GMat()
        g_out = cv.gapi.goodFeaturesToTrack(g_in, max_corners, quality_lvl,
                                            min_distance, mask, block_sz, use_harris_detector, k)

        comp = cv.GComputation(cv.GIn(g_in), cv.GOut(g_out))

        for pkg in pkgs:
            actual = comp.apply(cv.gin(in1), args=cv.compile_args(pkg))
            # NB: OpenCV & G-API have different output shapes:
            # OpenCV - (num_points, 1, 2)
            # G-API  - (num_points, 2)
            # Comparison
            self.assertEqual(0.0, cv.norm(expected.flatten(), actual.flatten(), cv.NORM_INF))


    def test_rgb2gray(self):
        # TODO: Extend to use any type and size here
        sz = (1280, 720, 3)
        in1 = np.random.randint(0, 100, sz).astype(np.uint8)

        # OpenCV
        expected = cv.cvtColor(in1, cv.COLOR_RGB2GRAY)

        # G-API
        g_in = cv.GMat()
        g_out = cv.gapi.RGB2Gray(g_in)

        comp = cv.GComputation(cv.GIn(g_in), cv.GOut(g_out))

        for pkg in pkgs:
            actual = comp.apply(cv.gin(in1), args=cv.compile_args(pkg))
            # Comparison
            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
