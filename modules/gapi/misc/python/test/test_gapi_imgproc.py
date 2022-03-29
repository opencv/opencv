#!/usr/bin/env python

import numpy as np
import cv2 as cv
import os
import sys
import unittest

from tests_common import NewOpenCVTests


try:

    if sys.version_info[:2] < (3, 0):
        raise unittest.SkipTest('Python 2.x is not supported')

    # Plaidml is an optional backend
    pkgs = [
               ('ocl'    , cv.gapi.core.ocl.kernels()),
               ('cpu'    , cv.gapi.core.cpu.kernels()),
               ('fluid'  , cv.gapi.core.fluid.kernels())
               # ('plaidml', cv.gapi.core.plaidml.kernels())
           ]


    class gapi_imgproc_test(NewOpenCVTests):

        def test_good_features_to_track(self):
            # TODO: Extend to use any type and size here
            img_path = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
            in1 = cv.cvtColor(cv.imread(img_path), cv.COLOR_RGB2GRAY)

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

            for pkg_name, pkg in pkgs:
                actual = comp.apply(cv.gin(in1), args=cv.gapi.compile_args(pkg))
                # NB: OpenCV & G-API have different output shapes:
                # OpenCV - (num_points, 1, 2)
                # G-API  - (num_points, 2)
                # Comparison
                self.assertEqual(0.0, cv.norm(expected.flatten(),
                                              np.array(actual, dtype=np.float32).flatten(),
                                              cv.NORM_INF),
                                 'Failed on ' + pkg_name + ' backend')


        def test_rgb2gray(self):
            # TODO: Extend to use any type and size here
            img_path = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
            in1 = cv.imread(img_path)

            # OpenCV
            expected = cv.cvtColor(in1, cv.COLOR_RGB2GRAY)

            # G-API
            g_in = cv.GMat()
            g_out = cv.gapi.RGB2Gray(g_in)

            comp = cv.GComputation(cv.GIn(g_in), cv.GOut(g_out))

            for pkg_name, pkg in pkgs:
                actual = comp.apply(cv.gin(in1), args=cv.gapi.compile_args(pkg))
                # Comparison
                self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF),
                                 'Failed on ' + pkg_name + ' backend')


        def test_bounding_rect(self):
            sz = 1280
            fscale = 256

            def sample_value(fscale):
                return np.random.uniform(0, 255 * fscale) / fscale

            points = np.array([(sample_value(fscale), sample_value(fscale)) for _ in range(1280)], np.float32)

            # OpenCV
            expected = cv.boundingRect(points)

            # G-API
            g_in  = cv.GMat()
            g_out = cv.gapi.boundingRect(g_in)

            comp = cv.GComputation(cv.GIn(g_in), cv.GOut(g_out))

            for pkg_name, pkg in pkgs:
                actual = comp.apply(cv.gin(points), args=cv.gapi.compile_args(pkg))
                # Comparison
                self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF),
                                 'Failed on ' + pkg_name + ' backend')


except unittest.SkipTest as e:

    message = str(e)

    class TestSkip(unittest.TestCase):
        def setUp(self):
            self.skipTest('Skip tests: ' + message)

        def test_skip():
            pass

    pass


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
