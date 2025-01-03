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

    class gapi_ot_test(NewOpenCVTests):

        def test_ot_smoke(self):
            # Input
            img_path = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
            in_image = cv.cvtColor(cv.imread(img_path), cv.COLOR_RGB2BGR)
            in_rects = [ (138, 89, 71, 64) ]
            in_rects_cls = [ 0 ]

            # G-API
            g_in = cv.GMat()
            g_in_rects = cv.GArray.Rect()
            g_in_rects_cls = cv.GArray.Int()
            delta = 0.5

            g_out_rects, g_out_rects_cls, g_track_ids, g_track_sts = \
                cv.gapi.ot.track(g_in, g_in_rects, g_in_rects_cls, delta)


            comp = cv.GComputation(cv.GIn(g_in, g_in_rects, g_in_rects_cls),
                                   cv.GOut(g_out_rects, g_out_rects_cls,
                                           g_track_ids, g_track_sts))

            __, __, __, sts = comp.apply(cv.gin(in_image, in_rects, in_rects_cls),
                args=cv.gapi.compile_args(cv.gapi.ot.cpu.kernels()))

            self.assertEqual(cv.gapi.ot.NEW, sts[0])

except unittest.SkipTest as e:

    message = str(e)

    class TestSkip(unittest.TestCase):
        def setUp(self):
            self.skipTest('Skip tests: ' + message)

        def test_skip():
            pass


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
