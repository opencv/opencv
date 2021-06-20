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

    class gapi_render_test(NewOpenCVTests):

        def test_rect(self):
            size = (100, 100, 3)
            rect = (30, 30, 50, 50)
            color = (0, 255, 0)
            thick = 3
            lt = cv.LINE_4
            shift = 1

            expected = np.zeros(size, dtype=np.uint8)
            actual = np.array(expected, copy=True)

            cv.rectangle(expected, rect, color, thick, lt, shift)

            g_in = cv.GMat()
            g_prims = cv.GArray.Prim()
            g_out = cv.gapi.wip.draw.render3ch(g_in, g_prims)

            prims = [cv.gapi.wip.draw.Rect(rect, color, thick, lt, shift)]
            comp = cv.GComputation(cv.GIn(g_in, g_prims), cv.GOut(g_out))
            actual = comp.apply(cv.gin(actual, prims))

            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


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
