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

    class gapi_kernels_test(NewOpenCVTests):

        def test_fluid_core_package(self):
            fluid_core = cv.gapi.core.fluid.kernels()
            self.assertLess(0, fluid_core.size())

        def test_fluid_imgproc_package(self):
            fluid_imgproc = cv.gapi.imgproc.fluid.kernels()
            self.assertLess(0, fluid_imgproc.size())

        def test_combine(self):
            fluid_core = cv.gapi.core.fluid.kernels()
            fluid_imgproc = cv.gapi.imgproc.fluid.kernels()
            fluid = cv.gapi.combine(fluid_core, fluid_imgproc)
            self.assertEqual(fluid_core.size() + fluid_imgproc.size(), fluid.size())

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
