#!/usr/bin/env python

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

# Hopefully it's possible to use plaidml backend as well
pkgs = [
         cv.gapi.core.ocl.kernels(),
         cv.gapi.core.cpu.kernels(),
         cv.gapi.core.fluid.kernels()
         # cv.gapi.core.plaidml.kernels()
       ]

class gapi_core_test(NewOpenCVTests):

    def test_core_add(self):
        # FIXME: Expand to use any possible types and shapes
        in1 = np.random.randint(0, 100, (3, 3, 3))
        in2 = np.random.randint(0, 100, (3, 3, 3))

        # OpenCV
        expected = in1 + in2

        # G-API
        g_in1 = cv.GMat()
        g_in2 = cv.GMat()
        g_out = cv.gapi.add(g_in1, g_in2)
        comp = cv.GComputation(g_in1, g_in2, g_out)

        for pkg in pkgs:
            actual = comp.apply(in1, in2, args=cv.compile_args(pkg))
            # Comparison
            self.assertEqual(expected.all(), actual.all())


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
