#!/usr/bin/env python

import numpy as np
import cv2 as cv
import itertools

from tests_common import NewOpenCVTests


# Plaidml is an optional backend
pkgs = [
         cv.gapi.core.ocl.kernels(),
         cv.gapi.core.cpu.kernels(),
         cv.gapi.core.fluid.kernels()
         # cv.gapi.core.plaidml.kernels()
       ]


def combine(cases):
    def decorate(func):
        def call(*args):
            for case in itertools.product(*cases):
                func(*args, *case)
        return call
    return decorate


class gapi_core_test(NewOpenCVTests):

    @combine([pkgs, [(1280, 720), (640, 480)]])
    def test_add(self, pkg, sz):
        # TODO: Extend to use any type here
        in1 = np.random.randint(0, 100, sz).astype(np.uint8)
        in2 = np.random.randint(0, 100, sz).astype(np.uint8)

        # OpenCV
        expected = in1 + in2

        # G-API
        g_in1 = cv.GMat()
        g_in2 = cv.GMat()
        g_out = cv.gapi.add(g_in1, g_in2)
        comp = cv.GComputation(g_in1, g_in2, g_out)

        actual = comp.apply(in1, in2, args=cv.compile_args(pkg))
        # Comparison
        np.testing.assert_array_equal(expected, actual)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
