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


class gapi_core_test(NewOpenCVTests):

    def test_add(self):
        # TODO: Extend to use any type and size here
        sz = (1280, 720)
        in1 = np.random.randint(0, 100, sz)
        in2 = np.random.randint(0, 100, sz)

        # OpenCV
        expected = cv.add(in1, in2)

        # G-API
        g_in1 = cv.GMat()
        g_in2 = cv.GMat()
        g_out = cv.gapi.add(g_in1, g_in2)
        comp = cv.GComputation(cv.GIn(g_in1, g_in2), cv.GOut(g_out))

        for pkg in pkgs:
            actual = comp.apply(cv.gin(in1, in2), args=cv.compile_args(pkg))
            # Comparison
            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))
            self.assertEqual(expected.dtype, actual.dtype)


    def test_add_uint8(self):
        sz = (1280, 720)
        in1 = np.random.randint(0, 100, sz).astype(np.uint8)
        in2 = np.random.randint(0, 100, sz).astype(np.uint8)

        # OpenCV
        expected = cv.add(in1, in2)

        # G-API
        g_in1 = cv.GMat()
        g_in2 = cv.GMat()
        g_out = cv.gapi.add(g_in1, g_in2)
        comp = cv.GComputation(cv.GIn(g_in1, g_in2), cv.GOut(g_out))

        for pkg in pkgs:
            actual = comp.apply(cv.gin(in1, in2), args=cv.compile_args(pkg))
            # Comparison
            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))
            self.assertEqual(expected.dtype, actual.dtype)


    def test_mean(self):
        sz = (1280, 720, 3)
        in_mat = np.random.randint(0, 100, sz)

        # OpenCV
        expected = cv.mean(in_mat)

        # G-API
        g_in = cv.GMat()
        g_out = cv.gapi.mean(g_in)
        comp = cv.GComputation(g_in, g_out)

        for pkg in pkgs:
            actual = comp.apply(cv.gin(in_mat), args=cv.compile_args(pkg))
            # Comparison
            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF))


    def test_split3(self):
        sz = (1280, 720, 3)
        in_mat = np.random.randint(0, 100, sz)

        # OpenCV
        expected = cv.split(in_mat)

        # G-API
        g_in = cv.GMat()
        b, g, r = cv.gapi.split3(g_in)
        comp = cv.GComputation(cv.GIn(g_in), cv.GOut(b, g, r))

        for pkg in pkgs:
            actual = comp.apply(cv.gin(in_mat), args=cv.compile_args(pkg))
            # Comparison
            for e, a in zip(expected, actual):
                self.assertEqual(0.0, cv.norm(e, a, cv.NORM_INF))
                self.assertEqual(e.dtype, a.dtype)


    def test_threshold(self):
        sz = (1280, 720)
        in_mat = np.random.randint(0, 100, sz).astype(np.uint8)
        rand_int = np.random.randint(0, 50)
        maxv = (rand_int, rand_int)

        # OpenCV
        expected_thresh, expected_mat = cv.threshold(in_mat, maxv[0], maxv[0], cv.THRESH_TRIANGLE)

        # G-API
        g_in = cv.GMat()
        g_sc = cv.GScalar()
        mat, threshold = cv.gapi.threshold(g_in, g_sc, cv.THRESH_TRIANGLE)
        comp = cv.GComputation(cv.GIn(g_in, g_sc), cv.GOut(mat, threshold))

        for pkg in pkgs:
            actual_mat, actual_thresh = comp.apply(cv.gin(in_mat, maxv), args=cv.compile_args(pkg))
            # Comparison
            self.assertEqual(0.0, cv.norm(expected_mat, actual_mat, cv.NORM_INF))
            self.assertEqual(expected_mat.dtype, actual_mat.dtype)
            self.assertEqual(expected_thresh, actual_thresh[0])


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
