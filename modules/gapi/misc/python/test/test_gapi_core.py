#!/usr/bin/env python

import numpy as np
import cv2 as cv
import os

from tests_common import NewOpenCVTests


# Plaidml is an optional backend
pkgs = [
          ('ocl'    , cv.gapi.core.ocl.kernels()),
          ('cpu'    , cv.gapi.core.cpu.kernels()),
          ('fluid'  , cv.gapi.core.fluid.kernels())
          # ('plaidml', cv.gapi.core.plaidml.kernels())
      ]


class gapi_core_test(NewOpenCVTests):

    def test_add(self):
        # TODO: Extend to use any type and size here
        sz = (720, 1280)
        in1 = np.full(sz, 100)
        in2 = np.full(sz, 50)

        # OpenCV
        expected = cv.add(in1, in2)

        # G-API
        g_in1 = cv.GMat()
        g_in2 = cv.GMat()
        g_out = cv.gapi.add(g_in1, g_in2)
        comp = cv.GComputation(cv.GIn(g_in1, g_in2), cv.GOut(g_out))

        for pkg_name, pkg in pkgs:
            actual = comp.apply(cv.gin(in1, in2), args=cv.compile_args(pkg))
            # Comparison
            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF),
                             'Failed on ' + pkg_name + ' backend')
            self.assertEqual(expected.dtype, actual.dtype, 'Failed on ' + pkg_name + ' backend')


    def test_add_uint8(self):
        sz = (720, 1280)
        in1 = np.full(sz, 100, dtype=np.uint8)
        in2 = np.full(sz, 50 , dtype=np.uint8)

        # OpenCV
        expected = cv.add(in1, in2)

        # G-API
        g_in1 = cv.GMat()
        g_in2 = cv.GMat()
        g_out = cv.gapi.add(g_in1, g_in2)
        comp = cv.GComputation(cv.GIn(g_in1, g_in2), cv.GOut(g_out))

        for pkg_name, pkg in pkgs:
            actual = comp.apply(cv.gin(in1, in2), args=cv.compile_args(pkg))
            # Comparison
            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF),
                             'Failed on ' + pkg_name + ' backend')
            self.assertEqual(expected.dtype, actual.dtype, 'Failed on ' + pkg_name + ' backend')


    def test_mean(self):
        img_path = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
        in_mat = cv.imread(img_path)

        # OpenCV
        expected = cv.mean(in_mat)

        # G-API
        g_in = cv.GMat()
        g_out = cv.gapi.mean(g_in)
        comp = cv.GComputation(g_in, g_out)

        for pkg_name, pkg in pkgs:
            actual = comp.apply(cv.gin(in_mat), args=cv.compile_args(pkg))
            # Comparison
            self.assertEqual(0.0, cv.norm(expected, actual, cv.NORM_INF),
                             'Failed on ' + pkg_name + ' backend')


    def test_split3(self):
        img_path = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
        in_mat = cv.imread(img_path)

        # OpenCV
        expected = cv.split(in_mat)

        # G-API
        g_in = cv.GMat()
        b, g, r = cv.gapi.split3(g_in)
        comp = cv.GComputation(cv.GIn(g_in), cv.GOut(b, g, r))

        for pkg_name, pkg in pkgs:
            actual = comp.apply(cv.gin(in_mat), args=cv.compile_args(pkg))
            # Comparison
            for e, a in zip(expected, actual):
                self.assertEqual(0.0, cv.norm(e, a, cv.NORM_INF),
                                 'Failed on ' + pkg_name + ' backend')
                self.assertEqual(e.dtype, a.dtype, 'Failed on ' + pkg_name + ' backend')


    def test_threshold(self):
        img_path = self.find_file('cv/face/david2.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
        in_mat = cv.cvtColor(cv.imread(img_path), cv.COLOR_RGB2GRAY)
        maxv = (30, 30)

        # OpenCV
        expected_thresh, expected_mat = cv.threshold(in_mat, maxv[0], maxv[0], cv.THRESH_TRIANGLE)

        # G-API
        g_in = cv.GMat()
        g_sc = cv.GScalar()
        mat, threshold = cv.gapi.threshold(g_in, g_sc, cv.THRESH_TRIANGLE)
        comp = cv.GComputation(cv.GIn(g_in, g_sc), cv.GOut(mat, threshold))

        for pkg_name, pkg in pkgs:
            actual_mat, actual_thresh = comp.apply(cv.gin(in_mat, maxv), args=cv.compile_args(pkg))
            # Comparison
            self.assertEqual(0.0, cv.norm(expected_mat, actual_mat, cv.NORM_INF),
                             'Failed on ' + pkg_name + ' backend')
            self.assertEqual(expected_mat.dtype, actual_mat.dtype,
                             'Failed on ' + pkg_name + ' backend')
            self.assertEqual(expected_thresh, actual_thresh[0],
                             'Failed on ' + pkg_name + ' backend')


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
