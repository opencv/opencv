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
                actual = comp.apply(cv.gin(in1, in2), args=cv.gapi.compile_args(pkg))
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
                actual = comp.apply(cv.gin(in1, in2), args=cv.gapi.compile_args(pkg))
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
                actual = comp.apply(cv.gin(in_mat), args=cv.gapi.compile_args(pkg))
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
                actual = comp.apply(cv.gin(in_mat), args=cv.gapi.compile_args(pkg))
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
                actual_mat, actual_thresh = comp.apply(cv.gin(in_mat, maxv), args=cv.gapi.compile_args(pkg))
                # Comparison
                self.assertEqual(0.0, cv.norm(expected_mat, actual_mat, cv.NORM_INF),
                                 'Failed on ' + pkg_name + ' backend')
                self.assertEqual(expected_mat.dtype, actual_mat.dtype,
                                 'Failed on ' + pkg_name + ' backend')
                self.assertEqual(expected_thresh, actual_thresh[0],
                                 'Failed on ' + pkg_name + ' backend')


        def test_kmeans(self):
            # K-means params
            count    = 100
            sz       = (count, 2)
            in_mat   = np.random.random(sz).astype(np.float32)
            K        = 5
            flags    = cv.KMEANS_RANDOM_CENTERS
            attempts = 1
            criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 30, 0)

            # G-API
            g_in = cv.GMat()
            compactness, out_labels, centers = cv.gapi.kmeans(g_in, K, criteria, attempts, flags)
            comp = cv.GComputation(cv.GIn(g_in), cv.GOut(compactness, out_labels, centers))

            compact, labels, centers = comp.apply(cv.gin(in_mat))

            # Assert
            self.assertTrue(compact >= 0)
            self.assertEqual(sz[0], labels.shape[0])
            self.assertEqual(1, labels.shape[1])
            self.assertTrue(labels.size != 0)
            self.assertEqual(centers.shape[1], sz[1])
            self.assertEqual(centers.shape[0], K)
            self.assertTrue(centers.size != 0)


        def generate_random_points(self, sz):
            arr = np.random.random(sz).astype(np.float32).T
            return list(zip(arr[0], arr[1]))


        def test_kmeans_2d(self):
            # K-means 2D params
            count     = 100
            sz        = (count, 2)
            amount    = sz[0]
            K         = 5
            flags     = cv.KMEANS_RANDOM_CENTERS
            attempts  = 1
            criteria  = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 30, 0)
            in_vector = self.generate_random_points(sz)
            in_labels = []

            # G-API
            data        = cv.GArrayT(cv.gapi.CV_POINT2F)
            best_labels = cv.GArrayT(cv.gapi.CV_INT)

            compactness, out_labels, centers = cv.gapi.kmeans(data, K, best_labels, criteria, attempts, flags)
            comp = cv.GComputation(cv.GIn(data, best_labels), cv.GOut(compactness, out_labels, centers))

            compact, labels, centers = comp.apply(cv.gin(in_vector, in_labels))

            # Assert
            self.assertTrue(compact >= 0)
            self.assertEqual(amount, len(labels))
            self.assertEqual(K, len(centers))


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
