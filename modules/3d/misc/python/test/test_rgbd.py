#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import os, numpy
import unittest
import cv2 as cv

from tests_common import NewOpenCVTests

class rgbd_test(NewOpenCVTests):

    def test_computeRgbdPlane(self):

        depth_image = self.get_sample('cv/rgbd/depth.png', cv.IMREAD_ANYDEPTH)
        if depth_image is None:
            raise unittest.SkipTest("Missing files with test data")

        K = numpy.array([[525, 0, 320.5], [0, 525, 240.5], [0, 0, 1]])
        points3d = cv.depthTo3d(depth_image, K)
        normals = cv.RgbdNormals_create(480, 640, cv.CV_32F, K).apply(points3d)
        _, planes_coeff = cv.findPlanes(points3d, normals, numpy.array([]), numpy.array([]), 40, 1600, 0.01, 0, 0, 0, cv.RGBD_PLANE_METHOD_DEFAULT)

        planes_coeff_expected = \
        numpy.asarray([[[-0.02447728, -0.8678335 , -0.49625182,  4.02800846]],
                        [[-0.05055107, -0.86144137, -0.50533485,  3.95456314]],
                        [[-0.03294908, -0.86964548, -0.49257591,  3.97052431]],
                        [[-0.02886586, -0.87153459, -0.48948362,  7.77550507]],
                        [[-0.04455929, -0.87659335, -0.47916424,  3.93200684]],
                        [[-0.21514639,  0.18835169, -0.95824611,  7.59479475]],
                        [[-0.01006953, -0.86679155, -0.49856904,  4.01355648]],
                        [[-0.00876531, -0.87571168, -0.48275498,  3.96768975]],
                        [[-0.06395926, -0.86951321, -0.48975089,  4.08618736]],
                        [[-0.01403128, -0.87593341, -0.48222789,  7.74559402]],
                        [[-0.01143177, -0.87495202, -0.4840748 ,  7.75355816]]],
                        dtype=numpy.float32)

        eps = 0.05
        self.assertLessEqual(cv.norm(planes_coeff, planes_coeff_expected, cv.NORM_L2), eps)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
