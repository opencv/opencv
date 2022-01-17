#!/usr/bin/env python

import numpy as np
import cv2 as cv
import unittest

from tests_common import NewOpenCVTests

class odometry_test(NewOpenCVTests):

    def test_OdometryDepth(self):
        depth_image = self.get_sample('cv/rgbd/depth.png', cv.IMREAD_ANYDEPTH)
        Rt = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        )
        Rt_res = np.zeros((4, 4))

        odometry = cv.Odometry(cv.DEPTH)
        isCorrect = odometry.compute(depth_image, depth_image, Rt_res)

        res = (Rt - Rt_res).sum()

        eps = 0.05
        self.assertLessEqual(res, eps)
        self.assertTrue(isCorrect)

    def test_OdometryRGB(self):
        rgb_image = self.get_sample('cv/rgbd/rgb.png', cv.IMREAD_ANYCOLOR)
        Rt = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        )
        Rt_res = np.zeros((4, 4))

        odometry = cv.Odometry(cv.RGB)
        isCorrect = odometry.compute(rgb_image, rgb_image, Rt_res)

        res = (Rt - Rt_res).sum()
        
        eps = 0.05
        self.assertLessEqual(res, eps)
        self.assertTrue(isCorrect)

    def test_OdometryRGB(self):
        depth_image = self.get_sample('cv/rgbd/depth.png', cv.IMREAD_ANYDEPTH)
        rgb_image = self.get_sample('cv/rgbd/rgb.png', cv.IMREAD_ANYCOLOR)
        Rt = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        )
        Rt_res = np.zeros((4, 4))

        odometry = cv.Odometry(cv.RGB_DEPTH)
        isCorrect = odometry.compute(depth_image, rgb_image, depth_image, rgb_image, Rt_res)

        res = (Rt - Rt_res).sum()

        eps = 0.05
        self.assertLessEqual(res, eps)
        self.assertTrue(isCorrect)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()


