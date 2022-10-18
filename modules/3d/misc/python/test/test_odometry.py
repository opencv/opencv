#!/usr/bin/env python

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class odometry_test(NewOpenCVTests):
    def testCommon(self, needRgb, otype):
        depth = self.get_sample('cv/rgbd/depth.png', cv.IMREAD_ANYDEPTH).astype(np.float32)
        if needRgb:
            rgb = self.get_sample('cv/rgbd/rgb.png', cv.IMREAD_ANYCOLOR)
        radian = np.radians(1)
        Rt_warp = np.array(
            [[np.cos(radian), -np.sin(radian), 0],
            [np.sin(radian), np.cos(radian), 0],
            [0, 0, 1]], dtype=np.float32
        )
        Rt_curr = np.array(
            [[np.cos(radian), -np.sin(radian), 0, 0],
            [np.sin(radian), np.cos(radian), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=np.float32
        )
        Rt_res = np.zeros((4, 4))

        if otype is not None:
            odometry = cv.Odometry(otype)
        else:
            odometry = cv.Odometry()
        warped_depth = cv.warpPerspective(depth, Rt_warp, (640, 480))
        if needRgb:
            warped_rgb = cv.warpPerspective(rgb, Rt_warp, (640, 480))
            isCorrect = odometry.compute(depth, rgb, warped_depth, warped_rgb, Rt_res)
        else:
            isCorrect = odometry.compute(depth, warped_depth, Rt_res)

        res = np.absolute(Rt_curr - Rt_res).sum()
        eps = 0.15
        self.assertLessEqual(res, eps)
        self.assertTrue(isCorrect)

    def test_OdometryDefault(self):
        testCommon(self, False, None)

    def test_OdometryDepth(self):
        testCommon(self, False, cv.DEPTH)

    def test_OdometryRGB(self):
        testCommon(self, True, cv.RGB)

    def test_OdometryRGB_Depth(self):
        testCommon(self, True, cv.RGB_DEPTH)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
