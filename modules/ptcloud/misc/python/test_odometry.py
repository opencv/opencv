#!/usr/bin/env python

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class odometry_test(NewOpenCVTests):
    def commonOdometryTest(self, needRgb, otype, algoType, useFrame):
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
            settings = cv.OdometrySettings()
            odometry = cv.Odometry(otype, settings, algoType)
        else:
            odometry = cv.Odometry()

        warped_depth = cv.warpPerspective(depth, Rt_warp, (640, 480))
        if needRgb:
            warped_rgb = cv.warpPerspective(rgb, Rt_warp, (640, 480))

        if useFrame:
            if needRgb:
                srcFrame = cv.OdometryFrame(depth, rgb)
                dstFrame = cv.OdometryFrame(warped_depth, warped_rgb)
            else:
                srcFrame = cv.OdometryFrame(depth)
                dstFrame = cv.OdometryFrame(warped_depth)
            odometry.prepareFrames(srcFrame, dstFrame)
            isCorrect = odometry.compute(srcFrame, dstFrame, Rt_res)
        else:
            if needRgb:
                isCorrect = odometry.compute(depth, rgb, warped_depth, warped_rgb, Rt_res)
            else:
                isCorrect = odometry.compute(depth, warped_depth, Rt_res)

        res = np.absolute(Rt_curr - Rt_res).sum()
        eps = 0.15
        self.assertLessEqual(res, eps)
        self.assertTrue(isCorrect)

    def test_OdometryDefault(self):
        self.commonOdometryTest(False, None, None, False)

    def test_OdometryDefaultFrame(self):
        self.commonOdometryTest(False, None, None, True)

    def test_OdometryDepth(self):
        self.commonOdometryTest(False, cv.OdometryType_DEPTH, cv.OdometryAlgoType_COMMON, False)

    def test_OdometryDepthFast(self):
        self.commonOdometryTest(False, cv.OdometryType_DEPTH, cv.OdometryAlgoType_FAST, False)

    def test_OdometryDepthFrame(self):
        self.commonOdometryTest(False, cv.OdometryType_DEPTH, cv.OdometryAlgoType_COMMON, True)

    def test_OdometryDepthFastFrame(self):
        self.commonOdometryTest(False, cv.OdometryType_DEPTH, cv.OdometryAlgoType_FAST, True)

    def test_OdometryRGB(self):
        self.commonOdometryTest(True, cv.OdometryType_RGB, cv.OdometryAlgoType_COMMON, False)

    def test_OdometryRGBFrame(self):
        self.commonOdometryTest(True, cv.OdometryType_RGB, cv.OdometryAlgoType_COMMON, True)

    def test_OdometryRGB_Depth(self):
        self.commonOdometryTest(True, cv.OdometryType_RGB_DEPTH, cv.OdometryAlgoType_COMMON, False)

    def test_OdometryRGB_DepthFrame(self):
        self.commonOdometryTest(True, cv.OdometryType_RGB_DEPTH, cv.OdometryAlgoType_COMMON, True)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
