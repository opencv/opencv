#!/usr/bin/env python

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class odometry_test(NewOpenCVTests):
    def test_OdometryDefault(self):
        depth = self.get_sample('cv/rgbd/depth.png', cv.IMREAD_ANYDEPTH).astype(np.float32)
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

        odometry = cv.Odometry()
        warped_depth = cv.warpPerspective(depth, Rt_warp, (640, 480))
        isCorrect = odometry.compute(depth, warped_depth, Rt_res)

        res = np.absolute(Rt_curr - Rt_res).sum()
        eps = 0.15
        self.assertLessEqual(res, eps)
        self.assertTrue(isCorrect)

    def test_OdometryDepth(self):
        depth = self.get_sample('cv/rgbd/depth.png', cv.IMREAD_ANYDEPTH).astype(np.float32)
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

        odometry = cv.Odometry(cv.DEPTH)
        warped_depth = cv.warpPerspective(depth, Rt_warp, (640, 480))

        isCorrect = odometry.compute(depth, warped_depth, Rt_res)
        res = np.absolute(Rt_curr - Rt_res).sum()
        eps = 0.15
        self.assertLessEqual(res, eps)
        self.assertTrue(isCorrect)

    def test_OdometryRGB(self):
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

        odometry = cv.Odometry(cv.RGB)
        warped_rgb = cv.warpPerspective(rgb, Rt_warp, (640, 480))
        isCorrect = odometry.compute(rgb, warped_rgb, Rt_res)

        res = np.absolute(Rt_curr - Rt_res).sum()
        eps = 0.15
        self.assertLessEqual(res, eps)
        self.assertTrue(isCorrect)

    def test_OdometryRGB_Depth(self):
        depth = self.get_sample('cv/rgbd/depth.png', cv.IMREAD_ANYDEPTH).astype(np.float32)
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

        odometry = cv.Odometry(cv.RGB_DEPTH)
        warped_depth = cv.warpPerspective(depth, Rt_warp, (640, 480))
        warped_rgb = cv.warpPerspective(rgb, Rt_warp, (640, 480))
        isCorrect = odometry.compute(depth, rgb, warped_depth, warped_rgb, Rt_res)

        res = np.absolute(Rt_curr - Rt_res).sum()
        eps = 0.15
        self.assertLessEqual(res, eps)
        self.assertTrue(isCorrect)

    def test_OdometryScale(self):
        depth = self.get_sample('cv/rgbd/depth.png', cv.IMREAD_ANYDEPTH).astype(np.float32)
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
        scale = 1.01
        scale_res = np.zeros((1, 1))

        odometry = cv.Odometry()
        warped_depth = cv.warpPerspective(depth, Rt_warp, (640, 480))

        isCorrect = odometry.compute(depth, warped_depth*scale, Rt_res, scale_res)
        Rt_diff = np.absolute(Rt_curr - Rt_res).sum()
        scale_diff = np.absolute(scale - scale_res[0][0])

        Rt_eps = 0.2
        scale_eps = 0.1
        self.assertLessEqual(Rt_diff, Rt_eps)
        self.assertLessEqual(scale_diff, scale_eps)
        self.assertTrue(isCorrect)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
