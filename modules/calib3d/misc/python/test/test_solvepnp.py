#!/usr/bin/env python
# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class solvepnp_test(NewOpenCVTests):

    def test_regression_16040(self):
        obj_points = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
        img_points = np.array(
            [[700, 400], [700, 600], [900, 600], [900, 400]], dtype=np.float32
        )

        cameraMatrix = np.array(
            [[712.0634, 0, 800], [0, 712.540, 500], [0, 0, 1]], dtype=np.float32
        )
        distCoeffs = np.array([[0, 0, 0, 0]], dtype=np.float32)
        r = np.array([], dtype=np.float32)
        x, r, t, e = cv.solvePnPGeneric(
            obj_points, img_points, cameraMatrix, distCoeffs, reprojectionError=r
        )

    def test_regression_16040_2(self):
        obj_points = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
        img_points = np.array(
            [[[700, 400], [700, 600], [900, 600], [900, 400]]], dtype=np.float32
        )

        cameraMatrix = np.array(
            [[712.0634, 0, 800], [0, 712.540, 500], [0, 0, 1]], dtype=np.float32
        )
        distCoeffs = np.array([[0, 0, 0, 0]], dtype=np.float32)
        r = np.array([], dtype=np.float32)
        x, r, t, e = cv.solvePnPGeneric(
            obj_points, img_points, cameraMatrix, distCoeffs, reprojectionError=r
        )

    def test_regression_16049(self):
        obj_points = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
        img_points = np.array(
            [[[700, 400], [700, 600], [900, 600], [900, 400]]], dtype=np.float32
        )

        cameraMatrix = np.array(
            [[712.0634, 0, 800], [0, 712.540, 500], [0, 0, 1]], dtype=np.float32
        )
        distCoeffs = np.array([[0, 0, 0, 0]], dtype=np.float32)
        x, r, t, e = cv.solvePnPGeneric(
            obj_points, img_points, cameraMatrix, distCoeffs
        )
        if e is None:
            # noArray() is supported, see https://github.com/opencv/opencv/issues/16049
            pass
        else:
            eDump = cv.utils.dumpInputArray(e)
            self.assertEqual(eDump, "InputArray: empty()=false kind=0x00010000 flags=0x01010000 total(-1)=1 dims(-1)=2 size(-1)=1x1 type(-1)=CV_32FC1")

    def test_SolvePnP(self):
        # analog of c++ Calib3d_SolvePnP.translation
        cameraIntrinsic = np.eye(3, 3, dtype = np.float32)
        crvec = np.zeros(3, dtype = np.float32)
        ctvec = np.array([100, 100, 0], dtype = np.float32)
        p3d = np.array([[0,0,0], [0,0,10], [0,10,10], [10,10,10], [2,5,5], [-4,8,6]], dtype = np.float32)
        p2d, _ = cv.projectPoints(p3d, crvec, ctvec, cameraIntrinsic, None)
        rvec = np.array([0, 0, 0], dtype = np.float32)
        tvec = np.array([100, 100, 0], dtype = np.float32)

        status, rvec, tvec = cv.solvePnP(p3d, p2d, cameraIntrinsic, None, rvec, tvec, useExtrinsicGuess=True)
        status, rvec2, tvec2 = cv.solvePnP(p3d, p2d, cameraIntrinsic, None)

    def test_SolvePnPRansac(self):
        # analog of c++ Calib3d_SolvePnP.translation
        cameraIntrinsic = np.eye(3, 3, dtype = np.float32)
        crvec = np.zeros(3, dtype = np.float32)
        ctvec = np.array([100, 100, 0], dtype = np.float32)
        p3d = np.array([[0,0,0], [0,0,10], [0,10,10], [10,10,10], [2,5,5], [-4,8,6]], dtype = np.float32)
        p2d, _ = cv.projectPoints(p3d, crvec, ctvec, cameraIntrinsic, None)
        rvec = np.array([0, 0, 0], dtype = np.float32)
        tvec = np.array([100, 100, 0], dtype = np.float32)

        status, rvec, tvec = cv.solvePnPRansac(p3d, p2d, cameraIntrinsic, None, rvec, tvec, useExtrinsicGuess=True)
        status, rvec2, tvec2 = cv.solvePnPRansac(p3d, p2d, cameraIntrinsic, None)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
