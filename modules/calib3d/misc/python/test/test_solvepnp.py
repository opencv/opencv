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


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
