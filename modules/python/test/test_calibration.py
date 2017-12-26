#!/usr/bin/env python

'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class calibration_test(NewOpenCVTests):

    def test_calibration(self):
        img_names = []
        for i in range(1, 15):
            if i < 10:
                img_names.append('samples/data/left0{}.jpg'.format(str(i)))
            elif i != 10:
                img_names.append('samples/data/left{}.jpg'.format(str(i)))

        square_size = 1.0
        pattern_size = (9, 6)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size

        obj_points = []
        img_points = []
        h, w = 0, 0
        for fn in img_names:
            img = self.get_sample(fn, 0)
            if img is None:
                continue

            h, w = img.shape[:2]
            found, corners = cv.findChessboardCorners(img, pattern_size)
            if found:
                term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
                cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

            if not found:
                continue

            img_points.append(corners.reshape(-1, 2))
            obj_points.append(pattern_points)

        # calculate camera distortion
        rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None, flags = 0)

        eps = 0.01
        normCamEps = 10.0
        normDistEps = 0.05

        cameraMatrixTest = [[ 532.80992189,    0.,          342.4952186 ],
         [   0.,         532.93346422,  233.8879292 ],
         [   0.,            0.,            1.        ]]

        distCoeffsTest = [ -2.81325576e-01,   2.91130406e-02,
           1.21234330e-03,  -1.40825372e-04, 1.54865844e-01]

        self.assertLess(abs(rms - 0.196334638034), eps)
        self.assertLess(cv.norm(camera_matrix - cameraMatrixTest, cv.NORM_L1), normCamEps)
        self.assertLess(cv.norm(dist_coefs - distCoeffsTest, cv.NORM_L1), normDistEps)



if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
