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

    def test_projectPoints(self):
        objectPoints = np.array([[181.24588 ,  87.80361 ,  11.421074],
            [ 87.17948 , 184.75563 ,  37.223446],
            [ 22.558456,  45.495266, 246.05797 ]], dtype=np.float32)
        rvec = np.array([[ 0.9357548 , -0.28316498,  0.21019171],
            [ 0.30293274,  0.9505806 , -0.06803132],
            [-0.18054008,  0.12733458,  0.9752903 ]], dtype=np.float32)
        tvec = np.array([ 69.32692 ,  17.602057, 135.77672 ], dtype=np.float32)
        cameraMatrix = np.array([[214.0047  ,  26.98735 , 253.37799 ],
            [189.8172  ,  10.038101,  18.862494],
            [114.07123 , 200.87277 , 194.56332 ]], dtype=np.float32)
        distCoeffs = distCoeffs = np.zeros((4, 1), dtype=np.float32)

        imagePoints, jacobian = cv.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)
        self.assertTrue(imagePoints is not None)
        self.assertTrue(jacobian is not None)

    def test_sampsonDistance_valid2D(self):
        pt1 = (np.random.rand(3, 10) * 256).astype(np.float64)
        pt2 = (np.random.rand(3, 10) * 256).astype(np.float64)
        F = (np.random.rand(3, 3) * 256).astype(np.float64)
        dist = cv.sampsonDistance(pt1, pt2, F)
        self.assertTrue(isinstance(dist, (float, np.floating)))
        self.assertGreaterEqual(dist, 0.0)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
