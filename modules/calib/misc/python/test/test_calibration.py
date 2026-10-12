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

    def test_fisheye_calibration_point_layout(self):
        camera_matrix = np.array([[600.0, 0.0, 320.0],
                                  [0.0, 600.0, 240.0],
                                  [0.0, 0.0, 1.0]])
        distortion = np.zeros((4, 1), dtype=np.float64)
        image_size = (640, 480)
        image_noise = 0.05
        tolerance = 1e-10
        max_iterations = 1
        points = np.array([
            [-0.4, -0.3, 0.0], [0.0, -0.3, 0.0], [0.4, -0.3, 0.0],
            [-0.4, 0.0, 0.0], [0.0, 0.0, 0.0], [0.4, 0.0, 0.0],
            [-0.4, 0.3, 0.0], [0.0, 0.3, 0.0], [0.4, 0.3, 0.0]])
        rotations = np.array([[0.15, -0.1, 0.08], [-0.1, 0.12, -0.06]])
        translations = np.array([[0.2, -0.15, 3.0], [-0.3, 0.1, 2.5]])
        object_rows = []
        image_rows = []
        for rotation, translation in zip(rotations, translations):
            objects = points.reshape(1, -1, 3).copy()
            projected, _ = cv.fisheye.projectPoints(
                objects, rotation, translation, camera_matrix, distortion)
            projected = projected.reshape(1, -1, 2)
            projected[0, 0, 0] += image_noise
            object_rows.append(objects)
            image_rows.append(projected)

        flags = (cv.CALIB_USE_INTRINSIC_GUESS |
                 cv.CALIB_RECOMPUTE_EXTRINSIC |
                 cv.CALIB_FIX_FOCAL_LENGTH |
                 cv.CALIB_FIX_PRINCIPAL_POINT |
                 cv.CALIB_FIX_SKEW |
                 cv.CALIB_FIX_K1 | cv.CALIB_FIX_K2 |
                 cv.CALIB_FIX_K3 | cv.CALIB_FIX_K4)
        criteria = (cv.TERM_CRITERIA_COUNT, max_iterations, 0.0)

        def calibrate(objects, images):
            return cv.fisheye.calibrate(
                objects, images, image_size, camera_matrix.copy(),
                distortion.copy(), flags=flags, criteria=criteria)

        reference = calibrate(object_rows, image_rows)
        self.assertGreater(reference[0], tolerance)
        for column_images in (True, False):
            for column_objects in (False, True):
                with self.subTest(column_images=column_images,
                                  column_objects=column_objects):
                    objects = [p.reshape(-1, 1, 3) if column_objects else p
                               for p in object_rows]
                    images = [p.reshape(-1, 1, 2) if column_images else p
                              for p in image_rows]
                    result = calibrate(objects, images)
                    for actual, expected in zip(result, reference):
                        np.testing.assert_allclose(
                            actual, expected, rtol=0.0, atol=tolerance)

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

    @staticmethod
    def _random_rt():
        # random rotation (axis-angle) + random translation
        axis = np.random.uniform(-1, 1, 3)
        axis /= np.linalg.norm(axis)
        angle = np.random.uniform(0.2, 1.0) * (1 if np.random.rand() > 0.5 else -1)
        rvec = (axis * angle).astype(np.float64)
        R, _ = cv.Rodrigues(rvec)
        t = np.random.uniform(-100, 100, 3).astype(np.float64)
        return R, t

    @staticmethod
    def _to_T(R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def test_calibrateHandEye(self):
        # Ground truth camera-to-gripper transform (what calibrateHandEye should recover)
        R_cam2gripper_gt, t_cam2gripper_gt = self._random_rt()
        X = self._to_T(R_cam2gripper_gt, t_cam2gripper_gt)

        # Arbitrary fixed target-to-base transform (the calibration board doesn't move)
        R_target2base, t_target2base = self._random_rt()
        Wt = self._to_T(R_target2base, t_target2base)

        n_poses = 10
        Rs_gripper2base, ts_gripper2base = [], []
        Rs_target2cam, ts_target2cam = [], []
        for _ in range(n_poses):
            R_g2b, t_g2b = self._random_rt()
            G = self._to_T(R_g2b, t_g2b)
            # Consistent synthetic data: Wt == G @ X @ T_target2cam for every pose
            T_target2cam = np.linalg.inv(X) @ np.linalg.inv(G) @ Wt

            Rs_gripper2base.append(R_g2b)
            ts_gripper2base.append(t_g2b)
            Rs_target2cam.append(T_target2cam[:3, :3])
            ts_target2cam.append(T_target2cam[:3, 3])

        R_cam2gripper, t_cam2gripper = cv.calibrateHandEye(
            Rs_gripper2base, ts_gripper2base, Rs_target2cam, ts_target2cam)

        self.assertLess(cv.norm(R_cam2gripper - R_cam2gripper_gt, cv.NORM_L1), 1e-3)
        self.assertLess(cv.norm(t_cam2gripper.flatten() - t_cam2gripper_gt, cv.NORM_L1), 1e-3)

    def test_calibrateRobotWorldHandEye(self):
        # Ground truth outputs (what calibrateRobotWorldHandEye should recover)
        R_base2world_gt, t_base2world_gt = self._random_rt()
        X = self._to_T(R_base2world_gt, t_base2world_gt)

        R_gripper2cam_gt, t_gripper2cam_gt = self._random_rt()
        Z = self._to_T(R_gripper2cam_gt, t_gripper2cam_gt)

        n_poses = 10
        Rs_world2cam, ts_world2cam = [], []
        Rs_base2gripper, ts_base2gripper = [], []
        for _ in range(n_poses):
            R_b2g, t_b2g = self._random_rt()
            B = self._to_T(R_b2g, t_b2g)
            # Consistent synthetic data: A_i @ X == Z @ B_i  =>  A_i = Z @ B_i @ inv(X)
            A = Z @ B @ np.linalg.inv(X)

            Rs_world2cam.append(A[:3, :3])
            # NOTE: unlike calibrateHandEye, calibrateRobotWorldHandEye's Shah/Li
            # solvers require translations as explicit (3,1) column vectors, not
            # flat (3,) arrays -- passing (3,) raises a gemm shape-assertion error.
            ts_world2cam.append(A[:3, 3].reshape(3, 1))
            Rs_base2gripper.append(R_b2g)
            ts_base2gripper.append(t_b2g.reshape(3, 1))

        R_base2world, t_base2world, R_gripper2cam, t_gripper2cam = cv.calibrateRobotWorldHandEye(
            Rs_world2cam, ts_world2cam, Rs_base2gripper, ts_base2gripper)

        self.assertLess(cv.norm(R_base2world - R_base2world_gt, cv.NORM_L1), 1e-3)
        self.assertLess(cv.norm(t_base2world.flatten() - t_base2world_gt, cv.NORM_L1), 1e-3)
        self.assertLess(cv.norm(R_gripper2cam - R_gripper2cam_gt, cv.NORM_L1), 1e-3)
        self.assertLess(cv.norm(t_gripper2cam.flatten() - t_gripper2cam_gt, cv.NORM_L1), 1e-3)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
