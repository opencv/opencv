#!/usr/bin/env python
# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class fisheye_test(NewOpenCVTests):

    def test_fisheye_projection_crosses_camera_plane(self):
        angles = np.radians([60.0, 89.0, 90.0, 91.0, 120.0])
        points = np.stack([np.sin(angles), np.zeros_like(angles),
                           np.cos(angles)], axis=1).reshape(1, -1, 3)
        points[0, 2, 2] = 0.0
        camera = np.eye(3)
        projected, jacobian = cv.fisheye.projectPoints(
            points, np.zeros(3), np.zeros(3), camera, np.zeros(4))
        expected = np.stack([angles, np.zeros_like(angles)], axis=1)
        np.testing.assert_allclose(projected.reshape(-1, 2), expected,
                                   rtol=0.0, atol=1e-12)
        self.assertTrue(np.isfinite(jacobian).all())

    def test_fisheye_projection_jacobian_crosses_camera_plane(self):
        points = np.array([[[1.0, 0.3, 0.0], [0.8, -0.2, -0.1]]])
        camera = np.array([[600.0, 0.0, 320.0],
                           [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]])
        parameters = np.r_[600.0, 610.0, 320.0, 240.0,
                           -0.02, 0.003, -0.0002, 0.00001,
                           np.zeros(6), 0.01]

        def project(values):
            matrix = camera.copy()
            matrix[0, 0], matrix[1, 1] = values[:2]
            matrix[0, 2], matrix[1, 2] = values[2:4]
            return cv.fisheye.projectPoints(
                points, values[8:11], values[11:14], matrix, values[4:8],
                alpha=values[14])

        _, jacobian = project(parameters)
        difference_step = 1e-6
        for column in range(len(parameters)):
            plus = parameters.copy()
            minus = parameters.copy()
            plus[column] += difference_step
            minus[column] -= difference_step
            numerical = ((project(plus)[0] - project(minus)[0]) /
                         (2.0 * difference_step)).ravel()
            np.testing.assert_allclose(jacobian[:, column], numerical,
                                       rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
