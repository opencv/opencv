#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import os, numpy as np

import cv2 as cv

from tests_common import NewOpenCVTests

class aruco_test(NewOpenCVTests):

    def test_aruco_detect_markers(self):
        aruco_params = cv.aruco.DetectorParameters()
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
        id = 2
        marker_size = 100
        offset = 10
        img_marker = cv.aruco.generateImageMarker(aruco_dict, id, marker_size, aruco_params.markerBorderBits)
        img_marker = np.pad(img_marker, pad_width=offset, mode='constant', constant_values=255)
        gold_corners = np.array([[offset, offset],[marker_size+offset-1.0,offset],
                                 [marker_size+offset-1.0,marker_size+offset-1.0],
                                 [offset, marker_size+offset-1.0]], dtype=np.float32)
        expected_corners, expected_ids, expected_rejected = cv.aruco.detectMarkers(img_marker, aruco_dict,
                                                                                   parameters=aruco_params)

        self.assertEqual(1, len(expected_ids))
        self.assertEqual(id, expected_ids[0])
        for i in range(0, len(expected_corners)):
            np.testing.assert_array_equal(gold_corners, expected_corners[i].reshape(4, 2))

    def test_drawCharucoDiamond(self):
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        img = cv.aruco.drawCharucoDiamond(aruco_dict, np.array([0, 1, 2, 3]), 100, 80)
        self.assertTrue(img is not None)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
