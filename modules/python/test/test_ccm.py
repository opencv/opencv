#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import unittest
import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class photo_test(NewOpenCVTests):

    def setUp(self):
        super(photo_test, self).setUp()
        self.image_cache = {}

    def test_ccm(self):
        # Create sample color checker data (24 Macbeth color patches)
        s = np.array([
            [214.11, 98.67, 37.97],
            [231.94, 153.1, 85.27],
            [204.08, 143.71, 78.46],
            [190.58, 122.99, 30.84],
            [230.93, 148.46, 100.84],
            [228.64, 206.97, 97.5],
            [229.09, 137.07, 55.29],
            [189.21, 111.22, 92.66],
            [223.5, 96.42, 75.45],
            [201.82, 69.71, 50.9],
            [240.52, 196.47, 59.3],
            [235.73, 172.13, 54.],
            [131.6, 75.04, 68.86],
            [189.04, 170.43, 42.05],
            [222.23, 74., 71.95],
            [241.01, 199.1, 61.15],
            [224.99, 101.4, 100.24],
            [174.58, 152.63, 91.52],
            [248.06, 227.69, 140.5],
            [241.15, 201.38, 115.58],
            [236.49, 175.87, 88.86],
            [212.19, 133.49, 54.79],
            [181.17, 102.94, 36.18],
            [115.1, 53.77, 15.23]
        ], dtype=np.float64) / 255.0

        # Create model and compute CCM
        model = cv.ccm_ColorCorrectionModel(s, cv.ccm_COLORCHECKER_MACBETH)
        colorCorrectionMat = model.compute()

        # Test with a sample RGBL value
        src_rgbl = np.array([0.68078957, 0.12382801, 0.01514889], dtype=np.float64)
        dst = model.infer(src_rgbl)
        
        # Verify the result is a valid color
        self.assertTrue(np.all(dst >= 0) and np.all(dst <= 1))
        self.assertEqual(dst.shape, (3,))

        # Test with multiple samples
        src_rgbls = np.array([
            [0.68078957, 0.12382801, 0.01514889],
            [0.81177942, 0.32550452, 0.089818],
            [0.61259378, 0.2831933, 0.07478902]
        ], dtype=np.float64)
        dsts = model.infer(src_rgbls)
        
        # Verify the results are valid colors
        self.assertTrue(np.all(dsts >= 0) and np.all(dsts <= 1))
        self.assertEqual(dsts.shape, (3, 3))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap() 