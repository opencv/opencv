#!/usr/bin/env python

'''
ECC multiscale alignment test
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import inspect
import math

from tests_common import NewOpenCVTests

class eccms_test(NewOpenCVTests):
    def test_eccms(self):
        expected_res = np.array([
            [1.0225, 0.0606, -28.6452],
            [-0.0475, 1.0314, 11.819],
            [8.21e-06, -3.65e-07, 1.0]
        ], dtype=np.float32)

        largeGray0 = self.get_sample('cv/shared/halmosh0.jpg', cv.IMREAD_GRAYSCALE)
        largeGray1 = self.get_sample('cv/shared/halmosh2.jpg', cv.IMREAD_GRAYSCALE)
        roiMask0 = self.get_sample('cv/shared/halmosh0mask.png', cv.IMREAD_GRAYSCALE)
        roiMask1 = self.get_sample('cv/shared/halmosh2mask.png', cv.IMREAD_GRAYSCALE)

        if largeGray0 is None or largeGray1 is None or roiMask0 is None or roiMask1 is None:
            self.assertEqual(0, 1, 'Missing test data')

        found = np.eye(3, 3, dtype=np.float32)
        n_iters = 23
        termination_eps = 1e-6
        params = cv.ECCParameters()
        params.criteria = (cv.TERM_CRITERIA_COUNT + cv.TERM_CRITERIA_EPS, n_iters, termination_eps)
        params.motionType = cv.MOTION_HOMOGRAPHY
        params.nlevels = 5
        params.itersPerLevel = [5, 10, 300, 300, 1000]

        _, found = cv.findTransformECCMultiScale(largeGray0,largeGray1, found, params, roiMask0, roiMask1)

        self.assertLess(cv.norm(found - expected_res, cv.NORM_L1), 0.1)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
