#!/usr/bin/env python

'''
Test cv::EMD wrapper and ensure proper pointers binding
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

from tests_common import NewOpenCVTests

class emd_test(NewOpenCVTests):

    def test_emd(self):
        success_error_level = 1e-6
        M = 10000
        emd0 = 2460./210
        cost = np.array(
        [
            [16, 16, 13, 22, 17],
            [14, 14, 13, 19, 15],
            [19, 19, 20, 23,  M],
            [M ,  0,  M,  0,  0]
        ], np.float32)
        w1 = np.array([ 50, 60, 50, 50 ], np.float32)
        w2 = np.array([ 30, 20, 70, 30, 60 ], np.float32)

        emd, lowerBound, flow = cv2.EMD(w1, w2, -1, cost)
        self.assertLess(np.fabs(emd - emd0), success_error_level * emd0), \
            "The computed distance is %.2f, while it should be %.2f\n" % (emd, emd0)

if __name__ == '__main__':
    import unittest
    unittest.main()
