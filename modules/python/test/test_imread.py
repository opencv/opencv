#!/usr/bin/env python

'''
Test for imread
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import numpy as np
import sys

from tests_common import NewOpenCVTests

class imread_test(NewOpenCVTests):
    def test_imread_to_buffer(self):
        path = self.extraTestDataPath + '/cv/shared/lena.png'
        ref = cv.imread(path)

        img = np.zeros_like(ref)
        cv.imread(path, img)
        self.assertEqual(cv.norm(ref, img, cv.NORM_INF), 0.0)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
