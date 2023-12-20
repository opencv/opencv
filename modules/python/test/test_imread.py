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
    def test_imread(self):

        img = np.zeros((1000, 1000), dtype=np.uint8)
        subImg = img[64:576, 128:640]
        cv.imread(self.extraTestDataPath + 'cv/shared/lena.png'), subImg)
        subImg = img.rowRange(64, 576).colRange(128, 640);
        ori = cv.imread(self.extraTestDataPath + 'cv/shared/lena.png'))
        self.assertEqual(cv.norm(ori, subImg, cv.NORM_INF), 0)
        self.assertEqual(cv.countNonZero(img[0:1000, 0:128]), 0)
        self.assertEqual(cv.countNonZero(img[0:1000, 640:1000]), 0)
        self.assertEqual(cv.countNonZero(img[0:64, 128:640]), 0)
        self.assertEqual(cv.countNonZero(img[576:1000, 128:640]), 0)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
