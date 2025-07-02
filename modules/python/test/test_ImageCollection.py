#!/usr/bin/env python

'''
Test for ImageCollection
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import numpy as np
import sys

from tests_common import NewOpenCVTests

class imread_test(NewOpenCVTests):
    def test_ImageCollection(self):
        path = self.extraTestDataPath + '/highgui/readwrite/033.png'
        animation = cv.Animation()
        success, animation = cv.imreadanimation(path)
        self.assertTrue(success)

        ic = cv.ImageCollection(path, cv.IMREAD_UNCHANGED)
        self.assertEqual(2, ic.size32())
        self.assertEqual(128, ic.width())
        self.assertEqual(64, height())
        self.assertEqual(cv.norm(animation.frames[0], ic.at(0), cv.NORM_INF), 0.0)
        self.assertEqual(cv.norm(animation.frames[1], ic.at(1), cv.NORM_INF), 0.0)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
