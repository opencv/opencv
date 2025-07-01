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
        img = ic.at(0)
        self.assertEqual(cv.norm(animation.frames[0], img, cv.NORM_INF), 0.0)
        img = ic.at(1)
        self.assertEqual(cv.norm(animation.frames[1], img, cv.NORM_INF), 0.0)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
