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

        # Load animation using imreadanimation
        animation = cv.Animation()
        success, animation = cv.imreadanimation(path)
        self.assertTrue(success)
        self.assertEqual(len(animation.frames), 2)

        # Test ImageCollection
        ic = cv.ImageCollection()
        self.assertEqual(ic.size(), 0)
        self.assertEqual(ic.getWidth(), 0)
        self.assertEqual(ic.getHeight(), 0)
        self.assertEqual(ic.getType(), 0)

        ic.init(path, cv.IMREAD_UNCHANGED)
        self.assertEqual(ic.size(), 2)

        # Check width, height and type
        self.assertEqual(ic.getWidth(), animation.frames[0].shape[1])
        self.assertEqual(ic.getHeight(), animation.frames[0].shape[0])
        self.assertEqual(ic.getType(), animation.frames[0].dtype)

        # Compare frames pixel-wise
        for i in range(ic.size()):
            self.assertEqual(
                cv.norm(animation.frames[i], ic.at(i), cv.NORM_INF),
                0.0,
                msg="Mismatch in frame {}".format(i)
            )

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
