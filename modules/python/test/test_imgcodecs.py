#!/usr/bin/env python

'''
Test for imread2
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import os

from tests_common import NewOpenCVTests

class imgcodecs_test(NewOpenCVTests):
    def test_imread2(self):
        path = self.find_file('python/images/baboon.jpg', [os.environ.get('OPENCV_TEST_DATA_PATH')])
        ret, img = cv.imread2(path, maxSize = (512, 512))
        self.assertEqual(0, ret)
        self.assertTrue(img.size)

        ret, img = cv.imread2(path, maxSize = (511, 512))
        self.assertEqual(3, ret)
        self.assertIsNone(img)

        ret, img = cv.imread2(path, maxPixels = 512*512)
        self.assertEqual(0, ret)
        self.assertIsNotNone(img)

        ret, img = cv.imread2(path, maxPixels = 512*512 - 1)
        self.assertEqual(3, ret)
        self.assertIsNone(img)

        ret, img = cv.imread2('nofile.png')
        self.assertEqual(1, ret)
        self.assertIsNone(img)

        ret, img = cv.imread2(path, flags = 1, maxPixels = 512*512, maxSize = (512,512))
        self.assertEqual(0, ret)
        self.assertIsNotNone(img)

        ret, img2 = cv.imread2(path, flags = 1, maxPixels = 512*512, maxSize = (512,512), scaleDenom = 2)
        self.assertEqual(0, ret)
        self.assertIsNotNone(img2)
        self.assertEqual(img2.size, img.size / 4)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
