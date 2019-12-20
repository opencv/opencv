#!/usr/bin/env python

'''
MSER detector test
'''
# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class mser_test(NewOpenCVTests):
    def test_mser(self):

        img = self.get_sample('cv/mser/puzzle.png', 0)
        smallImg = [
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255],
         [255, 255, 255, 255, 255,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255],
         [255, 255, 255, 255, 255,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255],
         [255, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255],
         [255, 255, 255, 255, 255, 255,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
        ]
        thresharr = [ 0, 70, 120, 180, 255 ]
        kDelta = 5
        mserExtractor = cv.MSER_create()
        mserExtractor.setDelta(kDelta)
        np.random.seed(10)

        for _i in range(100):

            use_big_image = int(np.random.rand(1,1)*7) != 0
            invert = int(np.random.rand(1,1)*2) != 0
            binarize = int(np.random.rand(1,1)*5) != 0 if use_big_image else False
            blur = int(np.random.rand(1,1)*2) != 0
            thresh = thresharr[int(np.random.rand(1,1)*5)]
            src0 = img if use_big_image else np.array(smallImg).astype('uint8')
            src = src0.copy()

            kMinArea = 256 if use_big_image else 10
            kMaxArea = int(src.shape[0]*src.shape[1]/4)

            mserExtractor.setMinArea(kMinArea)
            mserExtractor.setMaxArea(kMaxArea)
            if invert:
                cv.bitwise_not(src, src)
            if binarize:
                _, src = cv.threshold(src, thresh, 255, cv.THRESH_BINARY)
            if blur:
                src = cv.GaussianBlur(src, (5, 5), 1.5, 1.5)
            minRegs = 7 if use_big_image else 2
            maxRegs = 1000 if use_big_image else 20
            if binarize and (thresh == 0 or thresh == 255):
                minRegs = maxRegs = 0
            msers, boxes = mserExtractor.detectRegions(src)
            nmsers = len(msers)
            self.assertEqual(nmsers, len(boxes))
            self.assertLessEqual(minRegs, nmsers)
            self.assertGreaterEqual(maxRegs, nmsers)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
