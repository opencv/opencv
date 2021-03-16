#!/usr/bin/env python

'''
Test for copyto with mask
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import sys

from tests_common import NewOpenCVTests

class copytomask_test(NewOpenCVTests):
    def test_copytomask(self):

        img = self.get_sample('python/images/baboon.png', cv.IMREAD_COLOR)
        eps = 0.

        #Create mask using inRange
        valeurBGRinf = np.array([0,0,100])
        valeurBGRSup = np.array([70, 70,255])
        maskRed = cv.inRange(img, valeurBGRinf, valeurBGRSup)
        #New binding
        dstcv = np.ndarray(np.array((2, 2, 1))*img.shape, dtype=img.dtype)
        dstcv.fill(255)
        cv.copyTo(img, maskRed, dstcv[:img.shape[0],:img.shape[1],:])
        #using numpy
        dstnp = np.ndarray(np.array((2, 2, 1))*img.shape, dtype=img.dtype)
        dstnp.fill(255)
        mask2=maskRed.astype(bool)
        _, mask_b = np.broadcast_arrays(img, mask2[..., None])
        np.copyto(dstnp[:img.shape[0],:img.shape[1],:], img, where=mask_b)
        self.assertEqual(cv.norm(dstnp ,dstcv), eps)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
