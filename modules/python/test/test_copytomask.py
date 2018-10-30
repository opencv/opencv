#!/usr/bin/env python

'''
Test for copyto with mask
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import numpy as np
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
        dstcv  = cv.copyTo(img,maskRed)
        #using numpy
        mask2=maskRed.astype(bool)
        _, mask_b = np.broadcast_arrays(img, mask2[..., None])
        dstnp  = np.ma.masked_array(img, np.logical_not(mask_b))
        dstnp =np.ma.filled(dstnp,[0])
        self.assertEqual(cv.norm(dstnp ,dstcv), eps)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
