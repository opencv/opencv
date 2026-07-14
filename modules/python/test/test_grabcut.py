#!/usr/bin/env python
'''
===============================================================================
Interactive Image Segmentation using GrabCut algorithm.
===============================================================================
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import sys

from tests_common import NewOpenCVTests

class grabcut_test(NewOpenCVTests):

    def verify(self, mask, exp):

        maxDiffRatio = 0.02
        expArea = np.count_nonzero(exp)
        nonIntersectArea = np.count_nonzero(mask != exp)
        curRatio = float(nonIntersectArea) / expArea
        return curRatio < maxDiffRatio

    def scaleMask(self, mask):

        return np.where((mask==cv.GC_FGD) + (mask==cv.GC_PR_FGD),255,0).astype('uint8')

    def test_grabcut(self):

        img = self.get_sample('cv/shared/airplane.png')
        mask_prob = self.get_sample("cv/grabcut/mask_probpy.png", 0)
        exp_mask1 = self.get_sample("cv/grabcut/exp_mask1py.png", 0)
        exp_mask2 = self.get_sample("cv/grabcut/exp_mask2py.png", 0)

        if img is None:
            self.assertTrue(False, 'Missing test data')

        rect = (24, 126, 459, 168)
        mask = np.zeros(img.shape[:2], dtype = np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        cv.grabCut(img, mask, rect, bgdModel, fgdModel, 0, cv.GC_INIT_WITH_RECT)
        cv.grabCut(img, mask, rect, bgdModel, fgdModel, 2, cv.GC_EVAL)

        if mask_prob is None:
            mask_prob = mask.copy()
            cv.imwrite(self.extraTestDataPath + '/cv/grabcut/mask_probpy.png', mask_prob)
        if exp_mask1 is None:
            exp_mask1 = self.scaleMask(mask)
            cv.imwrite(self.extraTestDataPath + '/cv/grabcut/exp_mask1py.png', exp_mask1)

        self.assertEqual(self.verify(self.scaleMask(mask), exp_mask1), True)

        mask = mask_prob
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        cv.grabCut(img, mask, rect, bgdModel, fgdModel, 0, cv.GC_INIT_WITH_MASK)
        cv.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv.GC_EVAL)

        if exp_mask2 is None:
            exp_mask2 = self.scaleMask(mask)
            cv.imwrite(self.extraTestDataPath + '/cv/grabcut/exp_mask2py.png', exp_mask2)

        self.assertEqual(self.verify(self.scaleMask(mask), exp_mask2), True)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
