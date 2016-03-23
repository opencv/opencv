#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np

from tests_common import NewOpenCVTests

class TestGoodFeaturesToTrack_test(NewOpenCVTests):
    def test_goodFeaturesToTrack(self):
        arr = self.get_sample('samples/data/lena.jpg', 0)
        original = arr.copy(True)
        threshes = [ x / 100. for x in range(1,10) ]
        numPoints = 20000

        results = dict([(t, cv2.goodFeaturesToTrack(arr, numPoints, t, 2, useHarrisDetector=True)) for t in threshes])
        # Check that GoodFeaturesToTrack has not modified input image
        self.assertTrue(arr.tostring() == original.tostring())
        # Check for repeatability
        for i in range(1):
            results2 = dict([(t, cv2.goodFeaturesToTrack(arr, numPoints, t, 2, useHarrisDetector=True)) for t in threshes])
            for t in threshes:
                self.assertTrue(len(results2[t]) == len(results[t]))
                for i in range(len(results[t])):
                    self.assertTrue(cv2.norm(results[t][i][0] - results2[t][i][0]) == 0)

        for t0,t1 in zip(threshes, threshes[1:]):
             r0 = results[t0]
             r1 = results[t1]
             # Increasing thresh should make result list shorter
             self.assertTrue(len(r0) > len(r1))
             # Increasing thresh should monly truncate result list
             for i in range(len(r1)):
                self.assertTrue(cv2.norm(r1[i][0] - r0[i][0])==0)