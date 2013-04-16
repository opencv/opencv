#!/usr/bin/env python

import cv2.cv as cv
import unittest

class TestGoodFeaturesToTrack(unittest.TestCase):
    def test(self):
        arr = cv.LoadImage("../samples/c/lena.jpg", 0)
        original = cv.CloneImage(arr)
        size = cv.GetSize(arr)
        eig_image = cv.CreateImage(size, cv.IPL_DEPTH_32F, 1)
        temp_image = cv.CreateImage(size, cv.IPL_DEPTH_32F, 1)
        threshes = [ x / 100. for x in range(1,10) ]

        results = dict([(t, cv.GoodFeaturesToTrack(arr, eig_image, temp_image, 20000, t, 2, useHarris = 1)) for t in threshes])

        # Check that GoodFeaturesToTrack has not modified input image
        self.assert_(arr.tostring() == original.tostring())

        # Check for repeatability
        for i in range(10):
            results2 = dict([(t, cv.GoodFeaturesToTrack(arr, eig_image, temp_image, 20000, t, 2, useHarris = 1)) for t in threshes])
            self.assert_(results == results2)

        for t0,t1 in zip(threshes, threshes[1:]):
             r0 = results[t0]
             r1 = results[t1]

             # Increasing thresh should make result list shorter
             self.assert_(len(r0) > len(r1))

             # Increasing thresh should monly truncate result list
             self.assert_(r0[:len(r1)] == r1)

if __name__ == '__main__':
    unittest.main()
