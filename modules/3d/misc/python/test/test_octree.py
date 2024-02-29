#!/usr/bin/env python

import os
import numpy as np
import math
import unittest
import cv2 as cv

from tests_common import NewOpenCVTests

class octree_test(NewOpenCVTests):
    def test_octree_basic_test(self):
        pointCloudSize = 1000
        resolution = 0.0001
        scale = 1 << 20

        pointCloud = np.random.randint(-scale, scale, size=(pointCloudSize, 3)) * (10.0 / scale)
        pointCloud = pointCloud.astype(np.float32)

        octree = cv.Octree_createWithResolution(resolution, pointCloud)

        restPoint = np.random.randint(-scale, scale, size=(1, 3)) * (10.0 / scale)
        restPoint = [restPoint[0, 0], restPoint[0, 1], restPoint[0, 2]]

        self.assertTrue(octree.isPointInBound(restPoint))
        self.assertFalse(octree.deletePoint(restPoint))
        self.assertTrue(octree.insertPoint(restPoint))
        self.assertTrue(octree.deletePoint(restPoint))

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
