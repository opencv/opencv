#!/usr/bin/env python
# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np

from tests_common import NewOpenCVTests


class FlannBasedMatcher(NewOpenCVTests):
    def test_all_parameters_can_be_passed(self):
        img1 = self.get_sample("samples/data/right01.jpg")
        img2 = self.get_sample("samples/data/right02.jpg")

        orb = cv2.ORB.create()

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        FLANN_INDEX_KDTREE = 1
        index_param = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
        search_param = dict(checks=32, sorted=True, eps=0.5,
                            explore_all_trees=False)
        matcher = cv2.FlannBasedMatcher(index_param, search_param)
        matches = matcher.knnMatch(np.float32(des1), np.float32(des2), k=2)
        self.assertGreater(len(matches), 0)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
