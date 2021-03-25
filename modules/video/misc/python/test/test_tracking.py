#!/usr/bin/env python
import os
import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests, unittest

class tracking_test(NewOpenCVTests):

    def test_createTracker(self):
        t = cv.TrackerMIL_create()
        try:
            t = cv.TrackerGOTURN_create()
        except cv.error as e:
            pass  # may fail due to missing DL model files


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
