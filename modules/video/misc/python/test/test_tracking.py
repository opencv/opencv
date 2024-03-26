#!/usr/bin/env python
import os
import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests, unittest

class tracking_test(NewOpenCVTests):

    def test_createTracker(self):
        t = cv.TrackerMIL_create()


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
