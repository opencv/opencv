#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class Features2D_Tests(NewOpenCVTests):

    def test_issue_13406(self):
        self.assertEqual(True, hasattr(cv, 'drawKeypoints'))
        self.assertEqual(True, hasattr(cv, 'DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS'))
        self.assertEqual(True, hasattr(cv, 'DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS'))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
