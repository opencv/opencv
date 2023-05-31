#!/usr/bin/env python

'''
Watershed segmentation test
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class watershed_test(NewOpenCVTests):
    def test_watershed(self):

        img = self.get_sample('cv/inpaint/orig.png')
        markers = self.get_sample('cv/watershed/wshed_exp.png', 0)
        refSegments = self.get_sample('cv/watershed/wshed_segments.png')

        if img is None or markers is None:
            self.assertEqual(0, 1, 'Missing test data')

        colors = np.int32( list(np.ndindex(3, 3, 3)) ) * 122
        cv.watershed(img, np.int32(markers))
        segments = colors[np.maximum(markers, 0)]

        if refSegments is None:
            refSegments = segments.copy()
            cv.imwrite(self.extraTestDataPath + '/cv/watershed/wshed_segments.png', refSegments)

        self.assertLess(cv.norm(segments - refSegments, cv.NORM_L1) / 255.0, 50)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
