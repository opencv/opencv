#!/usr/bin/env python

'''
Texture flow direction estimation.

Sample shows how cv.cornerEigenValsAndVecs function can be used
to estimate image texture flow direction.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import sys

from tests_common import NewOpenCVTests


class texture_flow_test(NewOpenCVTests):

    def test_texture_flow(self):

        img = self.get_sample('samples/data/chessboard.png')

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        eigen = cv.cornerEigenValsAndVecs(gray, 5, 3)
        eigen = eigen.reshape(h, w, 3, 2)  # [[e1, e2], v1, v2]
        flow = eigen[:,:,2]

        d = 300
        eps = d / 30

        points =  np.dstack( np.mgrid[d/2:w:d, d/2:h:d] ).reshape(-1, 2)

        textureVectors = []
        for x, y in np.int32(points):
            textureVectors.append(np.int32(flow[y, x]*d))

        for i in range(len(textureVectors)):
            self.assertTrue(cv.norm(textureVectors[i], cv.NORM_L2) < eps
            or abs(cv.norm(textureVectors[i], cv.NORM_L2) - d) < eps)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
