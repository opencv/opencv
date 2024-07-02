#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import os, numpy
import math
import unittest
import cv2 as cv

from tests_common import NewOpenCVTests

class raster_test(NewOpenCVTests):

    def test_loadCloud(self):
        fin = self.find_file("pointcloudio/orig.obj")
        points, normals, colors = cv.loadPointCloud(fin)

        if points.shape != (8, 1, 3):
            self.fail('point array should be 8x1x3')
        if normals.shape != (6, 1, 3):
            self.fail('normals array should be 6x1x3')
        if colors.shape != (8, 1, 3):
            self.fail('colors array should be 8x1x3')

    def test_loadMesh(self):
        fin = self.find_file("pointcloudio/orig.obj")
        points, indices, normals, colors, texCoords = cv.loadMesh(fin)
        goodShapes = [(1, 18, 3), (18, 1, 3)]
        errorMsg = "%s array should be 18x1x3 or 1x18x3"
        for a, s in [(points, 'points'), (normals, 'normals'), (colors, 'colors')]:
            if a.shape not in goodShapes:
                self.fail(errorMsg % s)

        if texCoords.shape not in [(1, 18, 2), (18, 1, 2), (18, 2)]:
            self.fail('texture coordinates array should be 1x18x2 or 18x1x2')
        if isinstance(indices, numpy.ndarray):
            if indices.shape not in [(1, 6, 3), (6, 1, 3)]:
                self.fail('indices array should be 1x6x3 or 6x1x3')
        elif isinstance(indices, list) or isinstance(indices, tuple):
            for i in indices:
                if len(indices) != 6 or i.shape != (1, 3):
                    self.fail('indices array should be 1x6x3 or 6x1x3')

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
