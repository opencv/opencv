#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import os, numpy
import math
import unittest
import cv2 as cv
import numpy as np

from tests_common import NewOpenCVTests

def lookAtMatrixCal(position, lookat, upVector):
    tmp = position - lookat
    norm = numpy.linalg.norm(tmp)
    w = tmp / norm
    tmp = numpy.cross(upVector, w)
    norm = numpy.linalg.norm(tmp)
    u = tmp / norm
    v = numpy.cross(w, u)
    res = numpy.array([
        [u[0], u[1], u[2],   0],
        [v[0], v[1], v[2],   0],
        [w[0], w[1], w[2],   0],
        [0,    0,    0,   1.0]
    ], dtype=numpy.float32)
    translate = numpy.array([
        [1.0,   0,   0, -position[0]],
        [0, 1.0,   0, -position[1]],
        [0,   0, 1.0, -position[2]],
        [0,   0,   0,          1.0]
    ], dtype=numpy.float32)
    res = numpy.matmul(res, translate)
    return res

class raster_test(NewOpenCVTests):

    def prepareData(self):
        self.vertices = numpy.array([
            [ 2.0,  0.0, -2.0],
            [ 0.0, -6.0, -2.0],
            [-2.0,  0.0, -2.0],
            [ 3.5, -1.0, -5.0],
            [ 2.5, -2.5, -5.0],
            [-1.0,  1.0, -5.0],
            [-6.5, -1.0, -3.0],
            [-2.5, -2.0, -3.0],
            [ 1.0,  1.0, -5.0],
        ], dtype=numpy.float32)

        self.indices = numpy.array([ [0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.intc)

        col1 = [ 185.0,  238.0,  217.0 ]
        col2 = [ 238.0,  217.0,  185.0 ]
        col3 = [ 238.0,   10.0,  150.0 ]

        self.colors = numpy.array([
            col1, col2, col3,
            col2, col3, col1,
            col3, col1, col2,
        ], dtype=numpy.float32)

        self.colors = self.colors / 255.0

        self.zNear = 0.1
        self.zFar = 50.0
        self.fovy = 45.0 * math.pi / 180.0

        position = numpy.array([0.0, 0.0, 5.0], dtype=numpy.float32)
        lookat   = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float32)
        upVector = numpy.array([0.0, 1.0, 0.0], dtype=numpy.float32)
        self.cameraPose = lookAtMatrixCal(position, lookat, upVector)

        self.depth_buf = numpy.ones((240, 320), dtype=numpy.float32) * self.zFar
        self.color_buf = numpy.zeros((240, 320, 3), dtype=numpy.float32)

        self.settings = cv.TriangleRasterizeSettings().setShadingType(cv.RASTERIZE_SHADING_SHADED)
        self.settings = self.settings.setCullingMode(cv.RASTERIZE_CULLING_NONE)

    def compareResults(self, needRgb, needDepth):
        if needDepth:
            depth = self.get_sample('rendering/depth_image_Clipping_320x240_CullNone.png', cv.IMREAD_ANYDEPTH).astype(numpy.float32)
            depthFactor = 1000.0
            diff = depth/depthFactor - self.depth_buf
            norm = numpy.linalg.norm(diff)
            self.assertLessEqual(norm, 356.0)

        if needRgb:
            rgb = self.get_sample('rendering/example_image_Clipping_320x240_CullNone_Shaded.png', cv.IMREAD_ANYCOLOR)
            diff = rgb/255.0 - self.color_buf
            norm = numpy.linalg.norm(diff)
            self.assertLessEqual(norm, 11.62)

    def test_rasterizeBoth(self):
        self.prepareData()
        self.color_buf, self.depth_buf = cv.triangleRasterize(self.vertices, self.indices, self.colors, self.color_buf, self.depth_buf,
                                                              self.cameraPose, self.fovy, self.zNear, self.zFar, self.settings)
        self.compareResults(needRgb=True, needDepth=True)

    def test_rasterizeDepth(self):
        self.prepareData()
        self.depth_buf = cv.triangleRasterizeDepth(self.vertices, self.indices, self.depth_buf,
                                                   self.cameraPose, self.fovy, self.zNear, self.zFar, self.settings)
        self.compareResults(needRgb=False, needDepth=True)

    def test_rasterizeColor(self):
        self.prepareData()
        self.color_buf = cv.triangleRasterizeColor(self.vertices, self.indices, self.colors, self.color_buf,
                                                   self.cameraPose, self.fovy, self.zNear, self.zFar, self.settings)
        self.compareResults(needRgb=True, needDepth=False)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
