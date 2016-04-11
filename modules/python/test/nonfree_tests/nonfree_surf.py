#!/usr/bin/env python
# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import cv2.cv as cv

from tests_common import OpenCVTests

class NonFreeFunctionTests(OpenCVTests):

    def test_ExtractSURF(self):
        img = self.get_sample("samples/c/lena.jpg", 0)
        w,h = cv.GetSize(img)
        for hessthresh in [ 300,400,500]:
            for dsize in [0,1]:
                for layers in [1,3,10]:
                    kp,desc = cv.ExtractSURF(img, None, cv.CreateMemStorage(), (dsize, hessthresh, 3, layers))
                    self.assertTrue(len(kp) == len(desc))
                    for d in desc:
                        self.assertTrue(len(d) == {0:64, 1:128}[dsize])
                    for pt,laplacian,size,dir,hessian in kp:
                        self.assertTrue((0 <= pt[0]) and (pt[0] <= w))
                        self.assertTrue((0 <= pt[1]) and (pt[1] <= h))
                        self.assertTrue(laplacian in [-1, 0, 1])
                        self.assertTrue((0 <= dir) and (dir <= 360))
                        self.assertTrue(hessian >= hessthresh)
