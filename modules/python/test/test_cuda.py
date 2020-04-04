#!/usr/bin/env python

'''
CUDA-accelerated Computer Vision functions
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import os

from tests_common import NewOpenCVTests, unittest

class cuda_test(NewOpenCVTests):
    def setUp(self):
        super(cuda_test, self).setUp()
        if not cv.cuda.getCudaEnabledDeviceCount():
            self.skipTest("No CUDA-capable device is detected")

    def test_cuda_upload_download(self):
        npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
        cuMat = cv.cuda_GpuMat()
        cuMat.upload(npMat)

        self.assertTrue(np.allclose(cuMat.download(), npMat))

    def test_cuda_interop(self):
        npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
        cuMat = cv.cuda_GpuMat()
        cuMat.upload(npMat)
        self.assertTrue(cuMat.cudaPtr() != 0)
        stream = cv.cuda_Stream()
        self.assertTrue(stream.cudaPtr() != 0)

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
