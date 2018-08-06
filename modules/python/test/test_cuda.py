#!/usr/bin/env python

'''
CUDA-accelerated Computer Vision functions
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class cuda_test(NewOpenCVTests):
    def setUp(self):
        if not cv.cuda.getCudaEnabledDeviceCount():
            self.skipTest("No CUDA-capable device is detected")

    def test_cuda_upload_download(self):
        npMat = (np.random.random((200, 200, 3)) * 255).astype(np.uint8)
        gpuMat = cv.cuda_GpuMat()
        gpuMat.upload(npMat)

        self.assertTrue(np.allclose(gpuMat.download(), npMat))

    def test_cuda_imgproc_cvtColor(self):
        npMat = (np.random.random((200, 200, 3)) * 255).astype(np.uint8)
        gpuMat = cv.cuda_GpuMat()
        gpuMat.upload(npMat)
        gpuMat2 = cv.cuda.cvtColor(gpuMat, cv.COLOR_BGR2HSV)

        self.assertTrue(np.allclose(gpuMat2.download(), cv.cvtColor(npMat, cv.COLOR_BGR2HSV)))

    def test_cuda_filter_laplacian(self):
        npMat = (np.random.random((200, 200)) * 255).astype(np.uint16)
        gpuMat = cv.cuda_GpuMat()
        gpuMat.upload(npMat)
        gpuMat = cv.cuda.createLaplacianFilter(cv.CV_16UC1, -1, ksize=3).apply(gpuMat)

        self.assertTrue(np.allclose(gpuMat.download(), cv.Laplacian(npMat, cv.CV_16UC1, ksize=3)))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
