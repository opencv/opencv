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

    def test_cuda_upload_download_stream(self):
        stream = cv.cuda_Stream()
        npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
        cuMat = cv.cuda_GpuMat(128,128, cv.CV_8UC3)
        cuMat.upload(npMat, stream)
        npMat2 = cuMat.download(stream=stream)
        stream.waitForCompletion()
        self.assertTrue(np.allclose(npMat2, npMat))

    def test_cuda_interop(self):
        npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
        cuMat = cv.cuda_GpuMat()
        cuMat.upload(npMat)
        self.assertTrue(cuMat.cudaPtr() != 0)
        stream = cv.cuda_Stream()
        self.assertTrue(stream.cudaPtr() != 0)
        asyncstream = cv.cuda_Stream(1)  # cudaStreamNonBlocking
        self.assertTrue(asyncstream.cudaPtr() != 0)

    def test_cuda_buffer_pool(self):
        cv.cuda.setBufferPoolUsage(True)
        cv.cuda.setBufferPoolConfig(cv.cuda.getDevice(), 1024 * 1024 * 64, 2)
        stream_a = cv.cuda.Stream()
        pool_a = cv.cuda.BufferPool(stream_a)
        cuMat = pool_a.getBuffer(1024, 1024, cv.CV_8UC3)
        cv.cuda.setBufferPoolUsage(False)
        self.assertEqual(cuMat.size(), (1024, 1024))
        self.assertEqual(cuMat.type(), cv.CV_8UC3)

    def test_cuda_release(self):
        npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
        cuMat = cv.cuda_GpuMat()
        cuMat.upload(npMat)
        cuMat.release()
        self.assertTrue(cuMat.cudaPtr() == 0)
        self.assertTrue(cuMat.step == 0)
        self.assertTrue(cuMat.size() == (0, 0))

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
