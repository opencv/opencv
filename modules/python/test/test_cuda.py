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
        super(cuda_test, self).setUp()
        if not cv.cuda.getCudaEnabledDeviceCount():
            self.skipTest("No CUDA-capable device is detected")

    def test_cuda_upload_download(self):
        npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
        cuMat = cv.cuda_GpuMat()
        cuMat.upload(npMat)

        self.assertTrue(np.allclose(cuMat.download(), npMat))

    def test_cudaarithm_arithmetic(self):
        npMat1 = np.random.random((128, 128, 3)) - 0.5
        npMat2 = np.random.random((128, 128, 3)) - 0.5

        cuMat1 = cv.cuda_GpuMat()
        cuMat2 = cv.cuda_GpuMat()
        cuMat1.upload(npMat1)
        cuMat2.upload(npMat2)

        self.assertTrue(np.allclose(cv.cuda.add(cuMat1, cuMat2).download(),
                                         cv.add(npMat1, npMat2)))

        self.assertTrue(np.allclose(cv.cuda.subtract(cuMat1, cuMat2).download(),
                                         cv.subtract(npMat1, npMat2)))

        self.assertTrue(np.allclose(cv.cuda.multiply(cuMat1, cuMat2).download(),
                                         cv.multiply(npMat1, npMat2)))

        self.assertTrue(np.allclose(cv.cuda.divide(cuMat1, cuMat2).download(),
                                         cv.divide(npMat1, npMat2)))

        self.assertTrue(np.allclose(cv.cuda.absdiff(cuMat1, cuMat2).download(),
                                         cv.absdiff(npMat1, npMat2)))

        self.assertTrue(np.allclose(cv.cuda.compare(cuMat1, cuMat2, cv.CMP_GE).download(),
                                         cv.compare(npMat1, npMat2, cv.CMP_GE)))

        self.assertTrue(np.allclose(cv.cuda.abs(cuMat1).download(),
                                         np.abs(npMat1)))

        self.assertTrue(np.allclose(cv.cuda.sqrt(cv.cuda.sqr(cuMat1)).download(),
                                    cv.cuda.abs(cuMat1).download()))


        self.assertTrue(np.allclose(cv.cuda.log(cv.cuda.exp(cuMat1)).download(),
                                                            npMat1))

        self.assertTrue(np.allclose(cv.cuda.pow(cuMat1, 2).download(),
                                         cv.pow(npMat1, 2)))

    def test_cudaarithm_logical(self):
        npMat1 = (np.random.random((128, 128)) * 255).astype(np.uint8)
        npMat2 = (np.random.random((128, 128)) * 255).astype(np.uint8)

        cuMat1 = cv.cuda_GpuMat()
        cuMat2 = cv.cuda_GpuMat()
        cuMat1.upload(npMat1)
        cuMat2.upload(npMat2)

        self.assertTrue(np.allclose(cv.cuda.bitwise_or(cuMat1, cuMat2).download(),
                                         cv.bitwise_or(npMat1, npMat2)))

        self.assertTrue(np.allclose(cv.cuda.bitwise_and(cuMat1, cuMat2).download(),
                                         cv.bitwise_and(npMat1, npMat2)))

        self.assertTrue(np.allclose(cv.cuda.bitwise_xor(cuMat1, cuMat2).download(),
                                         cv.bitwise_xor(npMat1, npMat2)))

        self.assertTrue(np.allclose(cv.cuda.bitwise_not(cuMat1).download(),
                                         cv.bitwise_not(npMat1)))

        self.assertTrue(np.allclose(cv.cuda.min(cuMat1, cuMat2).download(),
                                         cv.min(npMat1, npMat2)))

        self.assertTrue(np.allclose(cv.cuda.max(cuMat1, cuMat2).download(),
                                         cv.max(npMat1, npMat2)))

    def test_cudabgsegm_existence(self):
        #Test at least the existence of wrapped functions for now

        bgsub = cv.cuda.createBackgroundSubtractorMOG()
        bgsub = cv.cuda.createBackgroundSubtractorMOG2()

        self.assertTrue(True) #It is sufficient that no exceptions have been there

    def test_cudacodec_existence(self):
        #Test at least the existence of wrapped functions for now

        try:
            writer = cv.cudacodec.createVideoWriter("tmp", (128, 128), 30)
            reader = cv.cudacodec.createVideoReader("tmp")
        except cv.error as e:
            self.assertEqual(e.code, cv.Error.StsNotImplemented)
            self.skipTest("NVCUVENC is not installed")

        self.assertTrue(True) #It is sufficient that no exceptions have been there

    def test_cudafeatures2d(self):
        npMat1 = self.get_sample("samples/data/right01.jpg")
        npMat2 = self.get_sample("samples/data/right02.jpg")

        cuMat1 = cv.cuda_GpuMat()
        cuMat2 = cv.cuda_GpuMat()
        cuMat1.upload(npMat1)
        cuMat2.upload(npMat2)

        cuMat1 = cv.cuda.cvtColor(cuMat1, cv.COLOR_RGB2GRAY)
        cuMat2 = cv.cuda.cvtColor(cuMat2, cv.COLOR_RGB2GRAY)

        fast = cv.cuda_FastFeatureDetector.create()
        kps = fast.detectAsync(cuMat1)

        orb = cv.cuda_ORB.create()
        kps1, descs1 = orb.detectAndComputeAsync(cuMat1, None)
        kps2, descs2 = orb.detectAndComputeAsync(cuMat2, None)

        bf = cv.cuda_DescriptorMatcher.createBFMatcher(cv.NORM_HAMMING)
        matches = bf.match(descs1, descs2)
        self.assertGreater(len(matches), 0)
        matches = bf.knnMatch(descs1, descs2, 2)
        self.assertGreater(len(matches), 0)
        matches = bf.radiusMatch(descs1, descs2, 0.1)
        self.assertGreater(len(matches), 0)

        self.assertTrue(True) #It is sufficient that no exceptions have been there

    def test_cudafilters_existence(self):
        #Test at least the existence of wrapped functions for now

        filter = cv.cuda.createBoxFilter(cv.CV_8UC1, -1, (3, 3))
        filter = cv.cuda.createLinearFilter(cv.CV_8UC4, -1, np.eye(3))
        filter = cv.cuda.createLaplacianFilter(cv.CV_16UC1, -1, ksize=3)
        filter = cv.cuda.createSeparableLinearFilter(cv.CV_8UC1, -1, np.eye(3), np.eye(3))
        filter = cv.cuda.createDerivFilter(cv.CV_8UC1, -1, 1, 1, 3)
        filter = cv.cuda.createSobelFilter(cv.CV_8UC1, -1, 1, 1)
        filter = cv.cuda.createScharrFilter(cv.CV_8UC1, -1, 1, 0)
        filter = cv.cuda.createGaussianFilter(cv.CV_8UC1, -1, (3, 3), 16)
        filter = cv.cuda.createMorphologyFilter(cv.MORPH_DILATE, cv.CV_32FC1, np.eye(3))
        filter = cv.cuda.createBoxMaxFilter(cv.CV_8UC1, (3, 3))
        filter = cv.cuda.createBoxMinFilter(cv.CV_8UC1, (3, 3))
        filter = cv.cuda.createRowSumFilter(cv.CV_8UC1, cv.CV_32FC1, 3)
        filter = cv.cuda.createColumnSumFilter(cv.CV_8UC1, cv.CV_32FC1, 3)
        filter = cv.cuda.createMedianFilter(cv.CV_8UC1, 3)

        self.assertTrue(True) #It is sufficient that no exceptions have been there

    def test_cudafilters_laplacian(self):
        npMat = (np.random.random((128, 128)) * 255).astype(np.uint16)
        cuMat = cv.cuda_GpuMat()
        cuMat.upload(npMat)

        self.assertTrue(np.allclose(cv.cuda.createLaplacianFilter(cv.CV_16UC1, -1, ksize=3).apply(cuMat).download(),
                                         cv.Laplacian(npMat, cv.CV_16UC1, ksize=3)))

    def test_cudaimgproc(self):
        npC1 = (np.random.random((128, 128)) * 255).astype(np.uint8)
        npC3 = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
        npC4 = (np.random.random((128, 128, 4)) * 255).astype(np.uint8)
        cuC1 = cv.cuda_GpuMat()
        cuC3 = cv.cuda_GpuMat()
        cuC4 = cv.cuda_GpuMat()
        cuC1.upload(npC1)
        cuC3.upload(npC3)
        cuC4.upload(npC4)

        cv.cuda.cvtColor(cuC3, cv.COLOR_RGB2HSV)
        cv.cuda.demosaicing(cuC1, cv.cuda.COLOR_BayerGR2BGR_MHT)
        cv.cuda.gammaCorrection(cuC3)
        cv.cuda.alphaComp(cuC4, cuC4, cv.cuda.ALPHA_XOR)
        cv.cuda.calcHist(cuC1)
        cv.cuda.equalizeHist(cuC1)
        cv.cuda.evenLevels(3, 0, 255)
        cv.cuda.meanShiftFiltering(cuC4, 10, 5)
        cv.cuda.meanShiftProc(cuC4, 10, 5)
        cv.cuda.bilateralFilter(cuC3, 3, 16, 3)
        cv.cuda.blendLinear

        cv.cuda.meanShiftSegmentation(cuC4, 10, 5, 5).download()

        clahe = cv.cuda.createCLAHE()
        clahe.apply(cuC1, cv.cuda_Stream.Null());

        histLevels = cv.cuda.histEven(cuC3, 20, 0, 255)
        cv.cuda.histRange(cuC1, histLevels)

        detector = cv.cuda.createCannyEdgeDetector(0, 100)
        detector.detect(cuC1)

        detector = cv.cuda.createHoughLinesDetector(3, np.pi / 180, 20)
        detector.detect(cuC1)

        detector = cv.cuda.createHoughSegmentDetector(3, np.pi / 180, 20, 5)
        detector.detect(cuC1)

        detector = cv.cuda.createHoughCirclesDetector(3, 20, 10, 10, 20, 100)
        detector.detect(cuC1)

        detector = cv.cuda.createGeneralizedHoughBallard()
        #BUG: detect accept only Mat!
        #Even if generate_gpumat_decls is set to True, it only wraps overload CUDA functions.
        #The problem is that Mat and GpuMat are not fully compatible to enable system-wide overloading
        #detector.detect(cuC1, cuC1, cuC1)

        detector = cv.cuda.createGeneralizedHoughGuil()
        #BUG: same as above..
        #detector.detect(cuC1, cuC1, cuC1)

        detector = cv.cuda.createHarrisCorner(cv.CV_8UC1, 15, 5, 1)
        detector.compute(cuC1)

        detector = cv.cuda.createMinEigenValCorner(cv.CV_8UC1, 15, 5, 1)
        detector.compute(cuC1)

        detector = cv.cuda.createGoodFeaturesToTrackDetector(cv.CV_8UC1)
        detector.detect(cuC1)

        matcher = cv.cuda.createTemplateMatching(cv.CV_8UC1, cv.TM_CCOEFF_NORMED)
        matcher.match(cuC3, cuC3)

        self.assertTrue(True) #It is sufficient that no exceptions have been there

    def test_cudaimgproc_cvtColor(self):
        npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
        cuMat = cv.cuda_GpuMat()
        cuMat.upload(npMat)

        self.assertTrue(np.allclose(cv.cuda.cvtColor(cuMat, cv.COLOR_BGR2HSV).download(),
                                         cv.cvtColor(npMat, cv.COLOR_BGR2HSV)))

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
