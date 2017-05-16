#!/usr/bin/env python

from __future__ import print_function
import unittest
import random
import time
import math
import sys
import array
import tarfile
import hashlib
import os
import getopt
import operator
import functools
import numpy as np
import cv2
import argparse

# Python 3 moved urlopen to urllib.requests
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

from tests_common import NewOpenCVTests

# Tests to run first; check the handful of basic operations that the later tests rely on

basedir = os.path.abspath(os.path.dirname(__file__))

def load_tests(loader, tests, pattern):
    tests.addTests(loader.discover(basedir, pattern='test_*.py'))
    return tests

class Hackathon244Tests(NewOpenCVTests):

    def test_int_array(self):
        a = np.array([-1, 2, -3, 4, -5])
        absa0 = np.abs(a)
        self.assertTrue(cv2.norm(a, cv2.NORM_L1) == 15)
        absa1 = cv2.absdiff(a, 0)
        self.assertEqual(cv2.norm(absa1, absa0, cv2.NORM_INF), 0)

    def test_imencode(self):
        a = np.zeros((480, 640), dtype=np.uint8)
        flag, ajpg = cv2.imencode("img_q90.jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.assertEqual(flag, True)
        self.assertEqual(ajpg.dtype, np.uint8)
        self.assertGreater(ajpg.shape[0], 1)
        self.assertEqual(ajpg.shape[1], 1)

    def test_projectPoints(self):
        objpt = np.float64([[1,2,3]])
        imgpt0, jac0 = cv2.projectPoints(objpt, np.zeros(3), np.zeros(3), np.eye(3), np.float64([]))
        imgpt1, jac1 = cv2.projectPoints(objpt, np.zeros(3), np.zeros(3), np.eye(3), None)
        self.assertEqual(imgpt0.shape, (objpt.shape[0], 1, 2))
        self.assertEqual(imgpt1.shape, imgpt0.shape)
        self.assertEqual(jac0.shape, jac1.shape)
        self.assertEqual(jac0.shape[0], 2*objpt.shape[0])

    def test_estimateAffine3D(self):
        pattern_size = (11, 8)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= 10
        (retval, out, inliers) = cv2.estimateAffine3D(pattern_points, pattern_points)
        self.assertEqual(retval, 1)
        if cv2.norm(out[2,:]) < 1e-3:
            out[2,2]=1
        self.assertLess(cv2.norm(out, np.float64([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])), 1e-3)
        self.assertEqual(cv2.countNonZero(inliers), pattern_size[0]*pattern_size[1])

    def test_fast(self):
        fd = cv2.FastFeatureDetector_create(30, True)
        img = self.get_sample("samples/data/right02.jpg", 0)
        img = cv2.medianBlur(img, 3)
        imgc = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        keypoints = fd.detect(img)
        self.assertTrue(600 <= len(keypoints) <= 700)
        for kpt in keypoints:
            self.assertNotEqual(kpt.response, 0)

    def check_close_angles(self, a, b, angle_delta):
        self.assertTrue(abs(a - b) <= angle_delta or
                        abs(360 - abs(a - b)) <= angle_delta)

    def check_close_pairs(self, a, b, delta):
        self.assertLessEqual(abs(a[0] - b[0]), delta)
        self.assertLessEqual(abs(a[1] - b[1]), delta)

    def check_close_boxes(self, a, b, delta, angle_delta):
        self.check_close_pairs(a[0], b[0], delta)
        self.check_close_pairs(a[1], b[1], delta)
        self.check_close_angles(a[2], b[2], angle_delta)

    def test_geometry(self):
        npt = 100
        np.random.seed(244)
        a = np.random.randn(npt,2).astype('float32')*50 + 150

        img = np.zeros((300, 300, 3), dtype='uint8')
        be = cv2.fitEllipse(a)
        br = cv2.minAreaRect(a)
        mc, mr = cv2.minEnclosingCircle(a)

        be0 = ((150.2511749267578, 150.77322387695312), (158.024658203125, 197.57696533203125), 37.57804489135742)
        br0 = ((161.2974090576172, 154.41793823242188), (199.2301483154297, 207.7177734375), -9.164555549621582)
        mc0, mr0 = (160.41790771484375, 144.55152893066406), 136.713500977

        self.check_close_boxes(be, be0, 5, 15)
        self.check_close_boxes(br, br0, 5, 15)
        self.check_close_pairs(mc, mc0, 5)
        self.assertLessEqual(abs(mr - mr0), 5)

    def test_inheritance(self):
        bm = cv2.StereoBM_create()
        bm.getPreFilterCap() # from StereoBM
        bm.getBlockSize() # from SteroMatcher

        boost = cv2.ml.Boost_create()
        boost.getBoostType() # from ml::Boost
        boost.getMaxDepth() # from ml::DTrees
        boost.isClassifier() # from ml::StatModel

    def test_umat_construct(self):
        data = np.random.random([512, 512])
        # UMat constructors
        data_um = cv2.UMat(data)  # from ndarray
        data_sub_um = cv2.UMat(data_um, [128, 256], [128, 256])  # from UMat
        data_dst_um = cv2.UMat(128, 128, cv2.CV_64F)  # from size/type
        # test continuous and submatrix flags
        assert data_um.isContinuous() and not data_um.isSubmatrix()
        assert not data_sub_um.isContinuous() and data_sub_um.isSubmatrix()
        # test operation on submatrix
        cv2.multiply(data_sub_um, 2., dst=data_dst_um)
        assert np.allclose(2. * data[128:256, 128:256], data_dst_um.get())

    def test_umat_handle(self):
        a_um = cv2.UMat(256, 256, cv2.CV_32F)
        ctx_handle = cv2.UMat.context()  # obtain context handle
        queue_handle = cv2.UMat.queue()  # obtain queue handle
        a_handle = a_um.handle(cv2.ACCESS_READ)  # obtain buffer handle
        offset = a_um.offset  # obtain buffer offset

    def test_umat_matching(self):
        img1 = self.get_sample("samples/data/right01.jpg")
        img2 = self.get_sample("samples/data/right02.jpg")

        orb = cv2.ORB_create()

        img1, img2 = cv2.UMat(img1), cv2.UMat(img2)
        ps1, descs_umat1 = orb.detectAndCompute(img1, None)
        ps2, descs_umat2 = orb.detectAndCompute(img2, None)

        self.assertIsInstance(descs_umat1, cv2.UMat)
        self.assertIsInstance(descs_umat2, cv2.UMat)
        self.assertGreater(len(ps1), 0)
        self.assertGreater(len(ps2), 0)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        res_umats = bf.match(descs_umat1, descs_umat2)
        res = bf.match(descs_umat1.get(), descs_umat2.get())

        self.assertGreater(len(res), 0)
        self.assertEqual(len(res_umats), len(res))

    def test_umat_optical_flow(self):
        img1 = self.get_sample("samples/data/right01.jpg", cv2.IMREAD_GRAYSCALE)
        img2 = self.get_sample("samples/data/right02.jpg", cv2.IMREAD_GRAYSCALE)
        # Note, that if you want to see performance boost by OCL implementation - you need enough data
        # For example you can increase maxCorners param to 10000 and increase img1 and img2 in such way:
        # img = np.hstack([np.vstack([img] * 6)] * 6)

        feature_params = dict(maxCorners=239,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)

        p0 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)
        p0_umat = cv2.goodFeaturesToTrack(cv2.UMat(img1), mask=None, **feature_params)
        self.assertEqual(p0_umat.get().shape, p0.shape)

        p0 = np.array(sorted(p0, key=lambda p: tuple(p[0])))
        p0_umat = cv2.UMat(np.array(sorted(p0_umat.get(), key=lambda p: tuple(p[0]))))
        self.assertTrue(np.allclose(p0_umat.get(), p0))

        p1_mask_err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None)

        p1_mask_err_umat0 = map(cv2.UMat.get, cv2.calcOpticalFlowPyrLK(img1, img2, p0_umat, None))
        p1_mask_err_umat1 = map(cv2.UMat.get, cv2.calcOpticalFlowPyrLK(cv2.UMat(img1), img2, p0_umat, None))
        p1_mask_err_umat2 = map(cv2.UMat.get, cv2.calcOpticalFlowPyrLK(img1, cv2.UMat(img2), p0_umat, None))

        # # results of OCL optical flow differs from CPU implementation, so result can not be easily compared
        # for p1_mask_err_umat in [p1_mask_err_umat0, p1_mask_err_umat1, p1_mask_err_umat2]:
        #     for data, data_umat in zip(p1_mask_err, p1_mask_err_umat):
        #         self.assertTrue(np.allclose(data, data_umat))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run OpenCV python tests')
    parser.add_argument('--repo', help='use sample image files from local git repository (path to folder), '
                                       'if not set, samples will be downloaded from github.com')
    parser.add_argument('--data', help='<not used> use data files from local folder (path to folder), '
                                        'if not set, data files will be downloaded from docs.opencv.org')
    args, other = parser.parse_known_args()
    print("Testing OpenCV", cv2.__version__)
    print("Local repo path:", args.repo)
    NewOpenCVTests.repoPath = args.repo
    try:
        NewOpenCVTests.extraTestDataPath = os.environ['OPENCV_TEST_DATA_PATH']
    except KeyError:
        print('Missing opencv extra repository. Some of tests may fail.')
    random.seed(0)
    unit_argv = [sys.argv[0]] + other;
    unittest.main(argv=unit_argv)
