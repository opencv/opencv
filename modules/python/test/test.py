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

class NewOpenCVTests(unittest.TestCase):

    # path to local repository folder containing 'samples' folder
    repoPath = None
    # github repository url
    repoUrl = 'https://raw.github.com/Itseez/opencv/master'

    def get_sample(self, filename, iscolor = cv2.IMREAD_COLOR):
        if not filename in self.image_cache:
            filedata = None
            if NewOpenCVTests.repoPath is not None:
                candidate = NewOpenCVTests.repoPath + '/' + filename
                if os.path.isfile(candidate):
                    with open(candidate, 'rb') as f:
                        filedata = f.read()
            if filedata is None:
                filedata = urlopen(NewOpenCVTests.repoUrl + '/' + filename).read()
            self.image_cache[filename] = cv2.imdecode(np.fromstring(filedata, dtype=np.uint8), iscolor)
        return self.image_cache[filename]

    def setUp(self):
        self.image_cache = {}

    def hashimg(self, im):
        """ Compute a hash for an image, useful for image comparisons """
        return hashlib.md5(im.tostring()).digest()

    if sys.version_info[:2] == (2, 6):
        def assertLess(self, a, b, msg=None):
            if not a < b:
                self.fail('%s not less than %s' % (repr(a), repr(b)))

        def assertLessEqual(self, a, b, msg=None):
            if not a <= b:
                self.fail('%s not less than or equal to %s' % (repr(a), repr(b)))

        def assertGreater(self, a, b, msg=None):
            if not a > b:
                self.fail('%s not greater than %s' % (repr(a), repr(b)))

# Tests to run first; check the handful of basic operations that the later tests rely on

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
    random.seed(0)
    unit_argv = [sys.argv[0]] + other;
    unittest.main(argv=unit_argv)
