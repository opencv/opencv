#/usr/bin/env python

import unittest
import random
import time
import math
import sys
import array
import urllib
import tarfile
import hashlib
import os
import getopt
import operator
import functools
import numpy as np
import cv2
import cv2.cv as cv

class NewOpenCVTests(unittest.TestCase):

    def get_sample(self, filename, iscolor = cv.CV_LOAD_IMAGE_COLOR):
        if not filename in self.image_cache:
            filedata = urllib.urlopen("https://raw.github.com/Itseez/opencv/master/" + filename).read()
            self.image_cache[filename] = cv2.imdecode(np.fromstring(filedata, dtype=np.uint8), iscolor)
        return self.image_cache[filename]

    def setUp(self):
        self.image_cache = {}

    def hashimg(self, im):
        """ Compute a hash for an image, useful for image comparisons """
        return hashlib.md5(im.tostring()).digest()

# Tests to run first; check the handful of basic operations that the later tests rely on

class Hackathon244Tests(NewOpenCVTests):
    
    def test_int_array(self):
        a = np.array([-1, 2, -3, 4, -5])
        absa0 = np.abs(a)
        self.assert_(cv2.norm(a, cv2.NORM_L1) == 15)
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
        fd = cv2.FastFeatureDetector(30, True)
        img = self.get_sample("samples/cpp/right02.jpg", 0)
        img = cv2.medianBlur(img, 3)
        imgc = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        keypoints = fd.detect(img)
        self.assert_(600 <= len(keypoints) <= 700)
        for kpt in keypoints:
            self.assertNotEqual(kpt.response, 0)


if __name__ == '__main__':
    print "testing", cv.__version__
    random.seed(0)
    unittest.main()
