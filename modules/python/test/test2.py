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
        self.assert_(cv2.norm(absa1, absa0, cv2.NORM_INF) == 0)
        
    def test_imencode(self):
        a = np.zeros((480, 640), dtype=np.uint8)
        flag, ajpg = cv2.imencode("img_q90.jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.assert_(flag == True and ajpg.dtype == np.uint8 and ajpg.shape[0] > 1 and ajpg.shape[1] == 1)

if __name__ == '__main__':
    print "testing", cv.__version__
    random.seed(0)
    unittest.main()
