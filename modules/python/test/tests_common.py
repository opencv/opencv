#!/usr/bin/env python

from __future__ import print_function

import unittest
import sys
import hashlib
import os
import numpy as np
import cv2

# Python 3 moved urlopen to urllib.requests
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

class NewOpenCVTests(unittest.TestCase):

    # path to local repository folder containing 'samples' folder
    repoPath = None
    extraTestDataPath = None
    # github repository url
    repoUrl = 'https://raw.github.com/opencv/opencv/master'

    def get_sample(self, filename, iscolor = cv2.IMREAD_COLOR):
        if not filename in self.image_cache:
            filedata = None
            if NewOpenCVTests.repoPath is not None:
                candidate = NewOpenCVTests.repoPath + '/' + filename
                if os.path.isfile(candidate):
                    with open(candidate, 'rb') as f:
                        filedata = f.read()
            if NewOpenCVTests.extraTestDataPath is not None:
                candidate = NewOpenCVTests.extraTestDataPath + '/' + filename
                if os.path.isfile(candidate):
                    with open(candidate, 'rb') as f:
                        filedata = f.read()
            if filedata is None:
                return None#filedata = urlopen(NewOpenCVTests.repoUrl + '/' + filename).read()
            self.image_cache[filename] = cv2.imdecode(np.fromstring(filedata, dtype=np.uint8), iscolor)
        return self.image_cache[filename]

    def setUp(self):
        cv2.setRNGSeed(10)
        self.image_cache = {}

    def hashimg(self, im):
        """ Compute a hash for an image, useful for image comparisons """
        return hashlib.md5(im.tostring()).hexdigest()

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

def intersectionRate(s1, s2):

    x1, y1, x2, y2 = s1
    s1 = np.array([[x1, y1], [x2,y1], [x2, y2], [x1, y2]])

    x1, y1, x2, y2 = s2
    s2 = np.array([[x1, y1], [x2,y1], [x2, y2], [x1, y2]])

    area, intersection = cv2.intersectConvexConvex(s1, s2)
    return 2 * area / (cv2.contourArea(s1) + cv2.contourArea(s2))

def isPointInRect(p, rect):
    if rect[0] <= p[0] and rect[1] <=p[1] and p[0] <= rect[2] and p[1] <= rect[3]:
        return True
    else:
        return False