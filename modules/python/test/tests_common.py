#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import unittest
import hashlib
import random
import argparse

import numpy as np
import cv2 as cv

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

    def get_sample(self, filename, iscolor = cv.IMREAD_COLOR):
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
            self.image_cache[filename] = cv.imdecode(np.fromstring(filedata, dtype=np.uint8), iscolor)
        return self.image_cache[filename]

    def setUp(self):
        cv.setRNGSeed(10)
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

    @staticmethod
    def bootstrap():
        parser = argparse.ArgumentParser(description='run OpenCV python tests')
        parser.add_argument('--repo', help='use sample image files from local git repository (path to folder), '
                                           'if not set, samples will be downloaded from github.com')
        parser.add_argument('--data', help='<not used> use data files from local folder (path to folder), '
                                            'if not set, data files will be downloaded from docs.opencv.org')
        args, other = parser.parse_known_args()
        print("Testing OpenCV", cv.__version__)
        print("Local repo path:", args.repo)
        NewOpenCVTests.repoPath = args.repo
        try:
            NewOpenCVTests.extraTestDataPath = os.environ['OPENCV_TEST_DATA_PATH']
        except KeyError:
            print('Missing opencv extra repository. Some of tests may fail.')
        random.seed(0)
        unit_argv = [sys.argv[0]] + other
        unittest.main(argv=unit_argv)


def intersectionRate(s1, s2):

    x1, y1, x2, y2 = s1
    s1 = np.array([[x1, y1], [x2,y1], [x2, y2], [x1, y2]])

    x1, y1, x2, y2 = s2
    s2 = np.array([[x1, y1], [x2,y1], [x2, y2], [x1, y2]])

    area, _intersection = cv.intersectConvexConvex(s1, s2)
    return 2 * area / (cv.contourArea(s1) + cv.contourArea(s2))

def isPointInRect(p, rect):
    if rect[0] <= p[0] and rect[1] <=p[1] and p[0] <= rect[2] and p[1] <= rect[3]:
        return True
    else:
        return False