#!/usr/bin/env python

from __future__ import print_function

import unittest
import sys
import hashlib
import os
import numpy as np
import cv2
import cv2.cv as cv

# Python 3 moved urlopen to urllib.requests
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

class OpenCVTests(unittest.TestCase):

    # path to local repository folder containing 'samples' folder
    repoPath = None
    # github repository url
    repoUrl = 'https://raw.github.com/Itseez/opencv/2.4'
    # path to local folder containing 'camera_calibration.tar.gz'
    dataPath = None
    # data url
    dataUrl = 'http://docs.opencv.org/data'

    depths = [ cv.IPL_DEPTH_8U, cv.IPL_DEPTH_8S, cv.IPL_DEPTH_16U, cv.IPL_DEPTH_16S, cv.IPL_DEPTH_32S, cv.IPL_DEPTH_32F, cv.IPL_DEPTH_64F ]

    mat_types = [
        cv.CV_8UC1,
        cv.CV_8UC2,
        cv.CV_8UC3,
        cv.CV_8UC4,
        cv.CV_8SC1,
        cv.CV_8SC2,
        cv.CV_8SC3,
        cv.CV_8SC4,
        cv.CV_16UC1,
        cv.CV_16UC2,
        cv.CV_16UC3,
        cv.CV_16UC4,
        cv.CV_16SC1,
        cv.CV_16SC2,
        cv.CV_16SC3,
        cv.CV_16SC4,
        cv.CV_32SC1,
        cv.CV_32SC2,
        cv.CV_32SC3,
        cv.CV_32SC4,
        cv.CV_32FC1,
        cv.CV_32FC2,
        cv.CV_32FC3,
        cv.CV_32FC4,
        cv.CV_64FC1,
        cv.CV_64FC2,
        cv.CV_64FC3,
        cv.CV_64FC4,
    ]
    mat_types_single = [
        cv.CV_8UC1,
        cv.CV_8SC1,
        cv.CV_16UC1,
        cv.CV_16SC1,
        cv.CV_32SC1,
        cv.CV_32FC1,
        cv.CV_64FC1,
    ]

    def depthsize(self, d):
        return { cv.IPL_DEPTH_8U : 1,
                 cv.IPL_DEPTH_8S : 1,
                 cv.IPL_DEPTH_16U : 2,
                 cv.IPL_DEPTH_16S : 2,
                 cv.IPL_DEPTH_32S : 4,
                 cv.IPL_DEPTH_32F : 4,
                 cv.IPL_DEPTH_64F : 8 }[d]

    def get_sample(self, filename, iscolor = cv.CV_LOAD_IMAGE_COLOR):
        if not filename in self.image_cache:
            filedata = None
            if OpenCVTests.repoPath is not None:
                candidate = OpenCVTests.repoPath + '/' + filename
                if os.path.isfile(candidate):
                    with open(candidate, 'rb') as f:
                        filedata = f.read()
            if filedata is None:
                filedata = urllib.urlopen(OpenCVTests.repoUrl + '/' + filename).read()
            imagefiledata = cv.CreateMatHeader(1, len(filedata), cv.CV_8UC1)
            cv.SetData(imagefiledata, filedata, len(filedata))
            self.image_cache[filename] = cv.DecodeImageM(imagefiledata, iscolor)
        return self.image_cache[filename]

    def get_data(self, filename, urlbase):
        if (not os.path.isfile(filename)):
            if OpenCVTests.dataPath is not None:
                candidate = OpenCVTests.dataPath + '/' + filename
                if os.path.isfile(candidate):
                    return candidate
            urllib.urlretrieve(urlbase + '/' + filename, filename)
        return filename

    def setUp(self):
        self.image_cache = {}

    def snap(self, img):
        self.snapL([img])

    def snapL(self, L):
        for i,img in enumerate(L):
            cv.NamedWindow("snap-%d" % i, 1)
            cv.ShowImage("snap-%d" % i, img)
        cv.WaitKey()
        cv.DestroyAllWindows()

    def hashimg(self, im):
        """ Compute a hash for an image, useful for image comparisons """
        return hashlib.md5(im.tostring()).digest()


class NewOpenCVTests(unittest.TestCase):

    # path to local repository folder containing 'samples' folder
    repoPath = None
    extraTestDataPath = None
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
