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