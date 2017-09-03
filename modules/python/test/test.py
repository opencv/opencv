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
    unit_argv = [sys.argv[0]] + other
    unittest.main(argv=unit_argv)
