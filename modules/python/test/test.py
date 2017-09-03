#!/usr/bin/env python

from __future__ import print_function

import os
import unittest

# Python 3 moved urlopen to urllib.requests
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

from tests_common import NewOpenCVTests


basedir = os.path.abspath(os.path.dirname(__file__))

def load_tests(loader, tests, pattern):
    tests.addTests(loader.discover(basedir, pattern='test_*.py'))
    return tests

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
