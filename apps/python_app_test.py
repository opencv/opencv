#!/usr/bin/env python

from __future__ import print_function

import sys
sys.dont_write_bytecode = True  # Don't generate .pyc files / __pycache__ directories

import os
import sys
import unittest

# Python 3 moved urlopen to urllib.requests
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

basedir = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(os.path.split(basedir)[0], "modules", "python", "test"))
from tests_common import NewOpenCVTests

def load_tests(loader, tests, pattern):
    cwd = os.getcwd()
    config_file = 'opencv_apps_python_tests.cfg'
    locations = [cwd, basedir]
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            locations += [str(s).strip() for s in f.readlines()]
    else:
        print('WARNING: OpenCV tests config file ({}) is missing, running subset of tests'.format(config_file))

    tests_pattern = os.environ.get('OPENCV_APPS_TEST_FILTER', 'test_*') + '.py'
    if tests_pattern != 'test_*.py':
        print('Tests filter: {}'.format(tests_pattern))

    processed = set()
    for l in locations:
        if not os.path.isabs(l):
            l = os.path.normpath(os.path.join(cwd, l))
        if l in processed:
            continue
        processed.add(l)
        print('Discovering python tests from: {}'.format(l))
        sys_path_modify = l not in sys.path
        if sys_path_modify:
            sys.path.append(l)  # Hack python loader
        discovered_tests = loader.discover(l, pattern=tests_pattern, top_level_dir=l)
        print('    found {} tests'.format(discovered_tests.countTestCases()))
        tests.addTests(loader.discover(l, pattern=tests_pattern))
        if sys_path_modify:
            sys.path.remove(l)
    return tests

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
