#!/usr/bin/env python

"""
A script for loading and running OpenCV tests with optional custom filtering.
"""

from __future__ import print_function
import os
import sys
import unittest

# Don't generate .pyc files / __pycache__ directories
sys.dont_write_bytecode = True  

# Python 3 moved urlopen to urllib.requests
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

# Importing OpenCV tests module
sys.path.append(os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], "modules", "python", "test"))
from tests_common import NewOpenCVTests


def load_tests(loader, tests, pattern):
    """
    Discovers and loads the tests from specified locations.

    Arguments:
    loader -- unittest.TestLoader instance
    tests -- initial TestSuite
    pattern -- string pattern for filtering test files

    Returns:
    tests -- TestSuite with all the discovered tests
    """
    cwd = os.getcwd()
    config_file = 'opencv_apps_python_tests.cfg'
    locations = [cwd, os.path.dirname(os.path.abspath(__file__))]

    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            locations += [str(s).strip() for s in f.readlines()]
    else:
        print(f'WARNING: OpenCV tests config file ({config_file}) is missing, running subset of tests')

    tests_pattern = os.environ.get('OPENCV_APPS_TEST_FILTER', 'test_*') + '.py'
    if tests_pattern != 'test_*.py':
        print(f'Tests filter: {tests_pattern}')

    processed = set()
    for l in locations:
        l = os.path.normpath(l) if not os.path.isabs(l) else l
        if l in processed:
            continue
        processed.add(l)
        print(f'Discovering python tests from: {l}')
        sys_path_modify = l not in sys.path
        if sys_path_modify:
            sys.path.append(l)  # Hack python loader

        discovered_tests = loader.discover(l, pattern=tests_pattern, top_level_dir=l)
        print(f'    found {discovered_tests.countTestCases()} tests')
        tests.addTests(discovered_tests)

        if sys_path_modify:
            sys.path.remove(l)
    return tests


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
