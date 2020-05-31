#!/usr/bin/env python
"""
This script runs OpenCV.framework tests for OSX.
"""

from __future__ import print_function
import os, os.path, sys, argparse, traceback, multiprocessing

# import common code
sys.path.insert(0, os.path.abspath(os.path.abspath(os.path.dirname(__file__))+'/../ios'))
from run_tests import TestRunner

MACOSX_DEPLOYMENT_TARGET='10.12'  # default, can be changed via command line options or environment variable

class OSXTestRunner(TestRunner):

    def getToolchain(self):
        return None

    def getCMakeArgs(self):
        args = TestRunner.getCMakeArgs(self)
        args = args + [
            '-DMACOSX_DEPLOYMENT_TARGET=%s' % os.environ['MACOSX_DEPLOYMENT_TARGET']
        ]
        return args


if __name__ == "__main__":
    script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
    parser = argparse.ArgumentParser(description='The script builds OpenCV.framework for OSX.')
    parser.add_argument('tests_dir', metavar='TEST_DIR', help='folder where test files are located')
    parser.add_argument('--build_dir', default=None, help='folder where test will be built (default is "../test_build" relative to tests_dir)')
    parser.add_argument('--framework_dir', default=None, help='folder where OpenCV framework is located')
    parser.add_argument('--framework_name', default='opencv2', help='Name of OpenCV framework (default: opencv2, will change to OpenCV in future version)')
    parser.add_argument('--macosx_deployment_target', default=os.environ.get('MACOSX_DEPLOYMENT_TARGET', MACOSX_DEPLOYMENT_TARGET), help='specify MACOSX_DEPLOYMENT_TARGET')
    parser.add_argument('--platform', default='macOS,arch=x86_64', help='xcodebuild platform parameter (default is macOS)')

    args = parser.parse_args()
    os.environ['MACOSX_DEPLOYMENT_TARGET'] = args.macosx_deployment_target
    arch = "x86_64"
    target = "macOS"

    r = OSXTestRunner(script_dir, args.tests_dir, args.build_dir if args.build_dir else os.path.join(args.tests_dir, "../test_build"), args.framework_dir, args.framework_name, arch, target, args.platform)
    r.run()
