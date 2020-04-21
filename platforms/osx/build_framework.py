#!/usr/bin/env python
"""
The script builds OpenCV.framework for OSX.
"""

from __future__ import print_function
import os, os.path, sys, argparse, traceback, multiprocessing

# import common code
sys.path.insert(0, os.path.abspath(os.path.abspath(os.path.dirname(__file__))+'/../ios'))
from build_framework import Builder

MACOSX_DEPLOYMENT_TARGET='10.12'  # default, can be changed via command line options or environment variable

class OSXBuilder(Builder):

    def getToolchain(self, arch, target):
        return None

    def getBuildCommand(self, archs, target):
        buildcmd = [
            "xcodebuild",
            "MACOSX_DEPLOYMENT_TARGET=" + os.environ['MACOSX_DEPLOYMENT_TARGET'],
            "ARCHS=%s" % archs[0],
            "-sdk", target.lower(),
            "-configuration", "Debug" if self.debug else "Release",
            "-parallelizeTargets",
            "-jobs", str(multiprocessing.cpu_count())
        ]
        return buildcmd

    def getInfoPlist(self, builddirs):
        return os.path.join(builddirs[0], "osx", "Info.plist")


if __name__ == "__main__":
    folder = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), "../.."))
    parser = argparse.ArgumentParser(description='The script builds OpenCV.framework for OSX.')
    parser.add_argument('out', metavar='OUTDIR', help='folder to put built framework')
    parser.add_argument('--opencv', metavar='DIR', default=folder, help='folder with opencv repository (default is "../.." relative to script location)')
    parser.add_argument('--contrib', metavar='DIR', default=None, help='folder with opencv_contrib repository (default is "None" - build only main framework)')
    parser.add_argument('--without', metavar='MODULE', default=[], action='append', help='OpenCV modules to exclude from the framework')
    parser.add_argument('--disable', metavar='FEATURE', default=[], action='append', help='OpenCV features to disable (add WITH_*=OFF)')
    parser.add_argument('--enable_nonfree', default=False, dest='enablenonfree', action='store_true', help='enable non-free modules (disabled by default)')
    parser.add_argument('--macosx_deployment_target', default=os.environ.get('MACOSX_DEPLOYMENT_TARGET', MACOSX_DEPLOYMENT_TARGET), help='specify MACOSX_DEPLOYMENT_TARGET')
    parser.add_argument('--debug', action='store_true', help='Build "Debug" binaries (CMAKE_BUILD_TYPE=Debug)')
    parser.add_argument('--debug_info', action='store_true', help='Build with debug information (useful for Release mode: BUILD_WITH_DEBUG_INFO=ON)')

    args = parser.parse_args()

    os.environ['MACOSX_DEPLOYMENT_TARGET'] = args.macosx_deployment_target
    print('Using MACOSX_DEPLOYMENT_TARGET=' + os.environ['MACOSX_DEPLOYMENT_TARGET'])

    b = OSXBuilder(args.opencv, args.contrib, False, False, args.without, args.disable, args.enablenonfree,
        [
            (["x86_64"], "MacOSX")
        ], args.debug, args.debug_info)
    b.build(args.out)
