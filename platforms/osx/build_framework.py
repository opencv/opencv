#!/usr/bin/env python
"""
The script builds OpenCV.framework for OSX.
"""

from __future__ import print_function
import os, os.path, sys, argparse, traceback, multiprocessing

# import common code
sys.path.insert(0, os.path.abspath(os.path.abspath(os.path.dirname(__file__))+'/../ios'))
from build_framework import Builder

class OSXBuilder(Builder):

    def getToolchain(self, arch, target):
        return None

    def getBuildCommand(self, archs, target):
        buildcmd = [
            "xcodebuild",
            "MACOSX_DEPLOYMENT_TARGET=10.9",
            "ARCHS=%s" % archs[0],
            "-sdk", target.lower(),
            "-configuration", "Release",
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
    parser.add_argument('--enable_nonfree', default=False, dest='enablenonfree', action='store_true', help='enable non-free modules (disabled by default)')
    args = parser.parse_args()

    b = OSXBuilder(args.opencv, args.contrib, False, False, args.without, args.enablenonfree,
        [
            (["x86_64"], "MacOSX")
        ])
    b.build(args.out)
