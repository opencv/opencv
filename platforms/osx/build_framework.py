#!/usr/bin/env python
"""
The script builds OpenCV.framework for OSX.
"""

from __future__ import print_function
import os, os.path, sys, argparse, traceback

# import common code
sys.path.insert(0, os.path.abspath(os.path.abspath(os.path.dirname(__file__))+'/../ios'))
from build_framework import Builder

class OSXBuilder(Builder):

    def getToolchain(self, arch, target):
        return None

    def getBuildCommand(self, arch, target):
        buildcmd = [
            "xcodebuild",
            "ARCHS=%s" % arch,
            "-sdk", target.lower(),
            "-configuration", "Release",
            "-parallelizeTargets",
            "-jobs", "4"
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
    args = parser.parse_args()

    b = OSXBuilder(args.opencv, args.contrib,
        [
            ("x86_64", "MacOSX")
        ])
    b.build(args.out)
