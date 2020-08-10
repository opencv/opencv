#!/usr/bin/env python
"""
This script builds OpenCV docs for macOS.
"""

from __future__ import print_function
import os, sys, multiprocessing, argparse, traceback
from subprocess import check_call, check_output, CalledProcessError, Popen

# import common code
sys.path.insert(0, os.path.abspath(os.path.abspath(os.path.dirname(__file__))+'/../ios'))
from build_docs import DocBuilder

class OSXDocBuilder(DocBuilder):

    def getToolchain(self):
        return None

if __name__ == "__main__":
    script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
    parser = argparse.ArgumentParser(description='The script builds OpenCV docs for macOS.')
    parser.add_argument('framework_dir', metavar='FRAMEWORK_DIR', help='folder where framework build files are located')
    parser.add_argument('--output_dir', default=None, help='folder where docs will be built (default is "../doc_build" relative to framework_dir)')
    parser.add_argument('--framework_header', default=None, help='umbrella header for OpenCV framework (default is "../../../lib/Release/{framework_name}.framework/Headers/{framework_name}.h")')
    parser.add_argument('--framework_name', default='opencv2', help='Name of OpenCV framework (default: opencv2, will change to OpenCV in future version)')

    args = parser.parse_args()
    arch = "x86_64"
    target = "macosx"

    b = OSXDocBuilder(script_dir, args.framework_dir, args.output_dir if args.output_dir else os.path.join(args.framework_dir, "../doc_build"), args.framework_header if args.framework_header else os.path.join(args.framework_dir, "../../../lib/Release/" + args.framework_name + ".framework/Headers/" + args.framework_name + ".h"), args.framework_name, arch, target)
    b.build()
