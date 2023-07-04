#!/usr/bin/env python
"""
This script builds OpenCV docs for iOS.
"""

from __future__ import print_function
import os, sys, multiprocessing, argparse, traceback
from subprocess import check_call, check_output, CalledProcessError, Popen

def execute(cmd, cwd = None, output = None):
    if not output:
        print("Executing: %s in %s" % (cmd, cwd), file=sys.stderr)
        print('Executing: ' + ' '.join(cmd))
        retcode = check_call(cmd, cwd = cwd)
        if retcode != 0:
            raise Exception("Child returned:", retcode)
    else:
        with open(output, "a") as f:
            f.flush()
            p = Popen(cmd, cwd = cwd, stdout = f)
            os.waitpid(p.pid, 0)

class DocBuilder:
    def __init__(self, script_dir, framework_dir, output_dir, framework_header, framework_name, arch, target):
        self.script_dir = script_dir
        self.framework_dir = framework_dir
        self.output_dir = output_dir
        self.framework_header = framework_header
        self.framework_name = framework_name
        self.arch = arch
        self.target = target

    def _build(self):
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        self.buildDocs()

    def build(self):
        try:
            self._build()
        except Exception as e:
            print("="*60, file=sys.stderr)
            print("ERROR: %s" % e, file=sys.stderr)
            print("="*60, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

    def getToolchain(self):
        return None

    def getSourceKitten(self):
        ret = check_output(["gem", "which", "jazzy"])
        if ret.find('ERROR:') == 0:
            raise Exception("Failed to find jazzy")
        else:
            return os.path.join(ret[0:ret.rfind('/')], '../bin/sourcekitten')

    def buildDocs(self):
        sourceKitten = self.getSourceKitten()
        sourceKittenSwiftDoc = [sourceKitten, "doc", "--module-name", self.framework_name, "--", "-project", self.framework_name + ".xcodeproj", "ARCHS=" + self.arch, "-sdk", self.target, "-configuration", "Release", "-parallelizeTargets", "-jobs", str(multiprocessing.cpu_count()), "-target", "opencv_objc_framework"]
        execute(sourceKittenSwiftDoc, cwd = self.framework_dir, output = os.path.join(self.output_dir, "swiftDoc.json"))
        sdk_dir = check_output(["xcrun", "--show-sdk-path", "--sdk", self.target]).rstrip()
        sourceKittenObjcDoc = [sourceKitten, "doc", "--objc", self.framework_header, "--", "-x", "objective-c", "-isysroot", sdk_dir, "-fmodules"]
        print(sourceKittenObjcDoc)
        execute(sourceKittenObjcDoc, cwd = self.framework_dir, output = os.path.join(self.output_dir, "objcDoc.json"))
        execute(["jazzy", "--author", "OpenCV", "--author_url", "http://opencv.org", "--github_url", "https://github.com/opencv/opencv", "--module", self.framework_name, "--undocumented-text", "\"\"", "--sourcekitten-sourcefile", "swiftDoc.json,objcDoc.json"], cwd = self.output_dir)

class iOSDocBuilder(DocBuilder):

    def getToolchain(self):
        return None

if __name__ == "__main__":
    script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
    parser = argparse.ArgumentParser(description='The script builds OpenCV docs for iOS.')
    parser.add_argument('framework_dir', metavar='FRAMEWORK_DIR', help='folder where framework build files are located')
    parser.add_argument('--output_dir', default=None, help='folder where docs will be built (default is "../doc_build" relative to framework_dir)')
    parser.add_argument('--framework_header', default=None, help='umbrella header for OpenCV framework (default is "../../../lib/Release/{framework_name}.framework/Headers/{framework_name}.h")')
    parser.add_argument('--framework_name', default='opencv2', help='Name of OpenCV framework (default: opencv2, will change to OpenCV in future version)')
    args = parser.parse_args()

    arch = "x86_64"
    target = "iphonesimulator"

    b = iOSDocBuilder(script_dir, args.framework_dir, args.output_dir if args.output_dir else os.path.join(args.framework_dir, "../doc_build"), args.framework_header if args.framework_header else os.path.join(args.framework_dir, "../../../lib/Release/" + args.framework_name + ".framework/Headers/" + args.framework_name + ".h"), args.framework_name, arch, target)
    b.build()
