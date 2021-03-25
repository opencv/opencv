#!/usr/bin/env python
"""
This script runs OpenCV.framework tests for iOS.
"""

from __future__ import print_function
import glob, re, os, os.path, shutil, string, sys, argparse, traceback, multiprocessing
from subprocess import check_call, check_output, CalledProcessError

IPHONEOS_DEPLOYMENT_TARGET='9.0'  # default, can be changed via command line options or environment variable

def execute(cmd, cwd = None):
    print("Executing: %s in %s" % (cmd, cwd), file=sys.stderr)
    print('Executing: ' + ' '.join(cmd))
    retcode = check_call(cmd, cwd = cwd)
    if retcode != 0:
        raise Exception("Child returned:", retcode)

class TestRunner:
    def __init__(self, script_dir, tests_dir, build_dir, framework_dir, framework_name, arch, target, platform):
        self.script_dir = script_dir
        self.tests_dir = tests_dir
        self.build_dir = build_dir
        self.framework_dir = framework_dir
        self.framework_name = framework_name
        self.arch = arch
        self.target = target
        self.platform = platform

    def _run(self):
        if not os.path.isdir(self.build_dir):
            os.makedirs(self.build_dir)

        self.runTest()

    def run(self):
        try:
            self._run()
        except Exception as e:
            print("="*60, file=sys.stderr)
            print("ERROR: %s" % e, file=sys.stderr)
            print("="*60, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

    def getToolchain(self):
        return None

    def getCMakeArgs(self):
        args = [
            "cmake",
            "-GXcode",
            "-DFRAMEWORK_DIR=%s" % self.framework_dir,
            "-DFRAMEWORK_NAME=%s" % self.framework_name,
        ]
        return args

    def makeCMakeCmd(self):
        toolchain = self.getToolchain()
        cmakecmd = self.getCMakeArgs() + \
            (["-DCMAKE_TOOLCHAIN_FILE=%s" % toolchain] if toolchain is not None else []) + \
            ["-DCMAKE_INSTALL_NAME_TOOL=install_name_tool"]
        cmakecmd.append(self.tests_dir)
        return cmakecmd

    def runTest(self):
        cmakecmd = self.makeCMakeCmd()
        execute(cmakecmd, cwd = self.build_dir)
        buildcmd = self.getTestCommand()
        execute(buildcmd, cwd = self.build_dir)

    def getTestCommand(self):
        testcmd = [
            "xcodebuild",
            "test",
            "-project", "OpenCVTest.xcodeproj",
            "-scheme", "OpenCVTestTests",
            "-destination", "platform=%s" % self.platform
        ]
        return testcmd

class iOSTestRunner(TestRunner):

    def getToolchain(self):
        toolchain = os.path.join(self.script_dir, "cmake", "Toolchains", "Toolchain-%s_Xcode.cmake" % self.target)
        return toolchain

    def getCMakeArgs(self):
        args = TestRunner.getCMakeArgs(self)
        args = args + [
            "-DIOS_ARCH=%s" % self.arch,
            "-DIPHONEOS_DEPLOYMENT_TARGET=%s" % os.environ['IPHONEOS_DEPLOYMENT_TARGET'],
        ]
        return args

if __name__ == "__main__":
    script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
    parser = argparse.ArgumentParser(description='The script builds OpenCV.framework for iOS.')
    parser.add_argument('tests_dir', metavar='TEST_DIR', help='folder where test files are located')
    parser.add_argument('--build_dir', default=None, help='folder where test will be built (default is "../test_build" relative to tests_dir)')
    parser.add_argument('--framework_dir', default=None, help='folder where OpenCV framework is located')
    parser.add_argument('--framework_name', default='opencv2', help='Name of OpenCV framework (default: opencv2, will change to OpenCV in future version)')
    parser.add_argument('--iphoneos_deployment_target', default=os.environ.get('IPHONEOS_DEPLOYMENT_TARGET', IPHONEOS_DEPLOYMENT_TARGET), help='specify IPHONEOS_DEPLOYMENT_TARGET')
    parser.add_argument('--platform', default='iOS Simulator,name=iPhone 11', help='xcodebuild platform parameter (default is iOS 11 simulator)')
    args = parser.parse_args()

    os.environ['IPHONEOS_DEPLOYMENT_TARGET'] = args.iphoneos_deployment_target
    print('Using IPHONEOS_DEPLOYMENT_TARGET=' + os.environ['IPHONEOS_DEPLOYMENT_TARGET'])
    arch = "x86_64"
    target = "iPhoneSimulator"
    print('Using iPhoneSimulator ARCH=' + arch)

    r = iOSTestRunner(script_dir, args.tests_dir, args.build_dir if args.build_dir else os.path.join(args.tests_dir, "../test_build"), args.framework_dir, args.framework_name, arch, target, args.platform)
    r.run()
