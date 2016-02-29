#!/usr/bin/env python
"""
The script builds OpenCV.framework for iOS.
The built framework is universal, it can be used to build app and run it on either iOS simulator or real device.

Usage:
    ./build_framework.py <outputdir>

By cmake conventions (and especially if you work with OpenCV repository),
the output dir should not be a subdirectory of OpenCV source tree.

Script will create <outputdir>, if it's missing, and a few its subdirectories:

    <outputdir>
        build/
            iPhoneOS-*/
               [cmake-generated build tree for an iOS device target]
            iPhoneSimulator-*/
               [cmake-generated build tree for iOS simulator]
        opencv2.framework/
            [the framework content]

The script should handle minor OpenCV updates efficiently
- it does not recompile the library from scratch each time.
However, opencv2.framework directory is erased and recreated on each run.
"""

from __future__ import print_function
import glob, re, os, os.path, shutil, string, sys, argparse, traceback
from subprocess import check_call, check_output, CalledProcessError

def execute(cmd, cwd = None):
    print("Executing: %s in %s" % (cmd, cwd), file=sys.stderr)
    retcode = check_call(cmd, cwd = cwd)
    if retcode != 0:
        raise Exception("Child returned:", retcode)

def getXCodeMajor():
    ret = check_output(["xcodebuild", "-version"])
    m = re.match(r'XCode\s+(\d)\..*', ret, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0

class Builder:
    def __init__(self, opencv, contrib, targets):
        self.opencv = os.path.abspath(opencv)
        self.contrib = None
        if contrib:
            modpath = os.path.join(contrib, "modules")
            if os.path.isdir(modpath):
                self.contrib = os.path.abspath(modpath)
            else:
                print("Note: contrib repository is bad - modules subfolder not found", file=sys.stderr)
        self.targets = targets

    def getBD(self, parent, t):
        res = os.path.join(parent, '%s-%s' % t)
        if not os.path.isdir(res):
            os.makedirs(res)
        return os.path.abspath(res)

    def _build(self, outdir):
        outdir = os.path.abspath(outdir)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        mainWD = os.path.join(outdir, "build")
        dirs = []

        xcode_ver = getXCodeMajor()

        for t in self.targets:
            mainBD = self.getBD(mainWD, t)
            dirs.append(mainBD)
            cmake_flags = []
            if self.contrib:
                cmake_flags.append("-DOPENCV_EXTRA_MODULES_PATH=%s" % self.contrib)
            if xcode_ver >= 7 and t[1] == 'iPhoneOS':
                cmake_flags.append("-DCMAKE_C_FLAGS=-fembed-bitcode")
                cmake_flags.append("-DCMAKE_CXX_FLAGS=-fembed-bitcode")
            self.buildOne(t[0], t[1], mainBD, cmake_flags)
            self.mergeLibs(mainBD)
        self.makeFramework(outdir, dirs)

    def build(self, outdir):
        try:
            self._build(outdir)
        except Exception as e:
            print("="*60, file=sys.stderr)
            print("ERROR: %s" % e, file=sys.stderr)
            print("="*60, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

    def getToolchain(self, arch, target):
        toolchain = os.path.join(self.opencv, "platforms", "ios", "cmake", "Toolchains", "Toolchain-%s_Xcode.cmake" % target)
        return toolchain

    def getCMakeArgs(self, arch, target):
        args = [
            "cmake",
            "-GXcode",
            "-DAPPLE_FRAMEWORK=ON",
            "-DCMAKE_INSTALL_PREFIX=install",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        return args

    def getBuildCommand(self, arch, target):
        buildcmd = [
            "xcodebuild",
            "IPHONEOS_DEPLOYMENT_TARGET=6.0",
            "ARCHS=%s" % arch,
            "-sdk", target.lower(),
            "-configuration", "Release",
            "-parallelizeTargets",
            "-jobs", "4"
        ]
        return buildcmd

    def getInfoPlist(self, builddirs):
        return os.path.join(builddirs[0], "ios", "Info.plist")

    def buildOne(self, arch, target, builddir, cmakeargs = []):
        # Run cmake
        toolchain = self.getToolchain(arch, target)
        cmakecmd = self.getCMakeArgs(arch, target) + \
            (["-DCMAKE_TOOLCHAIN_FILE=%s" % toolchain] if toolchain is not None else [])
        if arch.startswith("armv") or arch.startswith("arm64"):
            cmakecmd.append("-DENABLE_NEON=ON")
        cmakecmd.append(self.opencv)
        cmakecmd.extend(cmakeargs)
        execute(cmakecmd, cwd = builddir)
        # Clean and build
        clean_dir = os.path.join(builddir, "install")
        if os.path.isdir(clean_dir):
            shutil.rmtree(clean_dir)
        buildcmd = self.getBuildCommand(arch, target)
        execute(buildcmd + ["-target", "ALL_BUILD", "build"], cwd = builddir)
        execute(["cmake", "-P", "cmake_install.cmake"], cwd = builddir)

    def mergeLibs(self, builddir):
        res = os.path.join(builddir, "lib", "Release", "libopencv_merged.a")
        libs = glob.glob(os.path.join(builddir, "install", "lib", "*.a"))
        libs3 = glob.glob(os.path.join(builddir, "install", "share", "OpenCV", "3rdparty", "lib", "*.a"))
        print("Merging libraries:\n\t%s" % "\n\t".join(libs + libs3), file=sys.stderr)
        execute(["libtool", "-static", "-o", res] + libs + libs3)

    def makeFramework(self, outdir, builddirs):
        name = "opencv2"
        libname = "libopencv_merged.a"

        # set the current dir to the dst root
        framework_dir = os.path.join(outdir, "%s.framework" % name)
        if os.path.isdir(framework_dir):
            shutil.rmtree(framework_dir)
        os.makedirs(framework_dir)

        dstdir = os.path.join(framework_dir, "Versions", "A")

        # copy headers from one of build folders
        shutil.copytree(os.path.join(builddirs[0], "install", "include", "opencv2"), os.path.join(dstdir, "Headers"))

        # make universal static lib
        libs = [os.path.join(d, "lib", "Release", libname) for d in builddirs]
        lipocmd = ["lipo", "-create"]
        lipocmd.extend(libs)
        lipocmd.extend(["-o", os.path.join(dstdir, name)])
        print("Creating universal library from:\n\t%s" % "\n\t".join(libs), file=sys.stderr)
        execute(lipocmd)

        # copy Info.plist
        resdir = os.path.join(dstdir, "Resources")
        os.makedirs(resdir)
        shutil.copyfile(self.getInfoPlist(builddirs), os.path.join(resdir, "Info.plist"))

        # make symbolic links
        links = [
            (["A"], ["Versions", "Current"]),
            (["Versions", "Current", "Headers"], ["Headers"]),
            (["Versions", "Current", "Resources"], ["Resources"]),
            (["Versions", "Current", name], [name])
        ]
        for l in links:
            s = os.path.join(*l[0])
            d = os.path.join(framework_dir, *l[1])
            os.symlink(s, d)

if __name__ == "__main__":
    folder = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), "../.."))
    parser = argparse.ArgumentParser(description='The script builds OpenCV.framework for iOS.')
    parser.add_argument('out', metavar='OUTDIR', help='folder to put built framework')
    parser.add_argument('--opencv', metavar='DIR', default=folder, help='folder with opencv repository (default is "../.." relative to script location)')
    parser.add_argument('--contrib', metavar='DIR', default=None, help='folder with opencv_contrib repository (default is "None" - build only main framework)')
    args = parser.parse_args()

    b = Builder(args.opencv, args.contrib,
        [
            ("armv7", "iPhoneOS"),
            ("armv7s", "iPhoneOS"),
            ("arm64", "iPhoneOS"),
            ("i386", "iPhoneSimulator"),
            ("x86_64", "iPhoneSimulator"),
        ])
    b.build(args.out)
