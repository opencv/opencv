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

import glob, re, os, os.path, shutil, string, sys, exceptions, subprocess, argparse

opencv_contrib_path = None

def execute(cmd):
    try:
        print >>sys.stderr, "Executing:", cmd
        retcode = subprocess.call(cmd, shell=True)
        if retcode < 0:
            raise Exception("Child was terminated by signal:", -retcode)
        elif retcode > 0:
            raise Exception("Child returned:", retcode)
    except OSError as e:
        raise Exception("Execution failed:", e)

def build_opencv(srcroot, buildroot, target, arch):
    "builds OpenCV for device or simulator"

    builddir = os.path.join(buildroot, target + '-' + arch)
    if not os.path.isdir(builddir):
        os.makedirs(builddir)
    currdir = os.getcwd()
    os.chdir(builddir)
    # for some reason, if you do not specify CMAKE_BUILD_TYPE, it puts libs to "RELEASE" rather than "Release"
    cmakeargs = ("-GXcode " +
                "-DCMAKE_BUILD_TYPE=Release " +
                "-DCMAKE_TOOLCHAIN_FILE=%s/platforms/ios/cmake/Toolchains/Toolchain-%s_Xcode.cmake " +
                "-DCMAKE_C_FLAGS=\"-Wno-implicit-function-declaration\" " +
                "-DCMAKE_INSTALL_PREFIX=install") % (srcroot, target)

    if arch.startswith("armv"):
        cmakeargs += " -DENABLE_NEON=ON"

    if opencv_contrib_path is not None:
        cmakeargs += " -DCMAKE_SKIP_INSTALL_ALL_DEPENDENCY=ON -DOPENCV_EXTRA_MODULES_PATH=%s -DBUILD_opencv_contrib_world=ON" % opencv_contrib_path
        build_target = "opencv_contrib_world"
        libname = "libopencv_contrib_world.a"
    else:
        cmakeargs += " -DBUILD_opencv_world=ON"
        build_target = "ALL_BUILD"
        libname = "libopencv_world.a"

    # if cmake cache exists, just rerun cmake to update OpenCV.xcodeproj if necessary
    if os.path.isfile(os.path.join(builddir, "CMakeCache.txt")):
        execute("cmake %s ." % (cmakeargs,))
    else:
        execute("cmake %s %s" % (cmakeargs, srcroot))

    for wlib in [builddir + "/modules/world/UninstalledProducts/" + libname,
                 builddir + "/lib/Release/" + libname]:
        if os.path.isfile(wlib):
            os.remove(wlib)

    execute("xcodebuild IPHONEOS_DEPLOYMENT_TARGET=6.0 -parallelizeTargets ARCHS=%s -jobs 8 -sdk %s -configuration Release -target %s" % (arch, target.lower(), build_target))
    execute("xcodebuild IPHONEOS_DEPLOYMENT_TARGET=6.0 ARCHS=%s -sdk %s -configuration Release -target install install" % (arch, target.lower()))
    os.chdir(currdir)

def put_framework_together(srcroot, dstroot):
    "constructs the framework directory after all the targets are built"

    name = "opencv2" if opencv_contrib_path is None else "opencv2_contrib"
    libname = "libopencv_world.a" if opencv_contrib_path is None else "libopencv_contrib_world.a"

    # find the list of targets (basically, ["iPhoneOS", "iPhoneSimulator"])
    targetlist = glob.glob(os.path.join(dstroot, "build", "*"))
    targetlist = [os.path.basename(t) for t in targetlist]

    # set the current dir to the dst root
    currdir = os.getcwd()
    framework_dir = dstroot + "/%s.framework" % name
    if os.path.isdir(framework_dir):
        shutil.rmtree(framework_dir)
    os.makedirs(framework_dir)
    os.chdir(framework_dir)

    # form the directory tree
    dstdir = "Versions/A"
    os.makedirs(dstdir + "/Resources")

    tdir0 = "../build/" + targetlist[0]
    # copy headers
    shutil.copytree(tdir0 + "/install/include/opencv2", dstdir + "/Headers")

    # make universal static lib
    wlist = " ".join(["../build/" + t + "/lib/Release/" + libname for t in targetlist])
    execute("lipo -create " + wlist + " -o " + dstdir + "/%s" % name)

    # copy Info.plist
    shutil.copyfile(tdir0 + "/ios/Info.plist", dstdir + "/Resources/Info.plist")

    # make symbolic links
    os.symlink("A", "Versions/Current")
    os.symlink("Versions/Current/Headers", "Headers")
    os.symlink("Versions/Current/Resources", "Resources")
    os.symlink("Versions/Current/%s" % name, name)


def build_framework(srcroot, dstroot):
    "main function to do all the work"

    targets = [("armv7", "iPhoneOS"),
               ("armv7s", "iPhoneOS"),
               ("arm64", "iPhoneOS"),
               ("i386", "iPhoneSimulator"),
               ("x86_64", "iPhoneSimulator")]
    for t in targets:
        build_opencv(srcroot, os.path.join(dstroot, "build"), t[1], t[0])

    put_framework_together(srcroot, dstroot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The script builds OpenCV.framework for iOS.')
    parser.add_argument('outputdir', nargs=1, help='folder to put built framework')
    parser.add_argument('--contrib', help="folder with opencv_contrib repository")
    args = parser.parse_args()

    # path to OpenCV main repository - hardcoded ../..
    opencv_path = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), "../.."))
    print "OpenCV:", opencv_path

    # path to OpenCV_contrib repository, can be empty - global variable
    if hasattr(args, "contrib") and args.contrib is not None:
        if os.path.isdir(args.contrib + "/modules"):
            opencv_contrib_path = os.path.abspath(args.contrib + "/modules")
            print "Contrib:", opencv_contrib_path
        else:
            print "Note: contrib repository is bad: modules subfolder not found"

    # result path - folder where framework will be located
    output_path = os.path.abspath(args.outputdir[0])
    print "Output:", output_path

    try:
        build_framework(opencv_path, output_path)
    except Exception as e:
        print >>sys.stderr, e
        sys.exit(1)
