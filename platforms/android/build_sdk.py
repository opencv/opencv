#!/usr/bin/env python

import os, sys, subprocess, argparse, shutil, glob, re
import logging as log
import xml.etree.ElementTree as ET

class Fail(Exception):
    def __init__(self, text=None):
        self.t = text
    def __str__(self):
        return "ERROR" if self.t is None else self.t

def execute(cmd, shell=False, allowFail=False):
    try:
        log.info("Executing: %s" % cmd)
        retcode = subprocess.call(cmd, shell=shell)
        if retcode < 0:
            raise Fail("Child was terminated by signal:" %s -retcode)
        elif retcode > 0 and not allowFail:
            raise Fail("Child returned: %s" % retcode)
    except OSError as e:
        raise Fail("Execution failed: %d / %s" % (e.errno, e.strerror))

def rm_one(d):
    d = os.path.abspath(d)
    if os.path.exists(d):
        if os.path.isdir(d):
            log.info("Removing dir: %s", d)
            shutil.rmtree(d)
        elif os.path.isfile(d):
            log.info("Removing file: %s", d)
            os.remove(d)

def check_dir(d, create=False, clean=False):
    d = os.path.abspath(d)
    log.info("Check dir %s (create: %s, clean: %s)", d, create, clean)
    if os.path.exists(d):
        if not os.path.isdir(d):
            raise Fail("Not a directory: %s" % d)
        if clean:
            for x in glob.glob(os.path.join(d, "*")):
                rm_one(x)
    else:
        if create:
            os.makedirs(d)
    return d

def determine_opencv_version(version_hpp_path):
    # version in 2.4 - CV_VERSION_EPOCH.CV_VERSION_MAJOR.CV_VERSION_MINOR.CV_VERSION_REVISION
    # version in master - CV_VERSION_MAJOR.CV_VERSION_MINOR.CV_VERSION_REVISION-CV_VERSION_STATUS
    with open(version_hpp_path, "rt") as f:
        data = f.read()
        epoch = re.search(r'^#define\W+CV_VERSION_EPOCH\W+(\d+)$', data, re.MULTILINE).group(1)
        major = re.search(r'^#define\W+CV_VERSION_MAJOR\W+(\d+)$', data, re.MULTILINE).group(1)
        minor = re.search(r'^#define\W+CV_VERSION_MINOR\W+(\d+)$', data, re.MULTILINE).group(1)
        revision = re.search(r'^#define\W+CV_VERSION_REVISION\W+(\d+)$', data, re.MULTILINE).group(1)
        revision = '' if revision == '0' else '.' + revision
        return "%(epoch)s.%(major)s.%(minor)s%(revision)s" % locals()

#===================================================================================================

class ABI:
    def __init__(self, platform_id, name, toolchain, api_level=8, cmake_name=None):
        self.platform_id = platform_id # platform code to add to apk version (for cmake)
        self.name = name # general name (official Android ABI identifier)
        self.toolchain = toolchain # toolchain identifier (for cmake)
        self.api_level = api_level
        self.cmake_name = cmake_name # name of android toolchain (for cmake)
        if self.cmake_name is None:
            self.cmake_name = self.name
    def __str__(self):
        return "%s (%s)" % (self.name, self.toolchain)

ABIs = [
    ABI("2", "armeabi-v7a", "arm-linux-androideabi-4.6", cmake_name="armeabi-v7a with NEON"),
    ABI("1", "armeabi",     "arm-linux-androideabi-4.6"),
    ABI("4", "x86",         "x86-clang3.1", api_level=9),
    ABI("6", "mips",        "mipsel-linux-android-4.6", api_level=9)
]

#===================================================================================================

class Builder:
    def __init__(self, workdir, opencvdir):
        self.workdir = check_dir(workdir, create=True)
        self.opencvdir = check_dir(opencvdir)
        self.libdest = check_dir(os.path.join(self.workdir, "o4a"), create=True, clean=True)
        self.docdest = check_dir(os.path.join(self.workdir, "javadoc"), create=True, clean=True)
        self.resultdest = check_dir(os.path.join(self.workdir, "OpenCV-android-sdk"), create=True, clean=True)
        self.opencv_version = determine_opencv_version(os.path.join(self.opencvdir, "modules", "core", "include", "opencv2", "core", "version.hpp"))
        self.use_ccache = True

    def get_toolchain_file(self):
        return os.path.join(self.opencvdir, "platforms", "android", "android.toolchain.cmake")

    def clean_library_build_dir(self):
        for d in ["CMakeCache.txt", "CMakeFiles/", "bin/", "libs/", "lib/", "package/", "install/samples/"]:
            rm_one(d)

    def build_library(self, abi, do_install, build_docs):
        cmd = [
            "cmake",
            "-GNinja",
            "-DCMAKE_TOOLCHAIN_FILE='%s'" % self.get_toolchain_file(),
            "-DINSTALL_CREATE_DISTRIB=ON",
            #"-DWITH_OPENCL=OFF",
            "-DWITH_CUDA=OFF", "-DBUILD_opencv_gpu=OFF",
            "-DBUILD_opencv_nonfree=OFF",
            "-DWITH_TBB=OFF",
            "-DWITH_IPP=OFF",
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_PERF_TESTS=OFF",
            "-DBUILD_DOCS=OFF",
            "-DBUILD_ANDROID_EXAMPLES=ON",
            "-DINSTALL_ANDROID_EXAMPLES=ON",
            "-DANDROID_STL=gnustl_static",
            "-DANDROID_NATIVE_API_LEVEL=%s" % abi.api_level,
            "-DANDROID_ABI='%s'" % abi.cmake_name,
            "-DANDROID_TOOLCHAIN_NAME=%s" % abi.toolchain
        ]

        cmd.append(self.opencvdir)

        if self.use_ccache == True:
            cmd.append("-DNDK_CCACHE=ccache")
        if do_install:
            cmd.extend(["-DBUILD_TESTS=ON", "-DINSTALL_TESTS=ON"])
        if do_install and build_docs:
            cmd.extend(["-DBUILD_DOCS=ON"])
        execute(cmd)
        if do_install:
            execute(["cmake", "--build", "."])
            if do_install and build_docs:
                execute(["cmake", "--build", ".", "--target", "docs"])
            for c in ["libs", "dev", "java", "samples"] + (["docs"] if do_install and build_docs else []):
                execute(["cmake", "-DCOMPONENT=%s" % c, "-P", "cmake_install.cmake"])
        else:
            execute(["cmake", "--build", ".", "--target", "install/strip"])

    def build_javadoc(self):
        classpaths = [os.path.join(self.libdest, "bin", "classes")]
        for dir, _, files in os.walk(os.environ["ANDROID_SDK"]):
            for f in files:
                if f == "android.jar" or f == "annotations.jar":
                    classpaths.append(os.path.join(dir, f))
        cmd = [
            "javadoc",
            "-encoding", "UTF-8",
            "-header", "OpenCV %s" % self.opencv_version,
            "-nodeprecated",
            "-footer", '<a href="http://docs.opencv.org">OpenCV %s Documentation</a>' % self.opencv_version,
            "-public",
            "-sourcepath", os.path.join(self.libdest, "src"),
            "-d", self.docdest,
            "-classpath", ":".join(classpaths)
        ]
        for _, dirs, _ in os.walk(os.path.join(self.libdest, "src", "org", "opencv")):
            cmd.extend(["org.opencv." + d for d in dirs])
        execute(cmd, allowFail=True) # FIXIT javadoc currenly reports some errors

    def gather_results(self, with_samples_apk):
        # Copy all files
        root = os.path.join(self.libdest, "install")
        for item in os.listdir(root):
            name = item
            item = os.path.join(root, item)
            if os.path.isdir(item):
                log.info("Copy dir: %s", item)
                shutil.copytree(item, os.path.join(self.resultdest, name))
            elif os.path.isfile(item):
                log.info("Copy file: %s", item)
                shutil.copy2(item, os.path.join(self.resultdest, name))

        # Copy javadoc
        log.info("Copy docs: %s", self.docdest)
        shutil.copytree(self.docdest, os.path.join(self.resultdest, "sdk", "java", "javadoc"))

        # Clean samples
        path = os.path.join(self.resultdest, "samples")
        for item in os.listdir(path):
            item = os.path.join(path, item)
            if os.path.isdir(item):
                for name in ["build.xml", "local.properties", "proguard-project.txt"]:
                    rm_one(os.path.join(item, name))
            if not with_samples_apk:
                if re.search(r'\.apk$', item):  # reduce size of SDK
                    rm_one(item)


#===================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build OpenCV for Android SDK')
    parser.add_argument("work_dir", help="Working directory (and output)")
    parser.add_argument("opencv_dir", help="Path to OpenCV source dir")
    parser.add_argument('--ndk_path', help="Path to Android NDK to use for build")
    parser.add_argument('--sdk_path', help="Path to Android SDK to use for build")
    parser.add_argument('--build_doc', action="store_true", help="Build documentation")
    parser.add_argument('--build_javadoc', action="store_true", help="Build javadoc")
    parser.add_argument('--no_ccache', action="store_true", help="Do not use ccache during library build")
    parser.add_argument('--with_samples_apk', action="store_true", help="Include samples APKs")
    args = parser.parse_args()

    log.basicConfig(format='%(message)s', level=log.DEBUG)
    log.debug("Args: %s", args)

    if args.ndk_path is not None:
        os.environ["ANDROID_NDK"] = args.ndk_path
    if args.sdk_path is not None:
        os.environ["ANDROID_SDK"] = args.sdk_path

    log.info("Android NDK path: %s", os.environ["ANDROID_NDK"])
    log.info("Android SDK path: %s", os.environ["ANDROID_SDK"])

    builder = Builder(args.work_dir, args.opencv_dir)

    if args.no_ccache:
        builder.use_ccache = False

    log.info("Detected OpenCV version: %s", builder.opencv_version)

    for i, abi in enumerate(ABIs):
        do_install = (i == 0)

        log.info("=====")
        log.info("===== Building library for %s", abi)
        log.info("=====")

        os.chdir(builder.libdest)
        builder.clean_library_build_dir()
        builder.build_library(abi, do_install, build_docs=args.build_doc)

    if args.build_doc or args.build_javadoc:
        builder.build_javadoc()

    builder.gather_results(with_samples_apk=args.with_samples_apk)

    log.info("=====")
    log.info("===== Build finished")
    log.info("=====")
    log.info("SDK location: %s", builder.resultdest)
    log.info("Documentation location: %s", builder.docdest)
