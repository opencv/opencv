#!/usr/bin/env python

import os, sys, subprocess, argparse, shutil, glob, re
import logging as log
import xml.etree.ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class Fail(Exception):
    def __init__(self, text=None):
        self.t = text
    def __str__(self):
        return "ERROR" if self.t is None else self.t

def execute(cmd, shell=False):
    try:
        log.debug("Executing: %s" % cmd)
        log.info('Executing: ' + ' '.join(cmd))
        retcode = subprocess.call(cmd, shell=shell)
        if retcode < 0:
            raise Fail("Child was terminated by signal: %s" % -retcode)
        elif retcode > 0:
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
        major = re.search(r'^#define\W+CV_VERSION_MAJOR\W+(\d+)$', data, re.MULTILINE).group(1)
        minor = re.search(r'^#define\W+CV_VERSION_MINOR\W+(\d+)$', data, re.MULTILINE).group(1)
        revision = re.search(r'^#define\W+CV_VERSION_REVISION\W+(\d+)$', data, re.MULTILINE).group(1)
        version_status = re.search(r'^#define\W+CV_VERSION_STATUS\W+"([^"]*)"$', data, re.MULTILINE).group(1)
        return "%(major)s.%(minor)s.%(revision)s%(version_status)s" % locals()

# shutil.move fails if dst exists
def move_smart(src, dst):
    def move_recurse(subdir):
        s = os.path.join(src, subdir)
        d = os.path.join(dst, subdir)
        if os.path.exists(d):
            if os.path.isdir(d):
                for item in os.listdir(s):
                    move_recurse(os.path.join(subdir, item))
            elif os.path.isfile(s):
                shutil.move(s, d)
        else:
            shutil.move(s, d)
    move_recurse('')

# shutil.copytree fails if dst exists
def copytree_smart(src, dst):
    def copy_recurse(subdir):
        s = os.path.join(src, subdir)
        d = os.path.join(dst, subdir)
        if os.path.exists(d):
            if os.path.isdir(d):
                for item in os.listdir(s):
                    copy_recurse(os.path.join(subdir, item))
            elif os.path.isfile(s):
                shutil.copy2(s, d)
        else:
            if os.path.isdir(s):
                shutil.copytree(s, d)
            elif os.path.isfile(s):
                shutil.copy2(s, d)
    copy_recurse('')

#===================================================================================================

class ABI:
    def __init__(self, platform_id, name, toolchain, ndk_api_level = None, cmake_vars = dict()):
        self.platform_id = platform_id # platform code to add to apk version (for cmake)
        self.name = name # general name (official Android ABI identifier)
        self.toolchain = toolchain # toolchain identifier (for cmake)
        self.cmake_vars = dict(
            ANDROID_STL="gnustl_static",
            ANDROID_ABI=self.name,
            ANDROID_PLATFORM_ID=platform_id,
        )
        if toolchain is not None:
            self.cmake_vars['ANDROID_TOOLCHAIN_NAME'] = toolchain
        else:
            self.cmake_vars['ANDROID_TOOLCHAIN'] = 'clang'
            self.cmake_vars['ANDROID_STL'] = 'c++_static'
        if ndk_api_level:
            self.cmake_vars['ANDROID_NATIVE_API_LEVEL'] = ndk_api_level
        self.cmake_vars.update(cmake_vars)
    def __str__(self):
        return "%s (%s)" % (self.name, self.toolchain)
    def haveIPP(self):
        return self.name == "x86" or self.name == "x86_64"

#===================================================================================================

class Builder:
    def __init__(self, workdir, opencvdir, config):
        self.workdir = check_dir(workdir, create=True)
        self.opencvdir = check_dir(opencvdir)
        self.config = config
        self.libdest = check_dir(os.path.join(self.workdir, "o4a"), create=True, clean=True)
        self.resultdest = check_dir(os.path.join(self.workdir, 'OpenCV-android-sdk'), create=True, clean=True)
        self.docdest = check_dir(os.path.join(self.workdir, 'OpenCV-android-sdk', 'sdk', 'java', 'javadoc'), create=True, clean=True)
        self.extra_packs = []
        self.opencv_version = determine_opencv_version(os.path.join(self.opencvdir, "modules", "core", "include", "opencv2", "core", "version.hpp"))
        self.use_ccache = False if config.no_ccache else True

    def get_toolchain_file(self):
        if not self.config.force_opencv_toolchain:
            toolchain = os.path.join(os.environ['ANDROID_NDK'], 'build', 'cmake', 'android.toolchain.cmake')
            if os.path.exists(toolchain):
                return toolchain
        toolchain = os.path.join(SCRIPT_DIR, "android.toolchain.cmake")
        if os.path.exists(toolchain):
            return toolchain
        else:
            raise Fail("Can't find toolchain")

    def get_engine_apk_dest(self, engdest):
        return os.path.join(engdest, "platforms", "android", "service", "engine", ".build")

    def add_extra_pack(self, ver, path):
        if path is None:
            return
        self.extra_packs.append((ver, check_dir(path)))

    def clean_library_build_dir(self):
        for d in ["CMakeCache.txt", "CMakeFiles/", "bin/", "libs/", "lib/", "package/", "install/samples/"]:
            rm_one(d)

    def build_library(self, abi, do_install):
        cmd = ["cmake", "-GNinja"]
        cmake_vars = dict(
            CMAKE_TOOLCHAIN_FILE=self.get_toolchain_file(),
            INSTALL_CREATE_DISTRIB="ON",
            WITH_OPENCL="OFF",
            WITH_IPP=("ON" if abi.haveIPP() else "OFF"),
            WITH_TBB="ON",
            BUILD_EXAMPLES="OFF",
            BUILD_TESTS="OFF",
            BUILD_PERF_TESTS="OFF",
            BUILD_DOCS="OFF",
            BUILD_ANDROID_EXAMPLES="ON",
            INSTALL_ANDROID_EXAMPLES="ON",
        )

        if self.config.extra_modules_path is not None:
            cmd.append("-DOPENCV_EXTRA_MODULES_PATH='%s'" % self.config.extra_modules_path)

        if self.use_ccache == True:
            cmd.append("-DNDK_CCACHE=ccache")
        if do_install:
            cmd.extend(["-DBUILD_TESTS=ON", "-DINSTALL_TESTS=ON"])

        cmake_vars.update(abi.cmake_vars)
        cmd += [ "-D%s='%s'" % (k, v) for (k, v) in cmake_vars.items() if v is not None]
        cmd.append(self.opencvdir)
        execute(cmd)
        execute(["ninja", "install/strip"])

    def build_javadoc(self):
        classpaths = []
        for dir, _, files in os.walk(os.environ["ANDROID_SDK"]):
            for f in files:
                if f == "android.jar" or f == "annotations.jar":
                    classpaths.append(os.path.join(dir, f))
        cmd = [
            "javadoc",
            "-header", "OpenCV %s" % self.opencv_version,
            "-nodeprecated",
            "-footer", '<a href="http://docs.opencv.org">OpenCV %s Documentation</a>' % self.opencv_version,
            "-public",
            '-sourcepath', os.path.join(self.resultdest, 'sdk', 'java', 'src'),
            "-d", self.docdest,
            "-classpath", ":".join(classpaths),
            '-subpackages', 'org.opencv',
        ]
        execute(cmd)

    def gather_results(self):
        # Copy all files
        root = os.path.join(self.libdest, "install")
        for item in os.listdir(root):
            src = os.path.join(root, item)
            dst = os.path.join(self.resultdest, item)
            if os.path.isdir(src):
                log.info("Copy dir: %s", item)
                if self.config.force_copy:
                    copytree_smart(src, dst)
                else:
                    move_smart(src, dst)
            elif os.path.isfile(src):
                log.info("Copy file: %s", item)
                if self.config.force_copy:
                    shutil.copy2(src, dst)
                else:
                    shutil.move(src, dst)


#===================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build OpenCV for Android SDK')
    parser.add_argument("work_dir", nargs='?', default='.', help="Working directory (and output)")
    parser.add_argument("opencv_dir", nargs='?', default=os.path.join(SCRIPT_DIR, '../..'), help="Path to OpenCV source dir")
    parser.add_argument('--config', default='ndk-18.config.py', type=str, help="Package build configuration", )
    parser.add_argument('--ndk_path', help="Path to Android NDK to use for build")
    parser.add_argument('--sdk_path', help="Path to Android SDK to use for build")
    parser.add_argument("--extra_modules_path", help="Path to extra modules to use for build")
    parser.add_argument('--sign_with', help="Certificate to sign the Manager apk")
    parser.add_argument('--build_doc', action="store_true", help="Build javadoc")
    parser.add_argument('--no_ccache', action="store_true", help="Do not use ccache during library build")
    parser.add_argument('--force_copy', action="store_true", help="Do not use file move during library build (useful for debug)")
    parser.add_argument('--force_opencv_toolchain', action="store_true", help="Do not use toolchain from Android NDK")
    args = parser.parse_args()

    log.basicConfig(format='%(message)s', level=log.DEBUG)
    log.debug("Args: %s", args)

    if args.ndk_path is not None:
        os.environ["ANDROID_NDK"] = args.ndk_path
    if args.sdk_path is not None:
        os.environ["ANDROID_SDK"] = args.sdk_path

    if not 'ANDROID_HOME' in os.environ and 'ANDROID_SDK' in os.environ:
        os.environ['ANDROID_HOME'] = os.environ["ANDROID_SDK"]

    if os.path.realpath(args.work_dir) == os.path.realpath(SCRIPT_DIR):
        raise Fail("Specify workdir (building from script directory is not supported)")
    if os.path.realpath(args.work_dir) == os.path.realpath(args.opencv_dir):
        raise Fail("Specify workdir (building from OpenCV source directory is not supported)")

    # Relative paths become invalid in sub-directories
    if args.opencv_dir is not None and not os.path.isabs(args.opencv_dir):
        args.opencv_dir = os.path.abspath(args.opencv_dir)
    if args.extra_modules_path is not None and not os.path.isabs(args.extra_modules_path):
        args.extra_modules_path = os.path.abspath(args.extra_modules_path)

    cpath = args.config
    if not os.path.exists(cpath):
        cpath = os.path.join(SCRIPT_DIR, cpath)
        if not os.path.exists(cpath):
            raise Fail('Config "%s" is missing' % args.config)
    with open(cpath, 'r') as f:
        cfg = f.read()
    print("Package configuration:")
    print('=' * 80)
    print(cfg.strip())
    print('=' * 80)

    ABIs = None  # make flake8 happy
    exec(compile(cfg, cpath, 'exec'))

    log.info("Android NDK path: %s", os.environ["ANDROID_NDK"])
    log.info("Android SDK path: %s", os.environ["ANDROID_SDK"])

    builder = Builder(args.work_dir, args.opencv_dir, args)

    log.info("Detected OpenCV version: %s", builder.opencv_version)

    for i, abi in enumerate(ABIs):
        do_install = (i == 0)

        log.info("=====")
        log.info("===== Building library for %s", abi)
        log.info("=====")

        os.chdir(builder.libdest)
        builder.clean_library_build_dir()
        builder.build_library(abi, do_install)

    builder.gather_results()

    if args.build_doc:
        builder.build_javadoc()

    log.info("=====")
    log.info("===== Build finished")
    log.info("=====")
    log.info("SDK location: %s", builder.resultdest)
    log.info("Documentation location: %s", builder.docdest)
