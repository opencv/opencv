#!/usr/bin/env python

import os, sys
import argparse
import glob
import re
import shutil
import subprocess
import time

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

def check_executable(cmd):
    try:
        log.debug("Executing: %s" % cmd)
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        if not isinstance(result, str):
            result = result.decode("utf-8")
        log.debug("Result: %s" % (result+'\n').split('\n')[0])
        return True
    except Exception as e:
        log.debug('Failed: %s' % e)
        return False

def determine_engine_version(manifest_path):
    with open(manifest_path, "rt") as f:
        return re.search(r'android:versionName="(\d+\.\d+)"', f.read(), re.MULTILINE).group(1)

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
        self.engine_version = determine_engine_version(os.path.join(self.opencvdir, "platforms", "android", "service", "engine", "AndroidManifest.xml"))
        self.use_ccache = False if config.no_ccache else True
        self.cmake_path = self.get_cmake()
        self.ninja_path = self.get_ninja()

    def get_cmake(self):
        if not self.config.use_android_buildtools and check_executable(['cmake', '--version']):
            log.info("Using cmake from PATH")
            return 'cmake'
        # look to see if Android SDK's cmake is installed
        android_cmake = os.path.join(os.environ['ANDROID_SDK'], 'cmake')
        if os.path.exists(android_cmake):
            cmake_subdirs = [f for f in os.listdir(android_cmake) if check_executable([os.path.join(android_cmake, f, 'bin', 'cmake'), '--version'])]
            if len(cmake_subdirs) > 0:
                # there could be more than one - just take the first one
                cmake_from_sdk = os.path.join(android_cmake, cmake_subdirs[0], 'bin', 'cmake')
                log.info("Using cmake from Android SDK: %s", cmake_from_sdk)
                return cmake_from_sdk
        raise Fail("Can't find cmake")

    def get_ninja(self):
        if not self.config.use_android_buildtools and check_executable(['ninja', '--version']):
            log.info("Using ninja from PATH")
            return 'ninja'
        # Android SDK's cmake includes a copy of ninja - look to see if its there
        android_cmake = os.path.join(os.environ['ANDROID_SDK'], 'cmake')
        if os.path.exists(android_cmake):
            cmake_subdirs = [f for f in os.listdir(android_cmake) if check_executable([os.path.join(android_cmake, f, 'bin', 'ninja'), '--version'])]
            if len(cmake_subdirs) > 0:
                # there could be more than one - just take the first one
                ninja_from_sdk = os.path.join(android_cmake, cmake_subdirs[0], 'bin', 'ninja')
                log.info("Using ninja from Android SDK: %s", ninja_from_sdk)
                return ninja_from_sdk
        raise Fail("Can't find ninja")

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
        cmd = [self.cmake_path, "-GNinja"]
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
        if self.ninja_path != 'ninja':
            cmake_vars['CMAKE_MAKE_PROGRAM'] = self.ninja_path

        if self.config.modules_list is not None:
            cmd.append("-DBUILD_LIST='%s'" % self.config.modules_list)

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
        if do_install:
            execute([self.ninja_path])
            for c in ["libs", "dev", "java", "samples"]:
                execute([self.cmake_path, "-DCOMPONENT=%s" % c, "-P", "cmake_install.cmake"])
        else:
            execute([self.ninja_path, "install/strip"])

    def build_engine(self, abi, engdest):
        cmd = [self.cmake_path, "-GNinja"]
        cmake_vars = dict(
            CMAKE_TOOLCHAIN_FILE=self.get_toolchain_file(),
            WITH_OPENCL="OFF",
            WITH_IPP="OFF",
            BUILD_ANDROID_SERVICE = 'ON'
        )
        if self.ninja_path != 'ninja':
            cmake_vars['CMAKE_MAKE_PROGRAM'] = self.ninja_path
        cmake_vars.update(abi.cmake_vars)
        cmd += [ "-D%s='%s'" % (k, v) for (k, v) in cmake_vars.items() if v is not None]
        cmd.append(self.opencvdir)
        execute(cmd)
        apkdest = self.get_engine_apk_dest(engdest)
        assert os.path.exists(apkdest), apkdest
        # Add extra data
        apkxmldest = check_dir(os.path.join(apkdest, "res", "xml"), create=True)
        apklibdest = check_dir(os.path.join(apkdest, "libs", abi.name), create=True)
        for ver, d in self.extra_packs + [("3.4.11", os.path.join(self.libdest, "lib"))]:
            r = ET.Element("library", attrib={"version": ver})
            log.info("Adding libraries from %s", d)

            for f in glob.glob(os.path.join(d, abi.name, "*.so")):
                log.info("Copy file: %s", f)
                shutil.copy2(f, apklibdest)
                if "libnative_camera" in f:
                    continue
                log.info("Register file: %s", os.path.basename(f))
                n = ET.SubElement(r, "file", attrib={"name": os.path.basename(f)})

            if len(list(r)) > 0:
                xmlname = os.path.join(apkxmldest, "config%s.xml" % ver.replace(".", ""))
                log.info("Generating XML config: %s", xmlname)
                ET.ElementTree(r).write(xmlname, encoding="utf-8")

        execute([self.ninja_path, "opencv_engine"])
        execute(["ant", "-f", os.path.join(apkdest, "build.xml"), "debug"],
            shell=(sys.platform == 'win32'))
        # TODO: Sign apk

    def build_javadoc(self):
        classpaths = []
        for dir, _, files in os.walk(os.environ["ANDROID_SDK"]):
            for f in files:
                if f == "android.jar" or f == "annotations.jar":
                    classpaths.append(os.path.join(dir, f))
        srcdir = os.path.join(self.resultdest, 'sdk', 'java', 'src')
        dstdir = self.docdest
        # synchronize with modules/java/jar/build.xml.in
        shutil.copy2(os.path.join(SCRIPT_DIR, '../../doc/mymath.js'), dstdir)
        cmd = [
            "javadoc",
            '-windowtitle', 'OpenCV %s Java documentation' % self.opencv_version,
            '-doctitle', 'OpenCV Java documentation (%s)' % self.opencv_version,
            "-nodeprecated",
            "-public",
            '-sourcepath', srcdir,
            '-encoding', 'UTF-8',
            '-charset', 'UTF-8',
            '-docencoding', 'UTF-8',
            '--allow-script-in-comments',
            '-header',
'''
            <script>
              var url = window.location.href;
              var pos = url.lastIndexOf('/javadoc/');
              url = pos >= 0 ? (url.substring(0, pos) + '/javadoc/mymath.js') : (window.location.origin + '/mymath.js');
              var script = document.createElement('script');
              script.src = '%s/MathJax.js?config=TeX-AMS-MML_HTMLorMML,' + url;
              document.getElementsByTagName('head')[0].appendChild(script);
            </script>
''' % 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0',
            '-bottom', 'Generated on %s / OpenCV %s' % (time.strftime("%Y-%m-%d %H:%M:%S"), self.opencv_version),
            "-d", dstdir,
            "-classpath", ":".join(classpaths),
            '-subpackages', 'org.opencv',
        ]
        execute(cmd)

    def gather_results(self, engines):
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

        # Copy engines for all platforms
        for abi, engdest in engines:
            log.info("Copy engine: %s (%s)", abi, engdest)
            f = os.path.join(self.get_engine_apk_dest(engdest), "bin", "opencv_engine-debug.apk")
            resname = "OpenCV_%s_Manager_%s_%s.apk" % (self.opencv_version, self.engine_version, abi)
            dst = os.path.join(self.resultdest, "apk", resname)
            if self.config.force_copy:
                shutil.copy2(f, dst)
            else:
                shutil.move(f, dst)

        # Clean samples
        path = os.path.join(self.resultdest, "samples")
        for item in os.listdir(path):
            item = os.path.join(path, item)
            if os.path.isdir(item):
                for name in ["build.xml", "local.properties", "proguard-project.txt"]:
                    rm_one(os.path.join(item, name))


#===================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build OpenCV for Android SDK')
    parser.add_argument("work_dir", nargs='?', default='.', help="Working directory (and output)")
    parser.add_argument("opencv_dir", nargs='?', default=os.path.join(SCRIPT_DIR, '../..'), help="Path to OpenCV source dir")
    parser.add_argument('--config', default='ndk-10.config.py', type=str, help="Package build configuration", )
    parser.add_argument('--ndk_path', help="Path to Android NDK to use for build")
    parser.add_argument('--sdk_path', help="Path to Android SDK to use for build")
    parser.add_argument('--use_android_buildtools', action="store_true", help='Use cmake/ninja build tools from Android SDK')
    parser.add_argument("--modules_list", help="List of  modules to include for build")
    parser.add_argument("--extra_modules_path", help="Path to extra modules to use for build")
    parser.add_argument('--sign_with', help="Certificate to sign the Manager apk")
    parser.add_argument('--build_doc', action="store_true", help="Build javadoc")
    parser.add_argument('--no_ccache', action="store_true", help="Do not use ccache during library build")
    parser.add_argument('--extra_pack', action='append', help="provide extra OpenCV libraries for Manager apk in form <version>:<path-to-native-libs>, for example '2.4.11:unpacked/sdk/native/libs'")
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

    if not 'ANDROID_SDK' in os.environ:
        raise Fail("SDK location not set. Either pass --sdk_path or set ANDROID_SDK environment variable")

    # look for an NDK installed with the Android SDK
    if not 'ANDROID_NDK' in os.environ and 'ANDROID_SDK' in os.environ and os.path.exists(os.path.join(os.environ["ANDROID_SDK"], 'ndk-bundle')):
        os.environ['ANDROID_NDK'] = os.path.join(os.environ["ANDROID_SDK"], 'ndk-bundle')

    if not 'ANDROID_NDK' in os.environ:
        raise Fail("NDK location not set. Either pass --ndk_path or set ANDROID_NDK environment variable")

    if not check_executable(['ccache', '--version']):
        log.info("ccache not found - disabling ccache support")
        args.no_ccache = True

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
    log.info("Detected Engine version: %s", builder.engine_version)

    if args.extra_pack:
        for one in args.extra_pack:
            i = one.find(":")
            if i > 0 and i < len(one) - 1:
                builder.add_extra_pack(one[:i], one[i+1:])
            else:
                raise Fail("Bad extra pack provided: %s, should be in form '<version>:<path-to-native-libs>'" % one)

    engines = []
    for i, abi in enumerate(ABIs):
        do_install = (i == 0)
        engdest = check_dir(os.path.join(builder.workdir, "build_service_%s" % abi.name), create=True, clean=True)

        log.info("=====")
        log.info("===== Building library for %s", abi)
        log.info("=====")

        os.chdir(builder.libdest)
        builder.clean_library_build_dir()
        builder.build_library(abi, do_install)

        log.info("=====")
        log.info("===== Building engine for %s", abi)
        log.info("=====")

        os.chdir(engdest)
        builder.build_engine(abi, engdest)
        engines.append((abi.name, engdest))

    builder.gather_results(engines)

    if args.build_doc:
        builder.build_javadoc()

    log.info("=====")
    log.info("===== Build finished")
    log.info("=====")
    log.info("SDK location: %s", builder.resultdest)
    log.info("Documentation location: %s", builder.docdest)
