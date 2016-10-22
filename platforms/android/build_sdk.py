#!/usr/bin/env python

import os, sys, subprocess, argparse, shutil, glob, re
import logging as log
import xml.etree.ElementTree as ET

class Fail(Exception):
    def __init__(self, text=None):
        self.t = text
    def __str__(self):
        return "ERROR" if self.t is None else self.t

def execute(cmd, shell=False):
    try:
        log.info("Executing: %s" % cmd)
        retcode = subprocess.call(cmd, shell=shell)
        if retcode < 0:
            raise Fail("Child was terminated by signal:" %s -retcode)
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

#===================================================================================================

class ABI:
    def __init__(self, platform_id, name, toolchain, cmake_name=None):
        self.platform_id = platform_id # platform code to add to apk version (for cmake)
        self.name = name # general name (official Android ABI identifier)
        self.toolchain = toolchain # toolchain identifier (for cmake)
        self.cmake_name = cmake_name # name of android toolchain (for cmake)
        if self.cmake_name is None:
            self.cmake_name = self.name
    def __str__(self):
        return "%s (%s)" % (self.name, self.toolchain)
    def haveIPP(self):
        return self.name == "x86" or self.name == "x86_64"

ABIs = [
    ABI("2", "armeabi-v7a", "arm-linux-androideabi-4.8", cmake_name="armeabi-v7a with NEON"),
    ABI("1", "armeabi",     "arm-linux-androideabi-4.8"),
    ABI("3", "arm64-v8a",   "aarch64-linux-android-4.9"),
    ABI("5", "x86_64",      "x86_64-4.9"),
    ABI("4", "x86",         "x86-4.8"),
    ABI("7", "mips64",      "mips64el-linux-android-4.9"),
    ABI("6", "mips",        "mipsel-linux-android-4.8")
]

#===================================================================================================

class Builder:
    def __init__(self, workdir, opencvdir):
        self.workdir = check_dir(workdir, create=True)
        self.opencvdir = check_dir(opencvdir)
        self.extra_modules_path = None
        self.libdest = check_dir(os.path.join(self.workdir, "o4a"), create=True, clean=True)
        self.docdest = check_dir(os.path.join(self.workdir, "javadoc"), create=True, clean=True)
        self.resultdest = check_dir(os.path.join(self.workdir, "OpenCV-android-sdk"), create=True, clean=True)
        self.extra_packs = []
        self.opencv_version = determine_opencv_version(os.path.join(self.opencvdir, "modules", "core", "include", "opencv2", "core", "version.hpp"))
        self.engine_version = determine_engine_version(os.path.join(self.opencvdir, "platforms", "android", "service", "engine", "AndroidManifest.xml"))
        self.use_ccache = True

    def get_toolchain_file(self):
        return os.path.join(self.opencvdir, "platforms", "android", "android.toolchain.cmake")

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
        cmd = [
            "cmake",
            "-GNinja",
            "-DCMAKE_TOOLCHAIN_FILE='%s'" % self.get_toolchain_file(),
            "-DWITH_OPENCL=OFF",
            "-DWITH_CUDA=OFF",
            "-DWITH_IPP=%s" % ("ON" if abi.haveIPP() else "OFF"),
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_PERF_TESTS=OFF",
            "-DBUILD_DOCS=OFF",
            "-DBUILD_ANDROID_EXAMPLES=ON",
            "-DINSTALL_ANDROID_EXAMPLES=ON",
            "-DANDROID_STL=gnustl_static",
            "-DANDROID_NATIVE_API_LEVEL=9",
            "-DANDROID_ABI='%s'" % abi.cmake_name,
            "-DWITH_TBB=ON",
            "-DANDROID_TOOLCHAIN_NAME=%s" % abi.toolchain
        ]

        if self.extra_modules_path is not None:
            cmd.append("-DOPENCV_EXTRA_MODULES_PATH='%s'" % self.extra_modules_path)

        cmd.append(self.opencvdir)

        if self.use_ccache == True:
            cmd.append("-DNDK_CCACHE=ccache")
        if do_install:
            cmd.extend(["-DBUILD_TESTS=ON", "-DINSTALL_TESTS=ON"])
        execute(cmd)
        if do_install:
            execute(["ninja"])
            for c in ["libs", "dev", "java", "samples"]:
                execute(["cmake", "-DCOMPONENT=%s" % c, "-P", "cmake_install.cmake"])
        else:
            execute(["ninja", "install/strip"])

    def build_engine(self, abi, engdest):
        cmd = [
            "cmake",
            "-GNinja",
            "-DCMAKE_TOOLCHAIN_FILE='%s'" % self.get_toolchain_file(),
            "-DANDROID_ABI='%s'" % abi.cmake_name,
            "-DBUILD_ANDROID_SERVICE=ON",
            "-DANDROID_PLATFORM_ID=%s" % abi.platform_id,
            "-DWITH_CUDA=OFF",
            "-DWITH_OPENCL=OFF",
            "-DWITH_IPP=OFF",
            self.opencvdir
        ]
        execute(cmd)
        apkdest = self.get_engine_apk_dest(engdest)
        # Add extra data
        apkxmldest = check_dir(os.path.join(apkdest, "res", "xml"), create=True)
        apklibdest = check_dir(os.path.join(apkdest, "libs", abi.name), create=True)
        for ver, d in self.extra_packs + [("3.1.0", os.path.join(self.libdest, "lib"))]:
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

        execute(["ninja", "opencv_engine"])
        execute(["ant", "-f", os.path.join(apkdest, "build.xml"), "debug"],
            shell=(sys.platform == 'win32'))
        # TODO: Sign apk

    def build_javadoc(self):
        classpaths = [os.path.join(self.libdest, "bin", "classes")]
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
            "-sourcepath", os.path.join(self.libdest, "src"),
            "-d", self.docdest,
            "-classpath", ":".join(classpaths)
        ]
        for _, dirs, _ in os.walk(os.path.join(self.libdest, "src", "org", "opencv")):
            cmd.extend(["org.opencv." + d for d in dirs])
        execute(cmd)

    def gather_results(self, engines):
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

        # Copy engines for all platforms
        for abi, engdest in engines:
            log.info("Copy engine: %s (%s)", abi, engdest)
            f = os.path.join(self.get_engine_apk_dest(engdest), "bin", "opencv_engine-debug.apk")
            resname = "OpenCV_%s_Manager_%s_%s.apk" % (self.opencv_version, self.engine_version, abi)
            shutil.copy2(f, os.path.join(self.resultdest, "apk", resname))

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


#===================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build OpenCV for Android SDK')
    parser.add_argument("work_dir", help="Working directory (and output)")
    parser.add_argument("opencv_dir", help="Path to OpenCV source dir")
    parser.add_argument('--ndk_path', help="Path to Android NDK to use for build")
    parser.add_argument('--sdk_path', help="Path to Android SDK to use for build")
    parser.add_argument("--extra_modules_path", help="Path to extra modules to use for build")
    parser.add_argument('--sign_with', help="Sertificate to sign the Manager apk")
    parser.add_argument('--build_doc', action="store_true", help="Build javadoc")
    parser.add_argument('--no_ccache', action="store_true", help="Do not use ccache during library build")
    parser.add_argument('--extra_pack', action='append', help="provide extra OpenCV libraries for Manager apk in form <version>:<path-to-native-libs>, for example '2.4.11:unpacked/sdk/native/libs'")
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

    if args.extra_modules_path is not None:
        builder.extra_modules_path = os.path.abspath(args.extra_modules_path)

    if args.no_ccache:
        builder.use_ccache = False

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

    if args.build_doc:
        builder.build_javadoc()

    builder.gather_results(engines)

    log.info("=====")
    log.info("===== Build finished")
    log.info("=====")
    log.info("SDK location: %s", builder.resultdest)
    log.info("Documentation location: %s", builder.docdest)
