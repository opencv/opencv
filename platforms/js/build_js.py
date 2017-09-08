#!/usr/bin/env python

import os, sys, subprocess, argparse, shutil, glob, re, multiprocessing
import logging as log

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

def determine_emcc_version(emscripten_dir):
    ret = subprocess.check_output([os.path.join(emscripten_dir, "emcc"), "--version"])
    m = re.match(r'^emcc.*(\d+\.\d+\.\d+)', ret, flags=re.IGNORECASE)
    return m.group(1)

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

class Builder:
    def __init__(self, build_dir, opencv_dir, emscripten_dir):
        self.build_dir = check_dir(build_dir, create=True)
        self.opencv_dir = check_dir(opencv_dir)
        self.emscripten_dir = check_dir(emscripten_dir)
        self.opencv_version = determine_opencv_version(os.path.join(self.opencv_dir, "modules", "core", "include", "opencv2", "core", "version.hpp"))
        self.emcc_version = determine_emcc_version(self.emscripten_dir)

    def get_toolchain_file(self):
        return os.path.join(self.emscripten_dir, "cmake", "Modules", "Platform", "Emscripten.cmake")

    def clean_build_dir(self):
        for d in ["CMakeCache.txt", "CMakeFiles/", "bin/", "libs/", "lib/", "modules"]:
            rm_one(d)

    def get_cmake_cmd(self):
        cmd = ["cmake",
               "-DCMAKE_BUILD_TYPE=Release",
               "-DCMAKE_TOOLCHAIN_FILE='%s'" % self.get_toolchain_file(),
               "-DCPU_BASELINE=''",
               "-DCPU_DISPATCH=''",
               "-DCV_TRACE=OFF",
               "-DBUILD_SHARED_LIBS=OFF",
               "-DWITH_1394=OFF",
               "-DWITH_VTK=OFF",
               "-DWITH_CUDA=OFF",
               "-DWITH_CUFFT=OFF",
               "-DWITH_CUBLAS=OFF",
               "-DWITH_NVCUVID=OFF",
               "-DWITH_EIGEN=OFF",
               "-DWITH_FFMPEG=OFF",
               "-DWITH_GSTREAMER=OFF",
               "-DWITH_GTK=OFF",
               "-DWITH_GTK_2_X=OFF",
               "-DWITH_IPP=OFF",
               "-DWITH_JASPER=OFF",
               "-DWITH_JPEG=OFF",
               "-DWITH_WEBP=OFF",
               "-DWITH_OPENEXR=OFF",
               "-DWITH_OPENGL=OFF",
               "-DWITH_OPENVX=OFF",
               "-DWITH_OPENNI=OFF",
               "-DWITH_OPENNI2=OFF",
               "-DWITH_PNG=OFF",
               "-DWITH_TBB=OFF",
               "-DWITH_PTHREADS_PF=OFF",
               "-DWITH_TIFF=OFF",
               "-DWITH_V4L=OFF",
               "-DWITH_OPENCL=OFF",
               "-DWITH_OPENCL_SVM=OFF",
               "-DWITH_OPENCLAMDFFT=OFF",
               "-DWITH_OPENCLAMDBLAS=OFF",
               "-DWITH_MATLAB=OFF",
               "-DWITH_GPHOTO2=OFF",
               "-DWITH_LAPACK=OFF",
               "-DWITH_ITT=OFF",
               "-DBUILD_ZLIB=ON",
               "-DBUILD_opencv_apps=OFF",
               "-DBUILD_opencv_flann=OFF",
               "-DBUILD_opencv_ml=OFF",
               "-DBUILD_opencv_photo=OFF",
               "-DBUILD_opencv_imgcodecs=OFF",
               "-DBUILD_opencv_shape=OFF",
               "-DBUILD_opencv_videoio=OFF",
               "-DBUILD_opencv_highgui=OFF",
               "-DBUILD_opencv_superres=OFF",
               "-DBUILD_opencv_java=OFF",
               "-DBUILD_opencv_js=ON",
               "-DBUILD_opencv_python2=OFF",
               "-DBUILD_opencv_python3=OFF",
               "-DBUILD_EXAMPLES=OFF",
               "-DBUILD_PACKAGE=OFF",
               "-DBUILD_TESTS=OFF",
               "-DBUILD_PERF_TESTS=OFF",
               "-DBUILD_DOCS=ON"]
        return cmd;

    def config_asmjs(self):
        cmd = self.get_cmake_cmd()
        cmd.append(self.opencv_dir)
        execute(cmd)

    def config_wasm(self):
        cmd = self.get_cmake_cmd()
        cmd += ["-DCMAKE_C_FLAGS='-s WASM=1'",
                "-DCMAKE_CXX_FLAGS='-s WASM=1'"]
        cmd.append(self.opencv_dir)
        execute(cmd)

    def build_opencvjs(self):
        execute(["make", "-j", str(multiprocessing.cpu_count()), "opencv.js"])

    def build_test(self):
        execute(["make", "-j", str(multiprocessing.cpu_count()), "opencv_js_test"])

    def build_doc(self):
        execute(["make", "-j", str(multiprocessing.cpu_count()), "doxygen"])


#===================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build OpenCV.js by Emscripten')
    parser.add_argument("build_dir", help="Building directory (and output)")
    parser.add_argument("opencv_dir", help="Path to OpenCV source dir")
    parser.add_argument('--emscripten_dir', help="Path to Emscripten to use for build")
    parser.add_argument('--build_wasm', action="store_true", help="Build OpenCV.js in WebAssembly format")
    parser.add_argument('--build_test', action="store_true", help="Build tests")
    parser.add_argument('--build_doc', action="store_true", help="Build tutorials")
    parser.add_argument('--clean_build_dir', action="store_true", help="Clean build dir")
    parser.add_argument('--skip_config', action="store_true", help="Skip cmake config")
    parser.add_argument('--config_only', action="store_true", help="Only do cmake config")
    args = parser.parse_args()

    log.basicConfig(format='%(message)s', level=log.DEBUG)
    log.debug("Args: %s", args)

    emscripten_dir = args.emscripten_dir
    if emscripten_dir is None:
        if "EMSCRIPTEN" in os.environ:
            emscripten_dir = os.environ["EMSCRIPTEN"]
        if emscripten_dir is None:
            log.info("Cannot get emscripten path, please specify it either by EMSCRIPTEN environment variable or --emscripten_dir option.")
            sys.exit(-1)

    log.info("Emscripten path: %s", emscripten_dir)

    builder = Builder(args.build_dir, args.opencv_dir, emscripten_dir)

    log.info("Detected OpenCV version: %s", builder.opencv_version)
    log.info("Detected emcc version: %s", builder.emcc_version)

    os.chdir(builder.build_dir)

    if args.clean_build_dir:
        log.info("=====")
        log.info("===== Clean build dir %s", builder.build_dir)
        log.info("=====")
        builder.clean_build_dir()

    if not args.skip_config:
        if not args.build_wasm:
            log.info("=====")
            log.info("===== Config OpenCV.js build for asm.js")
            log.info("=====")
            builder.config_asmjs()
        else:
            log.info("=====")
            log.info("===== Config OpenCV.js build for wasm")
            log.info("=====")
            builder.config_wasm()

    if args.config_only:
        sys.exit(0);

    log.info("=====")
    log.info("===== Building OpenCV.js in %s", "asm.js" if not args.build_wasm else "wasm")
    log.info("=====")
    builder.build_opencvjs()

    if args.build_test:
        log.info("=====")
        log.info("===== Building OpenCV.js tests")
        log.info("=====")
        builder.build_test()

    if args.build_doc:
        log.info("=====")
        log.info("===== Building OpenCV.js tutorials")
        log.info("=====")
        builder.build_doc()


    log.info("=====")
    log.info("===== Build finished")
    log.info("=====")
    log.info("OpenCV.js location: %s", os.path.join(builder.build_dir, "bin", "opencv.js"))

    if args.build_test:
        log.info("OpenCV.js tests location: %s", os.path.join(builder.build_dir, "bin", "tests.html"))

    if args.build_doc:
        log.info("OpenCV.js tutorials location: %s", os.path.join(builder.build_dir, "doc", "doxygen", "html"))
