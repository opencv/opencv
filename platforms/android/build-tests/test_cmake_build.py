#!/usr/bin/env python

import unittest
import os, sys, subprocess, argparse, shutil, re
import logging as log

log.basicConfig(format='%(message)s', level=log.DEBUG)

CMAKE_TEMPLATE='''\
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET(PROJECT_NAME hello-android)
PROJECT(${PROJECT_NAME})
FIND_PACKAGE(OpenCV REQUIRED %(libset)s)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
FILE(GLOB srcs "*.cpp")
ADD_EXECUTABLE(${PROJECT_NAME} ${srcs})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS} dl z)
'''

CPP_TEMPLATE = '''\
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;
const char* message = "Hello Android!";
int main(int argc, char* argv[])
{
  (void)argc; (void)argv;
  printf("%s\\n", message);
  Size textsize = getTextSize(message, CV_FONT_HERSHEY_COMPLEX, 3, 5, 0);
  Mat img(textsize.height + 20, textsize.width + 20, CV_32FC1, Scalar(230,230,230));
  putText(img, message, Point(10, img.rows - 10), CV_FONT_HERSHEY_COMPLEX, 3, Scalar(0, 0, 0), 5);
  imwrite("/mnt/sdcard/HelloAndroid.png", img);
  return 0;
}
'''

#===================================================================================================

class TestCmakeBuild(unittest.TestCase):
    def __init__(self, libset, abi, toolchain, opencv_cmake_path, workdir, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.libset = libset
        self.abi = abi
        self.toolchain = toolchain
        self.opencv_cmake_path = opencv_cmake_path
        self.workdir = workdir
        self.srcdir = os.path.join(self.workdir, "src")
        self.bindir = os.path.join(self.workdir, "build")

    def shortDescription(self):
        return "ABI: %s, TOOLCHAIN: %s, LIBSET: %s" % (self.abi, self.toolchain, self.libset)

    def gen_cmakelists(self):
        return CMAKE_TEMPLATE % {"libset": self.libset}

    def gen_code(self):
        return CPP_TEMPLATE

    def write_src_file(self, fname, content):
        with open(os.path.join(self.srcdir, fname), "w") as f:
            f.write(content)

    def setUp(self):
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)
        os.mkdir(self.workdir)
        os.mkdir(self.srcdir)
        os.mkdir(self.bindir)
        self.write_src_file("CMakeLists.txt", self.gen_cmakelists())
        self.write_src_file("main.cpp", self.gen_code())
        os.chdir(self.bindir)

    def tearDown(self):
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)

    def runTest(self):
        cmd = [
            "cmake",
            "-GNinja",
            "-DOpenCV_DIR=%s" % self.opencv_cmake_path,
            "-DANDROID_ABI=%s" % self.abi,
            "-DCMAKE_TOOLCHAIN_FILE=%s" % os.path.join(self.opencv_cmake_path, "android.toolchain.cmake"),
            "-DANDROID_TOOLCHAIN_NAME=%s" % self.toolchain,
            self.srcdir
        ]
        log.info("Executing: %s" % cmd)
        retcode = subprocess.call(cmd)
        self.assertEqual(retcode, 0, "cmake failed")

        cmd = ["ninja"]
        log.info("Executing: %s" % cmd)
        retcode = subprocess.call(cmd)
        self.assertEqual(retcode, 0, "make failed")

def suite(workdir, opencv_cmake_path):
    abis = {
        "armeabi":"arm-linux-androideabi-4.8",
        "armeabi-v7a":"arm-linux-androideabi-4.8",
        "arm64-v8a":"aarch64-linux-android-4.9",
        "x86":"x86-4.8",
        "x86_64":"x86_64-4.9",
        "mips":"mipsel-linux-android-4.8",
        "mips64":"mips64el-linux-android-4.9"
    }

    suite = unittest.TestSuite()
    for libset in ["", "opencv_java"]:
        for abi, toolchain in abis.items():
            suite.addTest(TestCmakeBuild(libset, abi, toolchain, opencv_cmake_path, workdir))
    return suite


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test OpenCV for Android SDK with cmake')
    parser.add_argument('--sdk_path', help="Path to Android SDK to use for build")
    parser.add_argument('--ndk_path', help="Path to Android NDK to use for build")
    parser.add_argument("--workdir", default="testspace", help="Working directory (and output)")
    parser.add_argument("opencv_cmake_path", help="Path to folder with OpenCVConfig.cmake and android.toolchain.cmake (usually <SDK>/sdk/native/jni/")

    args = parser.parse_args()

    if args.sdk_path is not None:
        os.environ["ANDROID_SDK"] = os.path.abspath(args.sdk_path)
    if args.ndk_path is not None:
        os.environ["ANDROID_NDK"] = os.path.abspath(args.ndk_path)

    print("Using SDK: %s" % os.environ["ANDROID_SDK"])
    print("Using NDK: %s" % os.environ["ANDROID_NDK"])

    res = unittest.TextTestRunner(verbosity=3).run(suite(os.path.abspath(args.workdir), os.path.abspath(args.opencv_cmake_path)))
    if not res.wasSuccessful():
        sys.exit(res)
