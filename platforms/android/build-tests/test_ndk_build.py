#!/usr/bin/env python

import unittest
import os, sys, subprocess, argparse, shutil, re

TEMPLATE_ANDROID_MK = '''\
LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
{cut}
LOCAL_MODULE    := mixed_sample
LOCAL_SRC_FILES := {cpp1}
LOCAL_LDLIBS +=  -llog -ldl
include $(BUILD_SHARED_LIBRARY)
include $(CLEAR_VARS)
{cut}
LOCAL_MODULE    := mixed_sample2
LOCAL_SRC_FILES := {cpp2}
LOCAL_LDLIBS +=  -llog -ldl
LOCAL_SHARED_LIBS := mixed_sample
include $(BUILD_SHARED_LIBRARY)
'''

TEMPLATE_APPLICATION_MK = '''\
APP_STL := gnustl_static
APP_CPPFLAGS := -frtti -fexceptions
APP_ABI := {abi}
APP_PLATFORM := android-9
'''

TEMPLATE_JNI = '''\
#include <jni.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
using namespace std;
using namespace cv;
extern "C" {
JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial4_Sample4Mixed_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba);
JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial4_Sample4Mixed_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
{
    Mat& mGr  = *(Mat*)addrGray;
    Mat& mRgb = *(Mat*)addrRgba;
    vector<KeyPoint> v;
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(50);
    detector->detect(mGr, v);
    for( unsigned int i = 0; i < v.size(); i++ )
    {
        const KeyPoint& kp = v[i];
        circle(mRgb, Point(kp.pt.x, kp.pt.y), 10, Scalar(255,0,0,255));
    }
}
}
'''

#===================================================================================================

class TestNDKBuild(unittest.TestCase):
    def __init__(self, abi, libtype, opencv_mk_path, workdir, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.abi = abi # official NDK ABI name or 'all'
        self.libtype = libtype # 'static', etc
        self.opencv_mk_path = opencv_mk_path
        self.workdir = workdir
        self.jnidir = os.path.join(self.workdir, "jni")
        self.cpp1 = "jni_part1.cpp"
        self.cpp2 = "jni_part2.cpp"

    def shortDescription(self):
        return "ABI: %s, LIBTYPE: %s" % (self.abi, self.libtype)

    def gen_android_mk(self):
        p = []
        if self.libtype == "static":
            p.append("OPENCV_LIB_TYPE := STATIC")
        elif self.libtype == "shared_debug":
            p.append("OPENCV_LIB_TYPE := SHARED")
            p.append("OPENCV_CAMERA_MODULES:=on")
            p.append("OPENCV_INSTALL_MODULES:=on")
        elif self.libtype == "shared":
            p.append("OPENCV_LIB_TYPE := SHARED")
        p.append("include %s" % os.path.join(self.opencv_mk_path, "OpenCV.mk"))
        return TEMPLATE_ANDROID_MK.format(cut = "\n".join(p), cpp1 = self.cpp1, cpp2 = self.cpp2)

    def gen_jni_code(self):
        return TEMPLATE_JNI

    def gen_application_mk(self):
        return TEMPLATE_APPLICATION_MK.format(abi = self.abi)

    def write_jni_file(self, fname, contents):
        with open(os.path.join(self.jnidir, fname), "w") as f:
            f.write(contents)

    def setUp(self):
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)
        os.mkdir(self.workdir)
        os.mkdir(self.jnidir)
        self.write_jni_file("Android.mk", self.gen_android_mk())
        self.write_jni_file("Application.mk", self.gen_application_mk())
        self.write_jni_file(self.cpp1, self.gen_jni_code())
        self.write_jni_file(self.cpp2, self.gen_jni_code())
        os.chdir(self.workdir)

    def tearDown(self):
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)

    def runTest(self):
        ndk_path = os.environ["ANDROID_NDK"]
        retcode = subprocess.call([os.path.join(ndk_path, 'ndk-build'), "V=0"])
        self.assertEqual(retcode, 0)

def suite(workdir, opencv_mk_path):
    abis = ["armeabi", "armeabi-v7a", "x86", "mips"]
    ndk_path = os.environ["ANDROID_NDK"]
    with open(os.path.join(ndk_path, "RELEASE.TXT"), "r") as f:
        s = f.read()
        if re.search(r'r10[b-e]', s):
            abis.extend(["arm64-v8a", "x86", "x86_64"])
    abis.append("all")

    suite = unittest.TestSuite()
    for libtype in  ["static", "shared", "shared_debug"]:
        for abi in abis:
            suite.addTest(TestNDKBuild(abi, libtype, opencv_mk_path, workdir))
    return suite

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test OpenCV for Android SDK with NDK')
    parser.add_argument('--ndk_path', help="Path to Android NDK to use for build")
    parser.add_argument("--workdir", default="testspace", help="Working directory (and output)")
    parser.add_argument("opencv_mk_path", help="Path to folder with OpenCV.mk file (usually <SDK>/sdk/native/jni/")

    args = parser.parse_args()

    if args.ndk_path is not None:
        os.environ["ANDROID_NDK"] = os.path.abspath(args.ndk_path)

    print("Using NDK: %s" % os.environ["ANDROID_NDK"])

    res = unittest.TextTestRunner(verbosity=3).run(suite(os.path.abspath(args.workdir), os.path.abspath(args.opencv_mk_path)))
    if not res.wasSuccessful():
        sys.exit(res)
