#!/usr/bin/env python

import unittest
import os, sys, subprocess, argparse, shutil, re
from os.path import abspath

class TestAntBuild(unittest.TestCase):
    pass

    def __init__(self, target, workdir, lib_dir, sample_dir, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.target = target
        self.workdir = workdir
        self.src_lib_dir = lib_dir
        self.src_sample_dir = sample_dir
        self.lib_dir = os.path.join(self.workdir, "opencv")
        self.sample_dir = os.path.join(self.workdir, "project")

    def shortDescription(self):
        return "TARGET: %r, SAMPLE: %s" % (self.target, os.path.basename(self.src_sample_dir))

    def setUp(self):
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)
        os.mkdir(self.workdir)
        shutil.copytree(self.src_lib_dir, self.lib_dir)
        shutil.copytree(self.src_sample_dir, self.sample_dir)
        os.remove(os.path.join(self.sample_dir, "project.properties"))

    def tearDown(self):
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)

    def runTest(self):
        cmd = [os.path.join(os.environ["ANDROID_SDK"], "tools", "android"), "update", "project", "-p", self.lib_dir, "-t", self.target[0]]
        retcode = subprocess.call(cmd)
        self.assertEqual(retcode, 0, "android update opencv project failed")

        cmd = ["ant", "-f", os.path.join(self.lib_dir, "build.xml"), "debug"]
        retcode = subprocess.call(cmd)
        self.assertEqual(retcode, 0, "opencv ant build failed")

        cmd = [os.path.join(os.environ["ANDROID_SDK"], "tools", "android"), "update", "project", "-p", self.sample_dir, "-t", self.target[1], "-l", os.path.relpath(self.lib_dir, self.sample_dir)]
        retcode = subprocess.call(cmd)
        self.assertEqual(retcode, 0, "android update sample project failed")

        cmd = ["ant", "-f", os.path.join(self.sample_dir, "build.xml"), "debug"]
        retcode = subprocess.call(cmd)
        self.assertEqual(retcode, 0, "sample ant build failed")

def suite(workdir, opencv_lib_path, opencv_samples_path):
    suite = unittest.TestSuite()
    for target in [("android-21", "android-14"), ("android-21", "android-17")]:
        for item in os.listdir(opencv_samples_path):
            item = os.path.join(opencv_samples_path, item)
            if (os.path.exists(os.path.join(item, "AndroidManifest.xml"))):
                suite.addTest(TestAntBuild(target, workdir, opencv_lib_path, item))
    return suite

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test OpenCV for Android SDK with ant')
    parser.add_argument('--sdk_path', help="Path to Android SDK to use for build")
    parser.add_argument("--workdir", default="testspace", help="Working directory (and output)")
    parser.add_argument("opencv_lib_path", help="Path to folder with SDK java library (usually <SDK>/sdk/java/)")
    parser.add_argument("opencv_samples_path", help="Path to folder with SDK samples (usually <SDK>/samples/)")

    args = parser.parse_args()

    if args.sdk_path is not None:
        os.environ["ANDROID_SDK"] = os.path.abspath(args.sdk_path)

    print("Using SDK: %s" % os.environ["ANDROID_SDK"])

    s = suite(abspath(args.workdir), abspath(args.opencv_lib_path), abspath(args.opencv_samples_path))
    res = unittest.TextTestRunner(verbosity=3).run(s)
    if not res.wasSuccessful():
        sys.exit(res)
