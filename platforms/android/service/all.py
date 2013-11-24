#!/usr/bin/python

import os
import sys
import shutil

LOCAL_LOG_PATH = os.path.join(os.getcwd(), "logs")

if (__name__ ==  "__main__"):
    if (not os.path.exists(LOCAL_LOG_PATH)):
        os.makedirs(LOCAL_LOG_PATH)

    print("Building native part of OpenCV Manager...")
    HomeDir = os.getcwd()
    os.chdir(os.path.join(HomeDir, "engine"))
    shutil.rmtree(os.path.join(HomeDir, "engine", "libs"), ignore_errors=True)
    shutil.rmtree(os.path.join(HomeDir, "engine", "obj"), ignore_errors=True)
    BuildCommand = "ndk-build V=1 > \"%s\" 2>&1" % os.path.join(LOCAL_LOG_PATH, "build.log")
    #print(BuildCommand)
    res = os.system(BuildCommand)
    if (0 == res):
        print("Build\t[OK]")
    else:
        print("Build\t[FAILED]")
    sys.exit(-1)

    os.chdir(HomeDir)
    ConfFile = open("device.conf", "rt")

    for s in ConfFile.readlines():
        keys = s.split(";")
        if (len(keys) < 2):
           print("Error: invalid config line: \"%s\"" % s)
           continue
        Arch = keys[0]
        Name = keys[1]
        print("testing \"%s\" arch" % Arch)
        print("Pushing to device \"%s\"" % Name)
        PushCommand = "%s \"%s\" \"%s\" 2>&1" % (os.path.join(HomeDir, "push_native.py"), Arch, Name)
        os.system(PushCommand)
        print("Testing on device \"%s\"" % Name)
        TestCommand = "%s \"%s\" \"%s\" 2>&1" % (os.path.join(HomeDir, "test_native.py"), Arch, Name)
        os.system(TestCommand)
