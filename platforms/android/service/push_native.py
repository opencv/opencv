#!/usr/bin/python

import os
import sys

TARGET_DEVICE_PATH = "/data/data/EngineTest"
DEVICE_NAME = ""
DEVICE_STR = ""
DEVICE_ARCH = "armeabi"

if (__name__ ==  "__main__"):
    if (len(sys.argv) >= 3):
        DEVICE_ARCH = sys.argv[1]
        DEVICE_NAME = sys.argv[2]

    if (DEVICE_NAME != ""):
        DEVICE_STR = "-s \"" + DEVICE_NAME + "\""

    print("Waiting for device \"%s\" with arch \"%s\" ..." % (DEVICE_NAME, DEVICE_ARCH))
    os.system("adb %s wait-for-device" % DEVICE_STR)
    os.system("adb %s shell mkdir -p %s" % (DEVICE_STR, TARGET_DEVICE_PATH))
    os.system("adb %s push ./engine/libs/%s/libOpenCVEngine.so %s" % (DEVICE_STR, DEVICE_ARCH, TARGET_DEVICE_PATH))
    os.system("adb %s push ./engine/libs/%s/OpenCVEngineNativeClient %s" % (DEVICE_STR, DEVICE_ARCH, TARGET_DEVICE_PATH))
    os.system("adb %s push ./engine/libs/%s/OpenCVEngineNativeService %s" % (DEVICE_STR, DEVICE_ARCH, TARGET_DEVICE_PATH))
    os.system("adb %s push ./engine/libs/%s/OpenCVEngineTest %s" % (DEVICE_STR, DEVICE_ARCH, TARGET_DEVICE_PATH))
    os.system("adb %s push ./engine/libs/%s/OpenCVEngineTestApp %s" % (DEVICE_STR, DEVICE_ARCH, TARGET_DEVICE_PATH))
    os.system("adb %s push ./engine/libs/%s/libOpenCVEngine_jni.so %s" % (DEVICE_STR, DEVICE_ARCH, TARGET_DEVICE_PATH))
