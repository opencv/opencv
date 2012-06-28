#!/usr/bin/python

import os
import sys

DEVICE_NAME = ""
DEVICE_STR = ""
if (DEVICE_NAME != ""):
    DEVICE_STR = "-s " + DEVICE_NAME

LOCAL_LOG_PATH = os.path.join(os.getcwd(), "logs")
DEVICE_LOG_PATH = "/sdcard/OpenCVEngineLogs"
DEVICE_BIN_PATH = "/data/data/EngineTest"

def RunTestApp(AppName):
    TestLog = os.path.join(DEVICE_LOG_PATH, AppName + ".xml")
    os.system("adb %s shell \"LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH;%s --gtest_output=\"xml:%s\"\"" % (DEVICE_STR, DEVICE_BIN_PATH, os.path.join(DEVICE_BIN_PATH, AppName), TestLog))
    os.system("adb %s pull \"%s\" \"%s\"" % (DEVICE_STR, TestLog, LOCAL_LOG_PATH))

if (__name__ ==  "__main__"):
    
    if (not os.path.exists(LOCAL_LOG_PATH)):
	os.makedirs(LOCAL_LOG_PATH)
    
    print("Waiting for device ...")
    os.system("adb %s wait-for-device" % DEVICE_STR)
    
    os.system("adb %s shell rm -r \"%s\"" % (DEVICE_STR, DEVICE_LOG_PATH))
    os.system("adb %s shell mkdir -p \"%s\"" % (DEVICE_STR, DEVICE_LOG_PATH))

    RunTestApp("OpenCVEngineTestApp")

