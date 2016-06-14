LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE    := OpenCVLibrary
LOCAL_SRC_FILES := OpenCVLibrary.cpp

include $(BUILD_SHARED_LIBRARY)
