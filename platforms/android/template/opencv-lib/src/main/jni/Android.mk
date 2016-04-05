LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE    := OpenCVLibrary-3.0.0
LOCAL_SRC_FILES := OpenCVLibrary-3.0.0.cpp

include $(BUILD_SHARED_LIBRARY)
