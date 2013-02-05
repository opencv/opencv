LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

include ../../sdk/native/jni/OpenCV.mk

LOCAL_MODULE    := mixed_sample
LOCAL_SRC_FILES := jni_part.cpp
LOCAL_LDLIBS +=  -llog -ldl

include $(BUILD_SHARED_LIBRARY)
