LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

CUDA_TOOLKIT_DIR=../../../cuda-6.0
include ../../sdk/native/jni/OpenCV.mk

LOCAL_MODULE    := cuda_sample
LOCAL_SRC_FILES := jni_part.cpp
LOCAL_LDLIBS +=  -llog -ldl
LOCAL_LDFLAGS += -Os

include $(BUILD_SHARED_LIBRARY)
