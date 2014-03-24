LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

INSTALL_CUDA_LIBRARIES:=on
OPENCV_INSTALL_MODULES:=on
CUDA_TOOLKIT_DIR=$(CUDA_TOOLKIT_ROOT)
include ../../sdk/native/jni/OpenCV.mk

LOCAL_MODULE    := cuda_sample
LOCAL_SRC_FILES := jni_part.cpp
LOCAL_LDLIBS +=  -llog -ldl
LOCAL_LDFLAGS += -Os

include $(BUILD_SHARED_LIBRARY)
