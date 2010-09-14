LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := legacy
MODULE_PATH := $(OpenCV_Root)/modules/$(LOCAL_MODULE)

sources := $(wildcard $(MODULE_PATH)/src/*.cpp)
LOCAL_SRC_FILES := $(sources:%=../../%)

LOCAL_C_INCLUDES := \
        $(OpenCVInclude) \
        $(MODULE_PATH)/src/ \
   
include $(BUILD_STATIC_LIBRARY)
