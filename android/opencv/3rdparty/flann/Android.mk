LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := flann

MODULE_ROOT := ${OpenCV_Root}/3rdparty/flann

sources := $(wildcard $(MODULE_ROOT)/*.cpp)
LOCAL_SRC_FILES := $(sources:%=../../%)

LOCAL_C_INCLUDES := $(OpenCV_Root)/3rdparty/include/ $(MODULE_ROOT)

include $(BUILD_STATIC_LIBRARY)
