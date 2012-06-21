LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_MODULE_TAGS := optional

LOCAL_SRC_FILES := \
    src/info.c

LOCAL_C_INCLUDES := \
    $(LOCAL_PATH)/include \
    $(TOP)/frameworks/base/core/jni

LOCAL_PRELINK_MODULE := false

LOCAL_MODULE := libopencvinfo

include $(BUILD_SHARED_LIBRARY)
