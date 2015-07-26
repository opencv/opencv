LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE    := JNIrender
LOCAL_SRC_FILES := jni.c GLrender.cpp
LOCAL_LDLIBS    += -llog -lGLESv2 -lEGL

include $(BUILD_SHARED_LIBRARY)