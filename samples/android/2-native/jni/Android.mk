LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

OPENCV_MK_BUILD_PATH:=../../../../android/build/OpenCV.mk
ifeq ("$(wildcard $(OPENCV_MK_BUILD_PATH))","")
	include $(TOOLCHAIN_PREBUILT_ROOT)/user/share/OpenCV/OpenCV.mk
else
	include $(OPENCV_MK_BUILD_PATH)
endif

LOCAL_MODULE    := native_sample
LOCAL_SRC_FILES := jni_part.cpp
LOCAL_LDLIBS +=  -llog -ldl

include $(BUILD_SHARED_LIBRARY)
