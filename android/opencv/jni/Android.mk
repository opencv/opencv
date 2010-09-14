# date: Summer, 2010 
# author: Ethan Rublee
# contact: ethan.rublee@gmail.com
#
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

#pass in OPENCV_ROOT or define it here
#OPENCV_ROOT := ~/android-opencv/opencv

ifndef OPENCV_ROOT
OPENCV_ROOT := /home/ethan/workspace/googlecode_android_opencv/opencv
endif
#OPENCV_LIBS_DIR := $(OPENCV_ROOT)/bin/ndk/local/armeabi

#define OPENCV_INCLUDES
include $(OPENCV_ROOT)/includes.mk
#define OPENCV_LIBS
include $(OPENCV_ROOT)/libs.mk

LOCAL_LDLIBS += $(OPENCV_LIBS) -llog -lGLESv2
    
LOCAL_C_INCLUDES +=  $(OPENCV_INCLUDES) 

LOCAL_MODULE    := android-opencv

LOCAL_SRC_FILES := gen/android_cv_wrap.cpp image_pool.cpp yuv420sp2rgb.c gl_code.cpp Calibration.cpp

include $(BUILD_SHARED_LIBRARY)

