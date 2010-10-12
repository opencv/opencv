# date: Summer, 2010 
# author: Ethan Rublee
# contact: ethan.rublee@gmail.com
#
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

#pass in OPENCV_ROOT or define it here
#OPENCV_ROOT := ~/android-opencv/opencv
ifndef OPENCV_ROOT
${error please define OPENCV_ROOT before this point!}
endif

#define OPENCV_INCLUDES
include $(OPENCV_ROOT)/includes.mk
#define OPENCV_LIBS
include $(OPENCV_ROOT)/libs.mk

LOCAL_LDLIBS += $(OPENCV_LIBS) $(ANDROID_OPENCV_LIBS) -llog -lGLESv2
    
LOCAL_C_INCLUDES +=  $(OPENCV_INCLUDES) $(ANDROID_OPENCV_INCLUDES)

LOCAL_MODULE    := foobar


ifndef SWIG_C_OUT
${error please define SWIG_C_OUT before this point!}
endif

#make sure to pass in SWIG_C_OUT=gen/foobar_swig.cpp
#done in the makefile
LOCAL_SRC_FILES := ${SWIG_C_OUT} TestBar.cpp

include $(BUILD_SHARED_LIBRARY)

