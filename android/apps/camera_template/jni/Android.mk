# date: Summer, 2010 
# author: Ethan Rublee
# contact: ethan.rublee@gmail.com
#
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

#define OPENCV_INCLUDES and OPENCV_LIBS
include $(OPENCV_CONFIG)

LOCAL_LDLIBS += $(OPENCV_LIBS) $(ANDROID_OPENCV_LIBS) -llog -lGLESv2
    
LOCAL_C_INCLUDES +=  $(OPENCV_INCLUDES) $(ANDROID_OPENCV_INCLUDES)

LOCAL_MODULE    := foobar

#make sure to pass in SWIG_C_OUT=gen/foobar_swig.cpp
#done in the makefile
LOCAL_SRC_FILES := gen/foo_swig.cpp TestBar.cpp

include $(BUILD_SHARED_LIBRARY)

