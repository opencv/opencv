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

LOCAL_MODULE    := cvcamera

LOCAL_SRC_FILES := Processor.cpp gen/cvcamera_swig.cpp

include $(BUILD_SHARED_LIBRARY)

