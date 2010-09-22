# date: Summer, 2010 
# author: Ethan Rublee
# contact: ethan.rublee@gmail.com
#
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

#define OPENCV_INCLUDES and OPENCV_LIBS
include $(OPENCV_CONFIG)

LOCAL_LDLIBS += $(OPENCV_LIBS) -llog -lGLESv2
    
LOCAL_C_INCLUDES +=  $(OPENCV_INCLUDES) 

LOCAL_MODULE    := android-opencv

LOCAL_SRC_FILES := gen/android_cv_wrap.cpp image_pool.cpp \
    yuv420sp2rgb.c gl_code.cpp Calibration.cpp

include $(BUILD_SHARED_LIBRARY)

