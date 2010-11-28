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
    gl_code.cpp Calibration.cpp
    

#ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
#    LOCAL_CFLAGS := -DHAVE_NEON=1
#    LOCAL_SRC_FILES += yuv2rgb_neon.c.neon
#else
	LOCAL_SRC_FILES +=  yuv420sp2rgb.c
#endif
    

include $(BUILD_SHARED_LIBRARY)

