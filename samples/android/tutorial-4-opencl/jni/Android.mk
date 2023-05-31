LOCAL_PATH := $(call my-dir)

# add OpenCV
include $(CLEAR_VARS)
OPENCV_INSTALL_MODULES:=on
ifdef OPENCV_ANDROID_SDK
  ifneq ("","$(wildcard $(OPENCV_ANDROID_SDK)/OpenCV.mk)")
    include ${OPENCV_ANDROID_SDK}/OpenCV.mk
  else
    include ${OPENCV_ANDROID_SDK}/sdk/native/jni/OpenCV.mk
  endif
else
  include ../../sdk/native/jni/OpenCV.mk
endif

ifndef OPENCL_SDK
  $(error Specify OPENCL_SDK to Android OpenCL SDK location)
endif

# add OpenCL
LOCAL_C_INCLUDES += $(OPENCL_SDK)/include
LOCAL_LDLIBS += -L$(OPENCL_SDK)/lib/$(TARGET_ARCH_ABI) -lOpenCL

LOCAL_MODULE    := JNIpart
LOCAL_SRC_FILES := jni.c CLprocessor.cpp
LOCAL_LDLIBS    += -llog -lGLESv2 -lEGL
include $(BUILD_SHARED_LIBRARY)