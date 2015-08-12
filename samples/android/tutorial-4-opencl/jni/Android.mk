LOCAL_PATH := $(call my-dir)

# add OpenCV
include $(CLEAR_VARS)
OPENCV_INSTALL_MODULES:=on
ifeq ($(O4A_SDK_ROOT),)
    include ../../sdk/native/jni/OpenCV.mk
else
    include $(O4A_SDK_ROOT)/sdk/native/jni/OpenCV.mk
endif

# add OpenCL
LOCAL_C_INCLUDES += $(OPENCL_SDK)/include
LOCAL_LDLIBS += -L$(OPENCL_SDK)/lib/$(TARGET_ARCH_ABI) -lOpenCL

LOCAL_MODULE    := JNIrender
LOCAL_SRC_FILES := jni.c GLrender.cpp CLprocessor.cpp
LOCAL_LDLIBS    += -llog -lGLESv2 -lEGL
include $(BUILD_SHARED_LIBRARY)