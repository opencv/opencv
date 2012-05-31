LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

OPENCV_CAMERA_MODULES:=off

include ../includeOpenCV.mk
ifeq ("$(wildcard $(OPENCV_MK_PATH))","")
    #try to load OpenCV.mk from default install location
    include $(TOOLCHAIN_PREBUILT_ROOT)/user/share/OpenCV/OpenCV.mk
else
    include $(OPENCV_MK_PATH)
endif

LOCAL_SRC_FILES  := DetectionBasedTracker_jni.cpp
LOCAL_C_INCLUDES := $(LOCAL_PATH)
LOCAL_LDLIBS +=  -llog -ldl

LOCAL_MODULE     := detection_based_tacker

include $(BUILD_SHARED_LIBRARY)