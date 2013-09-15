LOCAL_PATH := $(call my-dir)

#---------------------------------------------------------------------
#        Binder component library
#---------------------------------------------------------------------

include $(CLEAR_VARS)

LOCAL_MODULE_TAGS := optional

LOCAL_SRC_FILES := \
    BinderComponent/OpenCVEngine.cpp \
    BinderComponent/BnOpenCVEngine.cpp \
    BinderComponent/BpOpenCVEngine.cpp \
    BinderComponent/ProcReader.cpp \
    BinderComponent/TegraDetector.cpp \
    BinderComponent/StringUtils.cpp \
    BinderComponent/HardwareDetector.cpp

LOCAL_C_INCLUDES := \
    $(LOCAL_PATH)/include \
    $(LOCAL_PATH)/BinderComponent \
    $(TOP)/frameworks/base/include \
    $(TOP)/system/core/include

LOCAL_CFLAGS += -DPLATFORM_ANDROID
LOCAL_CFLAGS += -D__SUPPORT_ARMEABI_V7A_FEATURES
LOCAL_CFLAGS += -D__SUPPORT_TEGRA3
LOCAL_CFLAGS += -D__SUPPORT_MIPS
#LOCAL_CFLAGS += -D__SUPPORT_ARMEABI_FEATURES

LOCAL_PRELINK_MODULE := false

LOCAL_MODULE := libOpenCVEngine

LOCAL_LDLIBS += -lz -lbinder -llog -lutils

LOCAL_LDFLAGS += -Wl,-allow-shlib-undefined

include $(BUILD_SHARED_LIBRARY)

#---------------------------------------------------------------------
#        JNI library for Java service
#---------------------------------------------------------------------

include $(CLEAR_VARS)

LOCAL_MODULE_TAGS := optional

LOCAL_SRC_FILES := \
    JNIWrapper/OpenCVEngine_jni.cpp \
    NativeService/CommonPackageManager.cpp \
    JNIWrapper/JavaBasedPackageManager.cpp \
    NativeService/PackageInfo.cpp \
    JNIWrapper/HardwareDetector_jni.cpp \
    JNIWrapper/OpenCVLibraryInfo.cpp

LOCAL_C_INCLUDES := \
    $(LOCAL_PATH)/include \
    $(LOCAL_PATH)/JNIWrapper \
    $(LOCAL_PATH)/NativeService \
    $(LOCAL_PATH)/BinderComponent \
    $(TOP)/frameworks/base/include \
    $(TOP)/system/core/include \
    $(TOP)/frameworks/base/core/jni

LOCAL_PRELINK_MODULE := false

LOCAL_CFLAGS += -DPLATFORM_ANDROID
LOCAL_CFLAGS += -D__SUPPORT_ARMEABI_V7A_FEATURES
LOCAL_CFLAGS += -D__SUPPORT_TEGRA3
LOCAL_CFLAGS += -D__SUPPORT_MIPS
#LOCAL_CFLAGS += -D__SUPPORT_ARMEABI_FEATURES

LOCAL_MODULE := libOpenCVEngine_jni

LOCAL_LDLIBS += -lz -lbinder -llog -lutils -landroid_runtime
LOCAL_SHARED_LIBRARIES = libOpenCVEngine

include $(BUILD_SHARED_LIBRARY)

#---------------------------------------------------------------------
#        Native test application
#---------------------------------------------------------------------

#include $(LOCAL_PATH)/Tests/Tests.mk
