#---------------------------------------------------------------------
#        Native test application
#---------------------------------------------------------------------

include $(CLEAR_VARS)

LOCAL_MODULE_TAGS := optional

LOCAL_SRC_FILES := \
    Tests/gtest/gtest-all.cpp \
    BinderComponent/ProcReader.cpp \
    BinderComponent/TegraDetector.cpp \
    BinderComponent/HardwareDetector.cpp \
    Tests/PackageManagerStub.cpp \
    NativeService/CommonPackageManager.cpp \
    NativeService/PackageInfo.cpp \
    Tests/PackageManagmentTest.cpp \
    Tests/PackageInfoTest.cpp \
    Tests/OpenCVEngineTest.cpp \
    Tests/TestMain.cpp
#     Tests/HardwareDetectionTest.cpp \

LOCAL_C_INCLUDES := \
    $(LOCAL_PATH)/Tests \
    $(LOCAL_PATH)/Tests/gtest \
    $(LOCAL_PATH)/include \
    $(LOCAL_PATH)/BinderComponent \
    $(LOCAL_PATH)/NativeService \
    $(LOCAL_PATH)/Tests/gtest/include \
    $(TOP)/frameworks/base/include \
    $(TOP)/system/core/include

LOCAL_CFLAGS += -O0 -DGTEST_HAS_CLONE=0 -DGTEST_OS_LINUX_ANDROID=1 -DGTEST_HAS_TR1_TUPLE=0
LOCAL_CFLAGS += -D__SUPPORT_ARMEABI_V7A_FEATURES
LOCAL_CFLAGS += -D__SUPPORT_TEGRA3
LOCAL_CFLAGS += -D__SUPPORT_MIPS
#LOCAL_CFLAGS += -D__SUPPORT_ARMEABI_FEATURES

#LOCAL_LDFLAGS = -Wl,-allow-shlib-undefined

LOCAL_MODULE := OpenCVEngineTestApp

LOCAL_LDLIBS += -lz -lbinder -llog -lutils

LOCAL_SHARED_LIBRARIES += libOpenCVEngine

include $(BUILD_EXECUTABLE)
