LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := highgui
MODULE_PATH := $(OpenCV_Root)/modules/$(LOCAL_MODULE)

sources := \
	bitstrm.cpp \
    cap.cpp \
    grfmt_base.cpp \
    grfmt_bmp.cpp \
    grfmt_jpeg2000.cpp \
    grfmt_jpeg.cpp \
    grfmt_png.cpp \
    grfmt_tiff.cpp \
    grfmt_sunras.cpp \
    grfmt_pxm.cpp \
    loadsave.cpp \
    precomp.cpp \
    utils.cpp \
    window.cpp

LOCAL_SRC_FILES := $(sources:%=../../$(MODULE_PATH)/src/%)

LOCAL_C_INCLUDES := \
        $(OpenCVInclude) \
        $(MODULE_PATH)/src/ \
   
include $(BUILD_STATIC_LIBRARY)
