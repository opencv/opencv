LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

OPENCV_CAMERA_MODULES := off
include OpenCV.mk

LOCAL_MODULE    := opencv_java
LOCAL_SRC_FILES := android.cpp calib3d.cpp features2d.cpp imgproc.cpp ml.cpp utils.cpp VideoCapture.cpp  core.cpp highgui.cpp Mat.cpp objdetect.cpp video.cpp
LOCAL_LDLIBS +=  -llog -ldl -ljnigraphics

include $(BUILD_SHARED_LIBRARY)
