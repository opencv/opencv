LOCAL_PATH := $(call my-dir)
OpenCV_Root := $(LOCAL_PATH)/../..
OpenCVInclude :=  $(LOCAL_PATH) $(OpenCV_Root)/include/opencv $(OpenCV_Root)/3rdparty/include/ \
    $(OpenCV_Root)/modules/core/include/ $(OpenCV_Root)/modules/highgui/include/ \
    $(OpenCV_Root)/modules/imgproc/include $(OpenCV_Root)/modules/ml/include \
    $(OpenCV_Root)/modules/features2d/include \
    $(OpenCV_Root)/modules/legacy/include \
    $(OpenCV_Root)/modules/calib3d/include \
    $(OpenCV_Root)/modules/objdetect/include \
    $(OpenCV_Root)/modules/video/include \
    $(OpenCV_Root)/modules/contrib/include
include 3rdparty/Android.mk
include modules/Android.mk


