#ifndef __ENGINE_COMMON_H__
#define __ENGINE_COMMON_H__

// Global tag for Logcat output
#undef LOG_TAG
#define LOG_TAG "OpenCVEngine"

// OpenCV Engine API version
#ifndef OPEN_CV_ENGINE_VERSION
    #define OPEN_CV_ENGINE_VERSION 2
#endif

#define LIB_OPENCV_INFO_NAME "libopencv_info.so"

// OpenCV Manager package name
#define OPENCV_ENGINE_PACKAGE "org.opencv.engine"
// Class name of OpenCV engine binder object. Is needned for connection to service
#define OPECV_ENGINE_CLASSNAME "org.opencv.engine.OpenCVEngineInterface"

typedef const char* (*InfoFunctionType)();

#endif
