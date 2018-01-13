// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Author: abratchik

#undef LOGE
#undef LOGD
#ifdef __ANDROID__
#  include <android/log.h>
#  define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))
#  ifdef DEBUG
#    define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#  else
#    define LOGD(...)
#  endif
#else
#  define LOGE(...)
#  define LOGD(...)
#endif

#ifndef OPENCV_JAVA_HPP
#define	OPENCV_JAVA_HPP

#define MATOFINT(ENV) static_cast<jclass>(ENV->NewGlobalRef(ENV->FindClass("org/opencv/core/MatOfInt")))
#define GETNATIVEOBJ(ENV, CLS, MAT) ENV->GetLongField(MAT, ENV->GetFieldID(CLS, "nativeObj", "J"))

#define CONSTRUCTOR(ENV, CLS) ENV->GetMethodID(CLS, "<init>", "(I)V")

#define ARRAYLIST(ENV) static_cast<jclass>(ENV->NewGlobalRef(ENV->FindClass("java/util/ArrayList")))
#define LIST_ADD(ENV, LIST) ENV->GetMethodID(LIST, "add", "(Ljava/lang/Object;)Z")
#define LIST_GET(ENV, LIST) ENV->GetMethodID(LIST, "get", "((I)Ljava/lang/Object;")
#define LIST_SIZE(ENV, LIST) ENV->GetMethodID(LIST, "size", "()I")
#define LIST_CLEAR(ENV, LIST) ENV->GetMethodID(LIST, "clear", "()V")

#define CHECK_MAT(cond) if(!(cond)){ LOGD("FAILED: " #cond); return; }

#endif	// OPENCV_JAVA_HPP
