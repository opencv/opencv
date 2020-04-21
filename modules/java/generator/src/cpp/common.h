// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef __OPENCV_JAVA_COMMON_H__
#define __OPENCV_JAVA_COMMON_H__

#include <stdexcept>
#include <string>

extern "C" {

#if !defined(__ppc__)
// to suppress warning from jni.h on OS X
# define TARGET_RT_MAC_CFM 0
#endif
#include <jni.h>

// make -fvisibility=hidden work with java 1.7
#if defined(__linux__) && !defined(__ANDROID__) && !defined (JNI_VERSION_1_8)
  // adapted from jdk1.8/jni.h
  #if (defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4) && (__GNUC_MINOR__ > 2))) || __has_attribute(visibility)
    #undef  JNIEXPORT
    #define JNIEXPORT     __attribute__((visibility("default")))
    #undef  JNIIMPORT
    #define JNIIMPORT     __attribute__((visibility("default")))
  #endif
#endif

} // extern "C"

#include "opencv_java.hpp"
#include "opencv2/core/utility.hpp"

#include "converters.h"
#include "listconverters.hpp"

#ifdef _MSC_VER
#  pragma warning(disable:4800 4244)
#endif

#endif //__OPENCV_JAVA_COMMON_H__
