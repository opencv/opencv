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

} // extern "C"

#include "opencv_java.hpp"
#include "opencv2/core/utility.hpp"

#include "converters.h"
#include "listconverters.hpp"

#ifdef _MSC_VER
#  pragma warning(disable:4800 4244)
#endif

#endif //__OPENCV_JAVA_COMMON_H__
