#ifndef __JAVA_COMMON_H__
#define __JAVA_COMMON_H__

#if !defined(__ppc__)
// to suppress warning from jni.h on OS X
# define TARGET_RT_MAC_CFM 0
#endif
#include <jni.h>

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

#include "opencv2/core/utility.hpp"

#include "converters.h"

#include "core_manual.hpp"
#include "features2d_manual.hpp"


#ifdef _MSC_VER
#  pragma warning(disable:4800 4244)
#endif

#endif //__JAVA_COMMON_H__
