/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __VIDEOIO_H_
#define __VIDEOIO_H_

#if defined(__OPENCV_BUILD) && defined(BUILD_PLUGIN)
#undef __OPENCV_BUILD  // allow public API only
#define OPENCV_HAVE_CVCONFIG_H 1  // but we still have access to cvconfig.h (TODO remove this)
#include <opencv2/core.hpp>
#include <opencv2/core/utils/trace.hpp>
#endif

#if defined __linux__ || defined __APPLE__ || defined __HAIKU__
#include <unistd.h>  // -D_FORTIFY_SOURCE=2 workaround: https://github.com/opencv/opencv/issues/15020
#endif

#include "opencv2/core/cvdef.h"
#include "opencv2/videoio.hpp"

#include "opencv2/core/utility.hpp"
#ifdef __OPENCV_BUILD
#include "opencv2/core/private.hpp"
#endif

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.defines.hpp>
#ifdef NDEBUG
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_DEBUG + 1
#else
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#endif
#include <opencv2/core/utils/logger.hpp>

#include "opencv2/imgcodecs.hpp"

#include "opencv2/imgproc.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <ctype.h>

#if defined _WIN32 || defined WINCE
    #if !defined _WIN32_WINNT
        #ifdef HAVE_MSMF
            #define _WIN32_WINNT 0x0600 // Windows Vista
        #else
            #define _WIN32_WINNT 0x0501 // Windows XP
        #endif
    #endif

    #include <windows.h>
    #undef small
    #undef min
    #undef max
    #undef abs
#endif

#include "cap_interface.hpp"

#endif /* __VIDEOIO_H_ */
