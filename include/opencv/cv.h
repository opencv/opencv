/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#ifndef __OPENCV_OLD_CV_H__
#define __OPENCV_OLD_CV_H__

#if defined(_MSC_VER)
    #define CV_DO_PRAGMA(x) __pragma(x)
    #define __CVSTR2__(x) #x
    #define __CVSTR1__(x) __CVSTR2__(x)
    #define __CVMSVCLOC__ __FILE__ "("__CVSTR1__(__LINE__)") : "
    #define CV_MSG_PRAGMA(_msg) CV_DO_PRAGMA(message (__CVMSVCLOC__ _msg))
#elif defined(__GNUC__)
    #define CV_DO_PRAGMA(x) _Pragma (#x)
    #define CV_MSG_PRAGMA(_msg) CV_DO_PRAGMA(message (_msg))
#else
    #define CV_DO_PRAGMA(x)
    #define CV_MSG_PRAGMA(_msg)
#endif
#define CV_WARNING(x) CV_MSG_PRAGMA("Warning: " #x)

//CV_WARNING("This is a deprecated opencv header provided for compatibility. Please include a header from a corresponding opencv module")

#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/legacy/compat.hpp"

#if !defined(CV_IMPL)
#define CV_IMPL extern "C"
#endif //CV_IMPL

#if defined(__cplusplus)
#include "opencv2/core/internal.hpp"
#endif //__cplusplus

#endif // __OPENCV_OLD_CV_H_

