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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Guoping Long, longguoping@gmail.com
//    Yao Wang, bitwangyaoyao@gmail.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
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

#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#if defined _MSC_VER && _MSC_VER >= 1200
#pragma warning( disable: 4267 4324 4244 4251 4710 4711 4514 4996 )
#endif

#if defined(_WIN32)
#include <windows.h>
#endif

#include "cvconfig.h"

#if defined(BUILD_SHARED_LIBS)
#if defined WIN32 || defined _WIN32 || defined WINCE
#define CL_RUNTIME_EXPORT __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#define CL_RUNTIME_EXPORT __attribute__ ((visibility ("default")))
#else
#define CL_RUNTIME_EXPORT
#endif
#else
#define CL_RUNTIME_EXPORT
#endif

#include <map>
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include <sstream>
#include <exception>
#include <stdio.h>

#undef OPENCV_NOSTL

#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/objdetect_c.h"
#include "opencv2/ocl.hpp"

#include "opencv2/core/utility.hpp"
#include "opencv2/core/private.hpp"

#define __ATI__

#if defined (HAVE_OPENCL)

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include "opencv2/ocl/private/util.hpp"
#include "safe_call.hpp"

#else /* defined(HAVE_OPENCL) */

static inline void throw_nogpu()
{
    CV_Error(CV_GpuNotSupported, "The library is compilled without OpenCL support.\n");
}

#endif /* defined(HAVE_OPENCL) */

#endif /* __OPENCV_PRECOMP_H__ */
