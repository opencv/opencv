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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

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
// This software is provided by the copyright holders and contributors as is and
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

#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wmissing-declarations"
#  if defined __clang__ || defined __APPLE__
#    pragma GCC diagnostic ignored "-Wmissing-prototypes"
#    pragma GCC diagnostic ignored "-Wextra"
#  endif
#endif

#ifndef __OPENCV_PERF_PRECOMP_HPP__
#define __OPENCV_PERF_PRECOMP_HPP__

#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wmissing-declarations"
#  if defined __clang__ || defined __APPLE__
#    pragma GCC diagnostic ignored "-Wmissing-prototypes"
#    pragma GCC diagnostic ignored "-Wextra"
#  endif
#endif

#include <iomanip>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cstdio>
#include <vector>
#include <numeric>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/video.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/ocl.hpp"
#include "opencv2/ts.hpp"

using namespace std;
using namespace cv;

#define OCL_SIZE_1000 Size(1000, 1000)
#define OCL_SIZE_2000 Size(2000, 2000)
#define OCL_SIZE_4000 Size(4000, 4000)

#define OCL_TYPICAL_MAT_SIZES ::testing::Values(OCL_SIZE_1000, OCL_SIZE_2000, OCL_SIZE_4000)

#define OCL_PERF_ENUM(type, ...) ::testing::Values(type, ## __VA_ARGS__ )

#define IMPL_OCL "ocl"
#define IMPL_GPU "gpu"
#define IMPL_PLAIN "plain"

#define RUN_OCL_IMPL (IMPL_OCL == getSelectedImpl())
#define RUN_PLAIN_IMPL (IMPL_PLAIN == getSelectedImpl())

#ifdef HAVE_OPENCV_GPU
# define RUN_GPU_IMPL (IMPL_GPU == getSelectedImpl())
#endif

#ifdef HAVE_OPENCV_GPU
#define OCL_PERF_ELSE               \
        if (RUN_GPU_IMPL)          \
            CV_TEST_FAIL_NO_IMPL(); \
        else                        \
            CV_TEST_FAIL_NO_IMPL();
#else
#define OCL_PERF_ELSE               \
            CV_TEST_FAIL_NO_IMPL();
#endif

#endif
