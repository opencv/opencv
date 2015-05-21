/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
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

#ifndef __OPENCV_HAL_HPP__
#define __OPENCV_HAL_HPP__

#include "opencv2/hal/defs.h"

/**
  @defgroup hal Hardware Acceleration Layer
*/

namespace cv { namespace hal {

namespace Error {

enum
{
    Ok = 0,
    Unknown = -1
};

}

CV_EXPORTS_W int normHamming(const uchar* a, int n);
CV_EXPORTS_W int normHamming(const uchar* a, const uchar* b, int n);

CV_EXPORTS_W int normHamming(const uchar* a, int n, int cellSize);
CV_EXPORTS_W int normHamming(const uchar* a, const uchar* b, int n, int cellSize);

//////////////////////////////// low-level functions ////////////////////////////////

CV_EXPORTS_W int LU(float* A, size_t astep, int m, float* b, size_t bstep, int n);
CV_EXPORTS_W int LU(double* A, size_t astep, int m, double* b, size_t bstep, int n);
CV_EXPORTS_W bool Cholesky(float* A, size_t astep, int m, float* b, size_t bstep, int n);
CV_EXPORTS_W bool Cholesky(double* A, size_t astep, int m, double* b, size_t bstep, int n);

CV_EXPORTS_W int normL1_(const uchar* a, const uchar* b, int n);
CV_EXPORTS_W float normL1_(const float* a, const float* b, int n);
CV_EXPORTS_W float normL2Sqr_(const float* a, const float* b, int n);

CV_EXPORTS_W void exp(const float* src, float* dst, int n);
CV_EXPORTS_W void exp(const double* src, double* dst, int n);
CV_EXPORTS_W void log(const float* src, float* dst, int n);
CV_EXPORTS_W void log(const double* src, double* dst, int n);

CV_EXPORTS_W void fastAtan2(const float* y, const float* x, float* dst, int n, bool angleInDegrees);
CV_EXPORTS_W void magnitude(const float* x, const float* y, float* dst, int n);
CV_EXPORTS_W void magnitude(const double* x, const double* y, double* dst, int n);
CV_EXPORTS_W void sqrt(const float* src, float* dst, int len);
CV_EXPORTS_W void sqrt(const double* src, double* dst, int len);
CV_EXPORTS_W void invSqrt(const float* src, float* dst, int len);
CV_EXPORTS_W void invSqrt(const double* src, double* dst, int len);

}} //cv::hal

#endif //__OPENCV_HAL_HPP__
