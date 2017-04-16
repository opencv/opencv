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
//
// @Authors
//    Fangfang Bai, fangfang@multicorewareinc.com
//    Jin Ma,       jin@multicorewareinc.com
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

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

///////////// BlendLinear ////////////////////////

typedef Size_MatType BlendLinearFixture;

OCL_PERF_TEST_P(BlendLinearFixture, BlendLinear, ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134))
{
    Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int srcType = get<1>(params);
    const double eps = CV_MAT_DEPTH(srcType) <= CV_32S ? 1.0 : 0.2;

    checkDeviceMaxMemoryAllocSize(srcSize, srcType);

    UMat src1(srcSize, srcType), src2(srcSize, srcType), dst(srcSize, srcType);
    UMat weights1(srcSize, CV_32FC1), weights2(srcSize, CV_32FC1);

    declare.in(src1, src2, WARMUP_RNG).in(weights1, weights2, WARMUP_READ).out(dst);
    randu(weights1, 0, 1);
    randu(weights2, 0, 1);

    OCL_TEST_CYCLE() cv::blendLinear(src1, src2, weights1, weights2, dst);

    SANITY_CHECK(dst, eps);
}

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
