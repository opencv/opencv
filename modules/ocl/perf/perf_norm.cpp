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
#include "perf_precomp.hpp"

using namespace perf;
using std::tr1::tuple;
using std::tr1::get;

///////////// norm////////////////////////

CV_ENUM(NormType, NORM_INF, NORM_L1, NORM_L2)

typedef std::tr1::tuple<Size, MatType, NormType> NormParams;
typedef TestBaseWithParam<NormParams> NormFixture;

OCL_PERF_TEST_P(NormFixture, Norm,
                ::testing::Combine(OCL_PERF_ENUM(OCL_SIZE_1, OCL_SIZE_2, OCL_SIZE_3),
                                   OCL_TEST_TYPES, NormType::all()))
{
    const NormParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const int normType = get<2>(params);
    perf::ERROR_TYPE errorType = type != NORM_INF ? ERROR_RELATIVE : ERROR_ABSOLUTE;
    double eps = 1e-5, value;

    Mat src1(srcSize, type), src2(srcSize, type);
    declare.in(src1, src2, WARMUP_RNG);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2);

        OCL_TEST_CYCLE() value = cv::ocl::norm(oclSrc1, oclSrc2, normType);

        SANITY_CHECK(value, eps, errorType);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() value = cv::norm(src1, src2, normType);

        SANITY_CHECK(value, eps, errorType);
    }
    else
        OCL_PERF_ELSE
}
