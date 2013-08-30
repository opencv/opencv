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
#include "perf_precomp.hpp"

using namespace perf;
using std::tr1::tuple;
using std::tr1::get;

///////////// ConvertTo////////////////////////

typedef Size_MatType ConvertToFixture;

PERF_TEST_P(ConvertToFixture, ConvertTo,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_8UC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src(srcSize, type), dst;
    const int dstType = CV_MAKE_TYPE(CV_32F, src.channels());
    dst.create(srcSize, dstType);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, dstType);

        OCL_TEST_CYCLE() oclSrc.convertTo(oclDst, dstType);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() src.convertTo(dst, dstType);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// copyTo////////////////////////

typedef Size_MatType copyToFixture;

PERF_TEST_P(copyToFixture, copyTo,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_8UC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        OCL_TEST_CYCLE() oclSrc.copyTo(oclDst);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() src.copyTo(dst);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// setTo////////////////////////

typedef Size_MatType setToFixture;

PERF_TEST_P(setToFixture, setTo,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               OCL_PERF_ENUM(CV_8UC1, CV_8UC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    const Scalar val(1, 2, 3, 4);

    Mat src(srcSize, type);
    declare.in(src);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(srcSize, type);

        OCL_TEST_CYCLE() oclSrc.setTo(val);
        oclSrc.download(src);

        SANITY_CHECK(src);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() src.setTo(val);

        SANITY_CHECK(src);
    }
    else
        OCL_PERF_ELSE
}
