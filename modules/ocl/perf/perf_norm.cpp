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

///////////// norm////////////////////////

typedef TestBaseWithParam<Size> normFixture;

PERF_TEST_P(normFixture, DISABLED_norm, OCL_TYPICAL_MAT_SIZES) // TODO doesn't work properly
{
    const Size srcSize = GetParam();
    const std::string impl = getSelectedImpl();
    double value = 0.0;

    Mat src1(srcSize, CV_8UC1), src2(srcSize, CV_8UC1);
    declare.in(src1, src2);
    randu(src1, 0, 1);
    randu(src2, 0, 1);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc1(src1), oclSrc2(src2);

        TEST_CYCLE() value = cv::ocl::norm(oclSrc1, oclSrc2, NORM_INF);

        SANITY_CHECK(value);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() value = cv::norm(src1, src2, NORM_INF);

        SANITY_CHECK(value);
    }
    else
        OCL_PERF_ELSE
}
