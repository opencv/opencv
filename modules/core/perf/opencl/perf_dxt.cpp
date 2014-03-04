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
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

///////////// dft ////////////////////////

typedef tuple<Size, int> DftParams;
typedef TestBaseWithParam<DftParams> DftFixture;

OCL_PERF_TEST_P(DftFixture, Dft, ::testing::Combine(Values(OCL_SIZE_1, OCL_SIZE_2, OCL_SIZE_3),
                                                Values((int)DFT_ROWS, (int)DFT_SCALE, (int)DFT_INVERSE,
                                                       (int)DFT_INVERSE | DFT_SCALE, (int)DFT_ROWS | DFT_INVERSE)))
{
    const DftParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int flags = get<1>(params);

    UMat src(srcSize, CV_32FC2), dst(srcSize, CV_32FC2);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::dft(src, dst, flags | DFT_COMPLEX_OUTPUT);

    SANITY_CHECK(dst, 1e-3);
}

///////////// MulSpectrums ////////////////////////

typedef tuple<Size, bool> MulSpectrumsParams;
typedef TestBaseWithParam<MulSpectrumsParams> MulSpectrumsFixture;

OCL_PERF_TEST_P(MulSpectrumsFixture, MulSpectrums,
                ::testing::Combine(Values(OCL_SIZE_1, OCL_SIZE_2, OCL_SIZE_3),
                                   Bool()))
{
    const MulSpectrumsParams params = GetParam();
    const Size srcSize = get<0>(params);
    const bool conj = get<1>(params);

    UMat src1(srcSize, CV_32FC2), src2(srcSize, CV_32FC2), dst(srcSize, CV_32FC2);
    declare.in(src1, src2, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::mulSpectrums(src1, src2, dst, 0, conj);

    SANITY_CHECK(dst, 1e-3);
}

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
