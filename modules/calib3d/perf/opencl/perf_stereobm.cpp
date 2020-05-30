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

namespace opencv_test {
namespace ocl {

typedef tuple<int, int> StereoBMFixture_t;
typedef TestBaseWithParam<StereoBMFixture_t> StereoBMFixture;

OCL_PERF_TEST_P(StereoBMFixture, StereoBM, ::testing::Combine(OCL_PERF_ENUM(32, 64, 128), OCL_PERF_ENUM(11,21) ) )
{
    const int n_disp = get<0>(GetParam()), winSize = get<1>(GetParam());
    UMat left, right, disp;

    imread(getDataPath("gpu/stereobm/aloe-L.png"), IMREAD_GRAYSCALE).copyTo(left);
    imread(getDataPath("gpu/stereobm/aloe-R.png"), IMREAD_GRAYSCALE).copyTo(right);
    ASSERT_FALSE(left.empty());
    ASSERT_FALSE(right.empty());

    declare.in(left, right);

    Ptr<StereoBM> bm = StereoBM::create( n_disp, winSize );
    bm->setPreFilterType(bm->PREFILTER_XSOBEL);
    bm->setTextureThreshold(0);

    OCL_TEST_CYCLE() bm->compute(left, right, disp);

    SANITY_CHECK(disp, 1e-3, ERROR_RELATIVE);
}

}//ocl
}//cvtest
#endif
