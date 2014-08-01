///////////////////////////////////////////////////////////////////////////////////////
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

#include "../test_precomp.hpp"
#include "cvconfig.h"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

PARAM_TEST_CASE(StereoBMFixture, int, int)
{
    int n_disp;
    int winSize;
    Mat left, right, disp;
    UMat uleft, uright, udisp;

    virtual void SetUp()
    {
        n_disp  = GET_PARAM(0);
        winSize = GET_PARAM(1);

        left  = readImage("gpu/stereobm/aloe-L.png", IMREAD_GRAYSCALE);
        right = readImage("gpu/stereobm/aloe-R.png", IMREAD_GRAYSCALE);

        ASSERT_FALSE(left.empty());
        ASSERT_FALSE(right.empty());

        left.copyTo(uleft);
        right.copyTo(uright);
    }

    void Near(double eps = 0.0)
    {
        EXPECT_MAT_NEAR_RELATIVE(disp, udisp, eps);
    }
};

OCL_TEST_P(StereoBMFixture, StereoBM)
{
    Ptr<StereoBM> bm = createStereoBM( n_disp, winSize);
    bm->setPreFilterType(bm->PREFILTER_XSOBEL);
    bm->setTextureThreshold(0);

    OCL_OFF(bm->compute(left, right, disp));
    OCL_ON(bm->compute(uleft, uright, udisp));

    Near(1e-3);
}

OCL_INSTANTIATE_TEST_CASE_P(StereoMatcher, StereoBMFixture, testing::Combine(testing::Values(32, 64, 128),
                                       testing::Values(11, 21)));
}//ocl
}//cvtest

#endif //HAVE_OPENCL
