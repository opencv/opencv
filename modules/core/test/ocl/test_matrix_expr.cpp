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
//    Jia Haipeng, jiahaipeng95@gmail.com
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

#include "test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

//////////////////////////////// UMat Expressions /////////////////////////////////////////////////

PARAM_TEST_CASE(UMatExpr, MatDepth, Channels)
{
    int type;
    Size size;

    virtual void SetUp()
    {
        type = CV_MAKE_TYPE(GET_PARAM(0), GET_PARAM(1));
    }

    void generateTestData()
    {
        size = randomSize(1, MAX_VALUE);
    }
};

//////////////////////////////// UMat::eye /////////////////////////////////////////////////

OCL_TEST_P(UMatExpr, Eye)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        Mat m = Mat::eye(size, type);
        UMat um = UMat::eye(size, type);

        EXPECT_MAT_NEAR(m, um, 0);
    }
}

//////////////////////////////// UMat::zeros /////////////////////////////////////////////////

OCL_TEST_P(UMatExpr, Zeros)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        Mat m = Mat::zeros(size, type);
        UMat um = UMat::zeros(size, type);

        EXPECT_MAT_NEAR(m, um, 0);
    }
}

//////////////////////////////// UMat::ones /////////////////////////////////////////////////

OCL_TEST_P(UMatExpr, Ones)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        Mat m = Mat::ones(size, type);
        UMat um = UMat::ones(size, type);

        EXPECT_MAT_NEAR(m, um, 0);
    }
}

//////////////////////////////// Instantiation /////////////////////////////////////////////////

OCL_INSTANTIATE_TEST_CASE_P(MatrixOperation, UMatExpr, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS));

} } // namespace cvtest::ocl

#endif
