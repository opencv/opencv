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
//    Peng Xiao, pengxiao@multicorewareinc.com
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

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

////////////////////////////////////////////////////////////////////////////
// GEMM

PARAM_TEST_CASE(Gemm,
                MatType,
                bool, // GEMM_1_T
                bool, // GEMM_2_T
                bool, // GEMM_3_T
                bool // ROI
                )
{
    bool use_roi;
    int type, flags;
    bool atrans, btrans, ctrans;

    double alpha, beta;

    int M, N, K;

    TEST_DECLARE_INPUT_PARAMETER(A);
    TEST_DECLARE_INPUT_PARAMETER(B);
    TEST_DECLARE_INPUT_PARAMETER(C);
    TEST_DECLARE_OUTPUT_PARAMETER(D);

    virtual void SetUp()
    {
        atrans = btrans = ctrans = false;

        type = GET_PARAM(0);
        use_roi = GET_PARAM(4);

        flags = 0;
        if (GET_PARAM(1))
            flags |= GEMM_1_T, atrans = true;
        if (GET_PARAM(2))
            flags |= GEMM_2_T, btrans = true;
        if (GET_PARAM(3))
            flags |= GEMM_3_T, ctrans = true;
    }

    void generateTestData()
    {
        M = (int)randomDoubleLog(1, 100);
        N = (int)randomDoubleLog(1, 100);
        K = (int)randomDoubleLog(1, 1200);

        M = roundUp(M, 1);
        N = roundUp(N, 1);
        K = roundUp(K, 1);

        Size ARoiSize = (atrans) ? Size(M, K) : Size(K, M);
        Border ABorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(A, A_roi, ARoiSize, ABorder, type, -11, 11);

        Size BRoiSize = (btrans) ? Size(K, N) : Size(N, K);
        Border BBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(B, B_roi, BRoiSize, BBorder, type, -11, 11);

        Size CRoiSize = (ctrans) ? Size(M, N) : Size(N, M);
        Border CBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(C, C_roi, CRoiSize, CBorder, type, -11, 11);

        Size DRoiSize = Size(N, M);
        Border DBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(D, D_roi, DRoiSize, DBorder, type, -11, 11);

        alpha = randomDouble(-4, 4);
        beta = randomDouble(-4, 4);

        UMAT_UPLOAD_INPUT_PARAMETER(A);
        UMAT_UPLOAD_INPUT_PARAMETER(B);
        UMAT_UPLOAD_INPUT_PARAMETER(C);
        UMAT_UPLOAD_OUTPUT_PARAMETER(D);
    }
};

OCL_TEST_P(Gemm, Accuracy)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        generateTestData();
        SCOPED_TRACE(cv::format("i=%d: M=%d N=%d K=%d", i, M, N, K));

        OCL_OFF(cv::gemm(A_roi, B_roi, alpha, C_roi, beta, D_roi, flags));
        OCL_ON(cv::gemm(uA_roi, uB_roi, alpha, uC_roi, beta, uD_roi, flags));

        double eps = D_roi.size().area() * (1e-5 * K);
        OCL_EXPECT_MATS_NEAR(D, eps);
    }
}

OCL_INSTANTIATE_TEST_CASE_P(Core, Gemm, ::testing::Combine(
                            testing::Values(CV_32FC1, CV_32FC2, CV_64FC1, CV_64FC2),
                            Bool(), Bool(), Bool(), Bool()));

// Test for non-Intel GPUs to check CL_INVALID_WORK_GROUP_SIZE when localsize > globalsize
OCL_TEST(Gemm, small)
{
    UMat A(2, 3, CV_32F), B(4, 3, CV_32F), uC(2, 4, CV_32F);
    Mat C(2, 4, CV_32F);

    randu(A, -1, 1);
    randu(B, -1, 1);

    OCL_OFF(cv::gemm(A, B, 1, noArray(), 0, C, GEMM_2_T));
    OCL_ON(cv::gemm(A, B, 1, noArray(), 0, uC, GEMM_2_T));

    EXPECT_LE(cvtest::norm(C, uC, cv::NORM_INF), 1e-5);
}

} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL
