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
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
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
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
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

///////////////////////// ArithmTestBase ///////////////////////////

PARAM_TEST_CASE(ArithmTestBase, MatDepth, Channels, bool)
{
    int depth;
    int cn;
    bool use_roi;
    cv::Scalar val;

    // declare Mat + UMat mirrors
    TEST_DECLARE_INPUT_PARATEMER(src1)
    TEST_DECLARE_INPUT_PARATEMER(src2)
    TEST_DECLARE_INPUT_PARATEMER(mask)
    TEST_DECLARE_OUTPUT_PARATEMER(dst1)
    TEST_DECLARE_OUTPUT_PARATEMER(dst2)

    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        cn = GET_PARAM(1);
        use_roi = GET_PARAM(2);
    }

    virtual void generateTestData()
    {
        const int type = CV_MAKE_TYPE(depth, cn);

        Size roiSize = randomSize(1, MAX_VALUE);
        Border src1Border = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src1, src1_roi, roiSize, src1Border, type, 2, 11);

        Border src2Border = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src2, src2_roi, roiSize, src2Border, type, -1540, 1740);

        Border dst1Border = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst1, dst1_roi, roiSize, dst1Border, type, 5, 16);

        Border dst2Border = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst2, dst2_roi, roiSize, dst2Border, type, 5, 16);

        Border maskBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(mask, mask_roi, roiSize, maskBorder, CV_8UC1, 0, 2);
        cv::threshold(mask, mask, 0.5, 255., CV_8UC1);

        val = cv::Scalar(rng.uniform(-100.0, 100.0), rng.uniform(-100.0, 100.0),
                         rng.uniform(-100.0, 100.0), rng.uniform(-100.0, 100.0));

        UMAT_UPLOAD_INPUT_PARAMETER(src1)
        UMAT_UPLOAD_INPUT_PARAMETER(src2)
        UMAT_UPLOAD_INPUT_PARAMETER(mask)
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst1)
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst2)
    }

    void Near(double threshold = 0.)
    {
        EXPECT_MAT_NEAR(dst1, udst1, threshold);
        EXPECT_MAT_NEAR(dst1_roi, udst1_roi, threshold);
    }

    void Near1(double threshold = 0.)
    {
        EXPECT_MAT_NEAR(dst2, udst2, threshold);
        EXPECT_MAT_NEAR(dst2_roi, udst2_roi, threshold);
    }
};

//////////////////////////////// Add /////////////////////////////////////////////////

typedef ArithmTestBase Add;

OCL_TEST_P(Add, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::add(src1_roi, src2_roi, dst1_roi));
        OCL_ON(cv::add(usrc1_roi, usrc2_roi, udst1_roi));
        Near(0);
    }
}

OCL_TEST_P(Add, Mat_Mask)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::add(src1_roi, src2_roi, dst1_roi, mask_roi));
        OCL_ON(cv::add(usrc1_roi, usrc2_roi, udst1_roi, umask_roi));
        Near(0);
    }
}

OCL_TEST_P(Add, Scalar)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::add(src1_roi, val, dst1_roi));
        OCL_ON(cv::add(usrc1_roi, val, udst1_roi));
        Near(1e-5);
    }
}

OCL_TEST_P(Add, Scalar_Mask)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::add(src1_roi, val, dst1_roi, mask_roi));
        OCL_ON(cv::add(usrc1_roi, val, udst1_roi, umask_roi));
        Near(1e-5);
    }
}



//////////////////////////////////////// Instantiation /////////////////////////////////////////

OCL_INSTANTIATE_TEST_CASE_P(Arithm, Add, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
