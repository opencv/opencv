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

//////////////////////////////// LUT /////////////////////////////////////////////////

PARAM_TEST_CASE(Lut, MatDepth, MatDepth, Channels, bool, bool)
{
    int src_depth, lut_depth;
    int cn;
    bool use_roi, same_cn;

    TEST_DECLARE_INPUT_PARAMETER(src)
    TEST_DECLARE_INPUT_PARAMETER(lut)
    TEST_DECLARE_OUTPUT_PARAMETER(dst)

    virtual void SetUp()
    {
        src_depth = GET_PARAM(0);
        lut_depth = GET_PARAM(1);
        cn = GET_PARAM(2);
        same_cn = GET_PARAM(3);
        use_roi = GET_PARAM(4);
    }

    void generateTestData()
    {
        const int src_type = CV_MAKE_TYPE(src_depth, cn);
        const int lut_type = CV_MAKE_TYPE(lut_depth, same_cn ? cn : 1);
        const int dst_type = CV_MAKE_TYPE(lut_depth, cn);

        Size roiSize = randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, src_type, 0, 256);

        Size lutRoiSize = Size(256, 1);
        Border lutBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(lut, lut_roi, lutRoiSize, lutBorder, lut_type, 5, 16);

        Border dstBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, dst_type, 5, 16);

        UMAT_UPLOAD_INPUT_PARAMETER(src)
        UMAT_UPLOAD_INPUT_PARAMETER(lut)
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst)
    }

    void Near(double threshold = 0.)
    {
        EXPECT_MAT_NEAR(dst, udst, threshold);
        EXPECT_MAT_NEAR(dst_roi, udst_roi, threshold);
    }
};

OCL_TEST_P(Lut, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::LUT(src_roi, lut_roi, dst_roi));
        OCL_ON(cv::LUT(usrc_roi, ulut_roi, udst_roi));

        Near();
    }
}

///////////////////////// ArithmTestBase ///////////////////////////

PARAM_TEST_CASE(ArithmTestBase, MatDepth, Channels, bool)
{
    int depth;
    int cn;
    bool use_roi;
    cv::Scalar val;

    // declare Mat + UMat mirrors
    TEST_DECLARE_INPUT_PARAMETER(src1)
    TEST_DECLARE_INPUT_PARAMETER(src2)
    TEST_DECLARE_INPUT_PARAMETER(mask)
    TEST_DECLARE_OUTPUT_PARAMETER(dst1)
    TEST_DECLARE_OUTPUT_PARAMETER(dst2)

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
        OCL_ON(cv::add(val, usrc1_roi, udst1_roi));
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

//////////////////////////////////////// Subtract //////////////////////////////////////////////

typedef ArithmTestBase Subtract;

OCL_TEST_P(Subtract, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::subtract(src1_roi, src2_roi, dst1_roi));
        OCL_ON(cv::subtract(usrc1_roi, usrc2_roi, udst1_roi));
        Near(0);
    }
}

OCL_TEST_P(Subtract, Mat_Mask)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::subtract(src1_roi, src2_roi, dst1_roi, mask_roi));
        OCL_ON(cv::subtract(usrc1_roi, usrc2_roi, udst1_roi, umask_roi));
        Near(0);
    }
}

OCL_TEST_P(Subtract, Scalar)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::subtract(val, src1_roi, dst1_roi));
        OCL_ON(cv::subtract(val, usrc1_roi, udst1_roi));
        Near(1e-5);
    }
}

OCL_TEST_P(Subtract, Scalar_Mask)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::subtract(src1_roi, val, dst1_roi, mask_roi));
        OCL_ON(cv::subtract(usrc1_roi, val, udst1_roi, umask_roi));
        Near(1e-5);
    }
}

//////////////////////////////////////// Log /////////////////////////////////////////

typedef ArithmTestBase Log;

OCL_TEST_P(Log, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::log(src1_roi, dst1_roi));
        OCL_ON(cv::log(usrc1_roi, udst1_roi));
        Near(1);
    }
}

//////////////////////////////////////// Exp /////////////////////////////////////////

typedef ArithmTestBase Exp;

OCL_TEST_P(Exp, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::exp(src1_roi, dst1_roi));
        OCL_ON(cv::exp(usrc1_roi, udst1_roi));
        Near(2);
    }
}

//////////////////////////////////////// Phase /////////////////////////////////////////

typedef ArithmTestBase Phase;

OCL_TEST_P(Phase, angleInDegree)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::phase(src1_roi, src2_roi, dst1_roi, true));
        OCL_ON(cv::phase(usrc1_roi, usrc2_roi, udst1_roi, true));
        Near(1e-2);
    }
}

OCL_TEST_P(Phase, angleInRadians)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::phase(src1_roi, src2_roi, dst1_roi));
        OCL_ON(cv::phase(usrc1_roi, usrc2_roi, udst1_roi));
        Near(1e-2);
    }
}

//////////////////////////////////////// Magnitude /////////////////////////////////////////

typedef ArithmTestBase Magnitude;

OCL_TEST_P(Magnitude, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::magnitude(src1_roi, src2_roi, dst1_roi));
        OCL_ON(cv::magnitude(usrc1_roi, usrc2_roi, udst1_roi));
        Near(depth == CV_64F ? 1e-5 : 1e-2);
    }
}

//////////////////////////////////////// Instantiation /////////////////////////////////////////

OCL_INSTANTIATE_TEST_CASE_P(Arithm, Lut, Combine(::testing::Values(CV_8U, CV_8S), OCL_ALL_DEPTHS, ::testing::Values(1, 2, 3, 4), Bool(), Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Add, Combine(OCL_ALL_DEPTHS, ::testing::Values(1, 2, 4), Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Subtract, Combine(OCL_ALL_DEPTHS, ::testing::Values(1, 2, 4), Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Log, Combine(::testing::Values(CV_32F, CV_64F), ::testing::Values(1, 2, 3, 4), Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Exp, Combine(::testing::Values(CV_32F, CV_64F), ::testing::Values(1, 2, 3, 4), Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Phase, Combine(::testing::Values(CV_32F, CV_64F), ::testing::Values(1, 2, 3, 4), Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Magnitude, Combine(::testing::Values(CV_32F, CV_64F), ::testing::Values(1, 2, 3, 4), Bool()));

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
