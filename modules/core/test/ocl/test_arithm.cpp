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
        OCL_EXPECT_MATS_NEAR(dst, threshold)
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
        OCL_EXPECT_MATS_NEAR(dst1, threshold)
    }

    void Near1(double threshold = 0.)
    {
        OCL_EXPECT_MATS_NEAR(dst2, threshold)
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

//////////////////////////////// Mul /////////////////////////////////////////////////

typedef ArithmTestBase Mul;

OCL_TEST_P(Mul, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::multiply(src1_roi, src2_roi, dst1_roi));
        OCL_ON(cv::multiply(usrc1_roi, usrc2_roi, udst1_roi));
        Near(0);
    }
}

OCL_TEST_P(Mul, Scalar)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::multiply(src1_roi, val, dst1_roi));
        OCL_ON(cv::multiply(val, usrc1_roi, udst1_roi));

        Near(udst1_roi.depth() >= CV_32F ? 1e-3 : 1);
    }
}

OCL_TEST_P(Mul, Mat_Scale)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::multiply(src1_roi, src2_roi, dst1_roi, val[0]));
        OCL_ON(cv::multiply(usrc1_roi, usrc2_roi, udst1_roi, val[0]));

        Near(udst1_roi.depth() >= CV_32F ? 1e-3 : 1);
    }
}

OCL_TEST_P(Mul, Mat_Scalar_Scale)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::multiply(src1_roi, val, dst1_roi, val[0]));
        OCL_ON(cv::multiply(usrc1_roi, val, udst1_roi, val[0]));

        Near(udst1_roi.depth() >= CV_32F ? 1e-2 : 1);
    }
}


//////////////////////////////// Div /////////////////////////////////////////////////

typedef ArithmTestBase Div;

OCL_TEST_P(Div, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::divide(src1_roi, src2_roi, dst1_roi));
        OCL_ON(cv::divide(usrc1_roi, usrc2_roi, udst1_roi));
        Near(1);
    }
}

OCL_TEST_P(Div, Scalar)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::divide(val, src1_roi, dst1_roi));
        OCL_ON(cv::divide(val, usrc1_roi, udst1_roi));

        Near(udst1_roi.depth() >= CV_32F ? 1e-3 : 1);
    }
}

OCL_TEST_P(Div, Scalar2)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::divide(src1_roi, val, dst1_roi));
        OCL_ON(cv::divide(usrc1_roi, val, udst1_roi));

        Near(udst1_roi.depth() >= CV_32F ? 1e-3 : 1);
    }
}

OCL_TEST_P(Div, Mat_Scale)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::divide(src1_roi, src2_roi, dst1_roi, val[0]));
        OCL_ON(cv::divide(usrc1_roi, usrc2_roi, udst1_roi, val[0]));

        Near(udst1_roi.depth() >= CV_32F ? 4e-3 : 1);
    }
}

OCL_TEST_P(Div, Mat_Scalar_Scale)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::divide(src1_roi, val, dst1_roi, val[0]));
        OCL_ON(cv::divide(usrc1_roi, val, udst1_roi, val[0]));

        Near(udst1_roi.depth() >= CV_32F ? 4e-3 : 1);
    }
}

OCL_TEST_P(Div, Recip)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::divide(val[0], src1_roi, dst1_roi));
        OCL_ON(cv::divide(val[0], usrc1_roi, udst1_roi));

        Near(udst1_roi.depth() >= CV_32F ? 1e-3 : 1);
    }
}

//////////////////////////////// Min/Max /////////////////////////////////////////////////

typedef ArithmTestBase Min;

OCL_TEST_P(Min, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::max(src1_roi, src2_roi, dst1_roi));
        OCL_ON(cv::max(usrc1_roi, usrc2_roi, udst1_roi));
        Near(0);
    }
}

typedef ArithmTestBase Max;

OCL_TEST_P(Max, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::min(src1_roi, src2_roi, dst1_roi));
        OCL_ON(cv::min(usrc1_roi, usrc2_roi, udst1_roi));
        Near(0);
    }
}

//////////////////////////////// Absdiff /////////////////////////////////////////////////

typedef ArithmTestBase Absdiff;

OCL_TEST_P(Absdiff, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::absdiff(src1_roi, src2_roi, dst1_roi));
        OCL_ON(cv::absdiff(usrc1_roi, usrc2_roi, udst1_roi));
        Near(0);
    }
}

OCL_TEST_P(Absdiff, Scalar)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::absdiff(src1_roi, val, dst1_roi));
        OCL_ON(cv::absdiff(usrc1_roi, val, udst1_roi));
        Near(1e-5);
    }
}

//////////////////////////////// CartToPolar /////////////////////////////////////////////////

typedef ArithmTestBase CartToPolar;

OCL_TEST_P(CartToPolar, angleInDegree)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::cartToPolar(src1_roi, src2_roi, dst1_roi, dst2_roi, true));
        OCL_ON(cv::cartToPolar(usrc1_roi, usrc2_roi, udst1_roi, udst2_roi, true));
        Near(0.5);
        Near1(0.5);
    }
}

OCL_TEST_P(CartToPolar, angleInRadians)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::cartToPolar(src1_roi, src2_roi, dst1_roi, dst2_roi));
        OCL_ON(cv::cartToPolar(usrc1_roi, usrc2_roi, udst1_roi, udst2_roi));
        Near(0.5);
        Near1(0.5);
    }
}

//////////////////////////////// PolarToCart /////////////////////////////////////////////////

typedef ArithmTestBase PolarToCart;

OCL_TEST_P(PolarToCart, angleInDegree)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::polarToCart(src1_roi, src2_roi, dst1_roi, dst2_roi, true));
        OCL_ON(cv::polarToCart(usrc1_roi, usrc2_roi, udst1_roi, udst2_roi, true));

        Near(0.5);
        Near1(0.5);
    }
}

OCL_TEST_P(PolarToCart, angleInRadians)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::polarToCart(src1_roi, src2_roi, dst1_roi, dst2_roi));
        OCL_ON(cv::polarToCart(usrc1_roi, usrc2_roi, udst1_roi, udst2_roi));

        Near(0.5);
        Near1(0.5);
    }
}

//////////////////////////////// Transpose /////////////////////////////////////////////////

typedef ArithmTestBase Transpose;

OCL_TEST_P(Transpose, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::transpose(src1_roi, dst1_roi));
        OCL_ON(cv::transpose(usrc1_roi, udst1_roi));

        Near(1e-5);
    }
}

OCL_TEST_P(Transpose, SquareInplace)
{
    const int type = CV_MAKE_TYPE(depth, cn);

    for (int j = 0; j < test_loop_times; j++)
    {
        Size roiSize = randomSize(1, MAX_VALUE);
        roiSize.height = roiSize.width; // make it square

        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src1, src1_roi, roiSize, srcBorder, type, 5, 16);

        UMAT_UPLOAD_OUTPUT_PARAMETER(src1);

        OCL_OFF(cv::transpose(src1_roi, src1_roi));
        OCL_ON(cv::transpose(usrc1_roi, usrc1_roi));

        OCL_EXPECT_MATS_NEAR(src1, 0)
    }
}

//////////////////////////////// Bitwise_and /////////////////////////////////////////////////

typedef ArithmTestBase Bitwise_and;

OCL_TEST_P(Bitwise_and, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::bitwise_and(src1_roi, src2_roi, dst1_roi));
        OCL_ON(cv::bitwise_and(usrc1_roi, usrc2_roi, udst1_roi));
        Near(0);
    }
}

OCL_TEST_P(Bitwise_and, Mat_Mask)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::bitwise_and(src1_roi, src2_roi, dst1_roi, mask_roi));
        OCL_ON(cv::bitwise_and(usrc1_roi, usrc2_roi, udst1_roi, umask_roi));
        Near(0);
    }
}

OCL_TEST_P(Bitwise_and, Scalar)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::bitwise_and(src1_roi, val, dst1_roi));
        OCL_ON(cv::bitwise_and(usrc1_roi, val, udst1_roi));
        Near(1e-5);
    }
}

OCL_TEST_P(Bitwise_and, Scalar_Mask)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::bitwise_and(src1_roi, val, dst1_roi, mask_roi));
        OCL_ON(cv::bitwise_and(usrc1_roi, val, udst1_roi, umask_roi));
        Near(1e-5);
    }
}

//////////////////////////////// Bitwise_or /////////////////////////////////////////////////

typedef ArithmTestBase Bitwise_or;

OCL_TEST_P(Bitwise_or, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::bitwise_or(src1_roi, src2_roi, dst1_roi));
        OCL_ON(cv::bitwise_or(usrc1_roi, usrc2_roi, udst1_roi));
        Near(0);
    }
}

OCL_TEST_P(Bitwise_or, Mat_Mask)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::bitwise_or(src1_roi, src2_roi, dst1_roi, mask_roi));
        OCL_ON(cv::bitwise_or(usrc1_roi, usrc2_roi, udst1_roi, umask_roi));
        Near(0);
    }
}

OCL_TEST_P(Bitwise_or, Scalar)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::bitwise_or(src1_roi, val, dst1_roi));
        OCL_ON(cv::bitwise_or(usrc1_roi, val, udst1_roi));
        Near(1e-5);
    }
}

OCL_TEST_P(Bitwise_or, Scalar_Mask)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::bitwise_or(src1_roi, val, dst1_roi, mask_roi));
        OCL_ON(cv::bitwise_or(val, usrc1_roi, udst1_roi, umask_roi));
        Near(1e-5);
    }
}

//////////////////////////////// Bitwise_xor /////////////////////////////////////////////////

typedef ArithmTestBase Bitwise_xor;

OCL_TEST_P(Bitwise_xor, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::bitwise_xor(src1_roi, src2_roi, dst1_roi));
        OCL_ON(cv::bitwise_xor(usrc1_roi, usrc2_roi, udst1_roi));
        Near(0);
    }
}

OCL_TEST_P(Bitwise_xor, Mat_Mask)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::bitwise_xor(src1_roi, src2_roi, dst1_roi, mask_roi));
        OCL_ON(cv::bitwise_xor(usrc1_roi, usrc2_roi, udst1_roi, umask_roi));
        Near(0);
    }
}

OCL_TEST_P(Bitwise_xor, Scalar)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::bitwise_xor(src1_roi, val, dst1_roi));
        OCL_ON(cv::bitwise_xor(usrc1_roi, val, udst1_roi));
        Near(1e-5);
    }
}

OCL_TEST_P(Bitwise_xor, Scalar_Mask)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::bitwise_xor(src1_roi, val, dst1_roi, mask_roi));
        OCL_ON(cv::bitwise_xor(usrc1_roi, val, udst1_roi, umask_roi));
        Near(1e-5);
    }
}

//////////////////////////////// Bitwise_not /////////////////////////////////////////////////

typedef ArithmTestBase Bitwise_not;

OCL_TEST_P(Bitwise_not, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::bitwise_not(src1_roi, dst1_roi));
        OCL_ON(cv::bitwise_not(usrc1_roi, udst1_roi));
        Near(0);
    }
}

//////////////////////////////// Compare /////////////////////////////////////////////////

typedef ArithmTestBase Compare;

OCL_TEST_P(Compare, Mat)
{
    int cmp_codes[] = { CMP_EQ, CMP_GT, CMP_GE, CMP_LT, CMP_LE, CMP_NE };
    int cmp_num = sizeof(cmp_codes) / sizeof(int);

    for (int i = 0; i < cmp_num; ++i)
        for (int j = 0; j < test_loop_times; j++)
        {
            generateTestData();

            OCL_OFF(cv::compare(src1_roi, src2_roi, dst1_roi, cmp_codes[i]));
            OCL_ON(cv::compare(usrc1_roi, usrc2_roi, udst1_roi, cmp_codes[i]));

            Near(0);
        }
}

//////////////////////////////// Pow /////////////////////////////////////////////////

typedef ArithmTestBase Pow;

OCL_TEST_P(Pow, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();
        double p = 4.5;

        OCL_OFF(cv::pow(src1_roi, p, dst1_roi));
        OCL_ON(cv::pow(usrc1_roi, p, udst1_roi));

        Near(1);
    }
}

//////////////////////////////// AddWeighted /////////////////////////////////////////////////

typedef ArithmTestBase AddWeighted;

OCL_TEST_P(AddWeighted, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        const double alpha = 2.0, beta = 1.0, gama = 3.0;

        OCL_OFF(cv::addWeighted(src1_roi, alpha, src2_roi, beta, gama, dst1_roi));
        OCL_ON(cv::addWeighted(usrc1_roi, alpha, usrc2_roi, beta, gama, udst1_roi));

        Near(3e-4);
    }
}

//////////////////////////////// setIdentity /////////////////////////////////////////////////

typedef ArithmTestBase SetIdentity;

OCL_TEST_P(SetIdentity, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::setIdentity(dst1_roi, val));
        OCL_ON(cv::setIdentity(udst1_roi, val));

        Near(0);
    }
}

//// Repeat

struct RepeatTestCase :
        public ArithmTestBase
{
    int nx, ny;

    virtual void generateTestData()
    {
        const int type = CV_MAKE_TYPE(depth, cn);

        nx = randomInt(1, 4);
        ny = randomInt(1, 4);

        Size srcRoiSize = randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src1, src1_roi, srcRoiSize, srcBorder, type, 2, 11);

        Size dstRoiSize(srcRoiSize.width * nx, srcRoiSize.height * ny);
        Border dst1Border = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst1, dst1_roi, dstRoiSize, dst1Border, type, 5, 16);

        UMAT_UPLOAD_INPUT_PARAMETER(src1)
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst1)
    }
};

typedef RepeatTestCase Repeat;

OCL_TEST_P(Repeat, Mat)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        generateTestData();

        OCL_OFF(cv::repeat(src1_roi, ny, nx, dst1_roi));
        OCL_ON(cv::repeat(usrc1_roi, ny, nx, udst1_roi));

        Near();
    }
}

//////////////////////////////// CountNonZero /////////////////////////////////////////////////

typedef ArithmTestBase CountNonZero;

OCL_TEST_P(CountNonZero, MAT)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        int cpures, gpures;
        OCL_OFF(cpures = cv::countNonZero(src1_roi));
        OCL_ON(gpures = cv::countNonZero(usrc1_roi));

        EXPECT_EQ(cpures, gpures);
    }
}

//////////////////////////////// Sum /////////////////////////////////////////////////

typedef ArithmTestBase Sum;

OCL_TEST_P(Sum, MAT)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        Scalar cpures, gpures;
        OCL_OFF(cpures = cv::sum(src1_roi));
        OCL_ON(gpures = cv::sum(usrc1_roi));

        for (int i = 0; i < cn; ++i)
            EXPECT_NEAR(cpures[i], gpures[i], 0.1);
    }
}

//////////////////////////////// meanStdDev /////////////////////////////////////////////////

typedef ArithmTestBase MeanStdDev;

OCL_TEST_P(MeanStdDev, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        Scalar cpu_mean, cpu_stddev;
        Scalar gpu_mean, gpu_stddev;

        OCL_OFF(cv::meanStdDev(src1_roi, cpu_mean, cpu_stddev));
        OCL_ON(cv::meanStdDev(usrc1_roi, gpu_mean, gpu_stddev));

        for (int i = 0; i < cn; ++i)
        {
            EXPECT_NEAR(cpu_mean[i], gpu_mean[i], 0.1);
            EXPECT_NEAR(cpu_stddev[i], gpu_stddev[i], 0.1);
        }
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

//////////////////////////////// Flip /////////////////////////////////////////////////

typedef ArithmTestBase Flip;

OCL_TEST_P(Flip, X)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::flip(src1_roi, dst1_roi, 0));
        OCL_ON(cv::flip(usrc1_roi, udst1_roi, 0));
        Near(0);
    }
}

OCL_TEST_P(Flip, Y)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::flip(src1_roi, dst1_roi, 1));
        OCL_ON(cv::flip(usrc1_roi, udst1_roi, 1));
        Near(0);
    }
}

OCL_TEST_P(Flip, BOTH)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::flip(src1_roi, dst1_roi, -1));
        OCL_ON(cv::flip(usrc1_roi, udst1_roi, -1));
        Near(0);
    }
}
//////////////////////////////////////// minMaxIdx /////////////////////////////////////////

typedef ArithmTestBase MinMaxIdx;

OCL_TEST_P(MinMaxIdx, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        int p1[2], p2[2], up1[2], up2[2];
        double minv, maxv, uminv, umaxv;

        if(src1_roi.channels() > 1)
        {
            OCL_OFF(cv::minMaxIdx(src2_roi, &minv, &maxv) );
            OCL_ON(cv::minMaxIdx(usrc2_roi, &uminv, &umaxv));

            EXPECT_DOUBLE_EQ(minv, uminv);
            EXPECT_DOUBLE_EQ(maxv, umaxv);
        }
        else
        {
            OCL_OFF(cv::minMaxIdx(src2_roi, &minv, &maxv, p1, p2, noArray()));
            OCL_ON(cv::minMaxIdx(usrc2_roi, &uminv, &umaxv, up1, up2, noArray()));

            EXPECT_DOUBLE_EQ(minv, uminv);
            EXPECT_DOUBLE_EQ(maxv, umaxv);
            for( int i = 0; i < 2; i++)
            {
                EXPECT_EQ(p1[i], up1[i]);
                EXPECT_EQ(p2[i], up2[i]);
            }
        }
    }
}

typedef ArithmTestBase MinMaxIdx_Mask;

OCL_TEST_P(MinMaxIdx_Mask, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        int p1[2], p2[2], up1[2], up2[2];
        double minv, maxv, uminv, umaxv;

        OCL_OFF(cv::minMaxIdx(src2_roi, &minv, &maxv, p1, p2, mask_roi));
        OCL_ON(cv::minMaxIdx(usrc2_roi, &uminv, &umaxv, up1, up2, umask_roi));

        EXPECT_DOUBLE_EQ(minv, uminv);
        EXPECT_DOUBLE_EQ(maxv, umaxv);
        for( int i = 0; i < 2; i++)
        {
            EXPECT_EQ(p1[i], up1[i]);
            EXPECT_EQ(p2[i], up2[i]);
        }

    }
}

//////////////////////////////// Norm /////////////////////////////////////////////////

static bool relativeError(double actual, double expected, double eps)
{
    return std::abs(actual - expected) / actual < eps;
}

typedef ArithmTestBase Norm;

OCL_TEST_P(Norm, NORM_INF_1arg)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(const double cpuRes = cv::norm(src1_roi, NORM_INF));
        OCL_ON(const double gpuRes = cv::norm(usrc1_roi, NORM_INF));

        EXPECT_NEAR(cpuRes, gpuRes, 0.1);
    }
}

OCL_TEST_P(Norm, NORM_L1_1arg)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(const double cpuRes = cv::norm(src1_roi, NORM_L1));
        OCL_ON(const double gpuRes = cv::norm(usrc1_roi, NORM_L1));

        EXPECT_PRED3(relativeError, cpuRes, gpuRes, 1e-6);
    }
}

OCL_TEST_P(Norm, NORM_L2_1arg)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(const double cpuRes = cv::norm(src1_roi, NORM_L2));
        OCL_ON(const double gpuRes = cv::norm(usrc1_roi, NORM_L2));

        EXPECT_PRED3(relativeError, cpuRes, gpuRes, 1e-6);
    }
}

OCL_TEST_P(Norm, NORM_INF_2args)
{
    for (int relative = 0; relative < 2; ++relative)
        for (int j = 0; j < test_loop_times; j++)
        {
            generateTestData();

            int type = NORM_INF;
            if (relative == 1)
                type |= NORM_RELATIVE;

            OCL_OFF(const double cpuRes = cv::norm(src1_roi, src2_roi, type));
            OCL_ON(const double gpuRes = cv::norm(usrc1_roi, usrc2_roi, type));

            EXPECT_NEAR(cpuRes, gpuRes, 0.1);
        }
}

OCL_TEST_P(Norm, NORM_L1_2args)
{
    for (int relative = 0; relative < 2; ++relative)
        for (int j = 0; j < test_loop_times; j++)
        {
            generateTestData();

            int type = NORM_L1;
            if (relative == 1)
                type |= NORM_RELATIVE;

            OCL_OFF(const double cpuRes = cv::norm(src1_roi, src2_roi, type));
            OCL_ON(const double gpuRes = cv::norm(usrc1_roi, usrc2_roi, type));

            EXPECT_PRED3(relativeError, cpuRes, gpuRes, 1e-6);
        }
}

OCL_TEST_P(Norm, NORM_L2_2args)
{
    for (int relative = 0; relative < 2; ++relative)
        for (int j = 0; j < test_loop_times; j++)
        {
            generateTestData();

            int type = NORM_L2;
            if (relative == 1)
                type |= NORM_RELATIVE;

            OCL_OFF(const double cpuRes = cv::norm(src1_roi, src2_roi, type));
            OCL_ON(const double gpuRes = cv::norm(usrc1_roi, usrc2_roi, type));

            EXPECT_PRED3(relativeError, cpuRes, gpuRes, 1e-6);
        }
}

//////////////////////////////// Sqrt ////////////////////////////////////////////////

typedef ArithmTestBase Sqrt;

OCL_TEST_P(Sqrt, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::sqrt(src1_roi, dst1_roi));
        OCL_ON(cv::sqrt(usrc1_roi, udst1_roi));

        Near(1);
    }
}

//////////////////////////////// Normalize ////////////////////////////////////////////////

typedef ArithmTestBase Normalize;

OCL_TEST_P(Normalize, Mat)
{
    static int modes[] = { CV_MINMAX, CV_L2, CV_L1, CV_C };

    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        for (int i = 0, size = sizeof(modes) / sizeof(modes[0]); i < size; ++i)
        {
            OCL_OFF(cv::normalize(src1_roi, dst1_roi, 10, 110, modes[i], src1_roi.type(), mask_roi));
            OCL_ON(cv::normalize(usrc1_roi, udst1_roi, 10, 110, modes[i], src1_roi.type(), umask_roi));

            Near(1);
        }
    }
}

//////////////////////////////////////// InRange ///////////////////////////////////////////////

PARAM_TEST_CASE(InRange, MatDepth, Channels, bool /*Scalar or not*/, bool /*Roi*/)
{
    int depth;
    int cn;
    bool scalars, use_roi;
    cv::Scalar val1, val2;

    TEST_DECLARE_INPUT_PARAMETER(src1)
    TEST_DECLARE_INPUT_PARAMETER(src2)
    TEST_DECLARE_INPUT_PARAMETER(src3)
    TEST_DECLARE_OUTPUT_PARAMETER(dst)

    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        cn = GET_PARAM(1);
        scalars = GET_PARAM(2);
        use_roi = GET_PARAM(3);
    }

    virtual void generateTestData()
    {
        const int type = CV_MAKE_TYPE(depth, cn);

        Size roiSize = randomSize(1, MAX_VALUE);
        Border src1Border = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src1, src1_roi, roiSize, src1Border, type, -40, 40);

        Border src2Border = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src2, src2_roi, roiSize, src2Border, type, -40, 40);

        Border src3Border = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src3, src3_roi, roiSize, src3Border, type, -40, 40);

        Border dstBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, CV_8UC1, 5, 16);

        val1 = cv::Scalar(rng.uniform(-100.0, 100.0), rng.uniform(-100.0, 100.0),
                          rng.uniform(-100.0, 100.0), rng.uniform(-100.0, 100.0));
        val2 = cv::Scalar(rng.uniform(-100.0, 100.0), rng.uniform(-100.0, 100.0),
                          rng.uniform(-100.0, 100.0), rng.uniform(-100.0, 100.0));

        UMAT_UPLOAD_INPUT_PARAMETER(src1)
        UMAT_UPLOAD_INPUT_PARAMETER(src2)
        UMAT_UPLOAD_INPUT_PARAMETER(src3)
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst)
    }

    void Near()
    {
        OCL_EXPECT_MATS_NEAR(dst, 0)
    }
};

OCL_TEST_P(InRange, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::inRange(src1_roi, src2_roi, src3_roi, dst_roi));
        OCL_ON(cv::inRange(usrc1_roi, usrc2_roi, usrc3_roi, udst_roi));

        Near();
    }
}

OCL_TEST_P(InRange, Scalar)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::inRange(src1_roi, val1, val2, dst_roi));
        OCL_ON(cv::inRange(usrc1_roi, val1, val2, udst_roi));

        Near();
    }
}


//////////////////////////////////////// Instantiation /////////////////////////////////////////

OCL_INSTANTIATE_TEST_CASE_P(Arithm, Lut, Combine(::testing::Values(CV_8U, CV_8S), OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool(), Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Add, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Subtract, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Mul, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Div, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Min, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Max, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Absdiff, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, CartToPolar, Combine(testing::Values(CV_32F, CV_64F), OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, PolarToCart, Combine(testing::Values(CV_32F, CV_64F), OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Transpose, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Bitwise_and, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Bitwise_not, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Bitwise_xor, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Bitwise_or, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Pow, Combine(testing::Values(CV_32F, CV_64F), OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Compare, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, AddWeighted, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, SetIdentity, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Repeat, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, CountNonZero, Combine(OCL_ALL_DEPTHS, testing::Values(Channels(1)), Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Sum, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, MeanStdDev, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Log, Combine(::testing::Values(CV_32F, CV_64F), OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Exp, Combine(::testing::Values(CV_32F, CV_64F), OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Phase, Combine(::testing::Values(CV_32F, CV_64F), OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Magnitude, Combine(::testing::Values(CV_32F, CV_64F), OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Flip, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, MinMaxIdx, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, MinMaxIdx_Mask, Combine(OCL_ALL_DEPTHS, ::testing::Values(Channels(1)), Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Norm, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Sqrt, Combine(::testing::Values(CV_32F, CV_64F), OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, Normalize, Combine(OCL_ALL_DEPTHS, Values(Channels(1)), Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Arithm, InRange, Combine(OCL_ALL_DEPTHS, OCL_ALL_CHANNELS, Bool(), Bool()));

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
