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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Shengen Yan, yanshengen@gmail.com
//    Jiang Liyuan,jlyuan001.good@163.com
//    Rock Li, Rock.Li@amd.com
//    Zailong Wu, bullet@yeah.net
//    Yao Wang, bitwangyaoyao@gmail.com
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
#include <iomanip>

#ifdef HAVE_OPENCL

using namespace cv;
using namespace cv::ocl;
using namespace cvtest;
using namespace testing;
using namespace std;

//////////////////////////////// LUT /////////////////////////////////////////////////

PARAM_TEST_CASE(Lut, MatDepth, MatDepth, bool, bool)
{
    int lut_depth;
    int cn;
    bool use_roi, same_cn;

    // src mat
    cv::Mat src;
    cv::Mat lut;
    cv::Mat dst;

    // src mat with roi
    cv::Mat src_roi;
    cv::Mat lut_roi;
    cv::Mat dst_roi;

    // ocl dst mat for testing
    cv::ocl::oclMat gsrc_whole;
    cv::ocl::oclMat glut_whole;
    cv::ocl::oclMat gdst_whole;

    // ocl mat with roi
    cv::ocl::oclMat gsrc_roi;
    cv::ocl::oclMat glut_roi;
    cv::ocl::oclMat gdst_roi;

    virtual void SetUp()
    {
        lut_depth = GET_PARAM(0);
        cn = GET_PARAM(1);
        same_cn = GET_PARAM(2);
        use_roi = GET_PARAM(3);
    }

    void random_roi()
    {
        const int src_type = CV_MAKE_TYPE(CV_8U, cn);
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

        generateOclMat(gsrc_whole, gsrc_roi, src, roiSize, srcBorder);
        generateOclMat(glut_whole, glut_roi, lut, lutRoiSize, lutBorder);
        generateOclMat(gdst_whole, gdst_roi, dst, roiSize, dstBorder);
    }

    void Near(double threshold = 0.)
    {
        EXPECT_MAT_NEAR(dst, Mat(gdst_whole), threshold);
        EXPECT_MAT_NEAR(dst_roi, Mat(gdst_roi), threshold);
    }
};

OCL_TEST_P(Lut, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::LUT(src_roi, lut_roi, dst_roi);
        cv::ocl::LUT(gsrc_roi, glut_roi, gdst_roi);

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

    // src mat
    cv::Mat src1;
    cv::Mat src2;
    cv::Mat mask;
    cv::Mat dst1;
    cv::Mat dst2;

    // src mat with roi
    cv::Mat src1_roi;
    cv::Mat src2_roi;
    cv::Mat mask_roi;
    cv::Mat dst1_roi;
    cv::Mat dst2_roi;

    // ocl dst mat for testing
    cv::ocl::oclMat gsrc1_whole;
    cv::ocl::oclMat gsrc2_whole;
    cv::ocl::oclMat gdst1_whole;
    cv::ocl::oclMat gdst2_whole;
    cv::ocl::oclMat gmask_whole;

    // ocl mat with roi
    cv::ocl::oclMat gsrc1_roi;
    cv::ocl::oclMat gsrc2_roi;
    cv::ocl::oclMat gdst1_roi;
    cv::ocl::oclMat gdst2_roi;
    cv::ocl::oclMat gmask_roi;

    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        cn = GET_PARAM(1);
        use_roi = GET_PARAM(2);

        val = cv::Scalar(rng.uniform(-100.0, 100.0), rng.uniform(-100.0, 100.0),
                         rng.uniform(-100.0, 100.0), rng.uniform(-100.0, 100.0));
    }

    void random_roi()
    {
        const int type = CV_MAKE_TYPE(depth, cn);

        Size roiSize = randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src1, src1_roi, roiSize, srcBorder, type, 2, 11);

        Border src2Border = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src2, src2_roi, roiSize, src2Border, type, -1540, 1740);

        Border dst1Border = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst1, dst1_roi, roiSize, dst1Border, type, 5, 16);

        Border dst2Border = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst2, dst2_roi, roiSize, dst2Border, type, 5, 16);

        Border maskBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(mask, mask_roi, roiSize, maskBorder, CV_8UC1, 0, 2);
        cv::threshold(mask, mask, 0.5, 255., CV_8UC1);


        generateOclMat(gsrc1_whole, gsrc1_roi, src1, roiSize, srcBorder);
        generateOclMat(gsrc2_whole, gsrc2_roi, src2, roiSize, src2Border);
        generateOclMat(gdst1_whole, gdst1_roi, dst1, roiSize, dst1Border);
        generateOclMat(gdst2_whole, gdst2_roi, dst2, roiSize, dst2Border);
        generateOclMat(gmask_whole, gmask_roi, mask, roiSize, maskBorder);
    }

    void Near(double threshold = 0.)
    {
        EXPECT_MAT_NEAR(dst1, Mat(gdst1_whole), threshold);
        EXPECT_MAT_NEAR(dst1_roi, Mat(gdst1_roi), threshold);
    }

    void Near1(double threshold = 0.)
    {
        EXPECT_MAT_NEAR(dst2, Mat(gdst2_whole), threshold);
        EXPECT_MAT_NEAR(dst2_roi, Mat(gdst2_roi), threshold);
    }
};

//////////////////////////////// Exp /////////////////////////////////////////////////

typedef ArithmTestBase Exp;

OCL_TEST_P(Exp, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::exp(src1_roi, dst1_roi);
        cv::ocl::exp(gsrc1_roi, gdst1_roi);

        Near(2);
    }
}

//////////////////////////////// Log /////////////////////////////////////////////////

typedef ArithmTestBase Log;

OCL_TEST_P(Log, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::log(src1_roi, dst1_roi);
        cv::ocl::log(gsrc1_roi, gdst1_roi);
        Near(1);
    }
}

//////////////////////////////// Add /////////////////////////////////////////////////

typedef ArithmTestBase Add;

OCL_TEST_P(Add, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::add(src1_roi, src2_roi, dst1_roi);
        cv::ocl::add(gsrc1_roi, gsrc2_roi, gdst1_roi);
        Near(0);
    }
}

OCL_TEST_P(Add, Mat_Mask)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::add(src1_roi, src2_roi, dst1_roi, mask_roi);
        cv::ocl::add(gsrc1_roi, gsrc2_roi, gdst1_roi, gmask_roi);
        Near(0);
    }
}

OCL_TEST_P(Add, Scalar)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::add(src1_roi, val, dst1_roi);
        cv::ocl::add(gsrc1_roi, val, gdst1_roi);
        Near(1e-5);
    }
}

OCL_TEST_P(Add, Scalar_Mask)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::add(src1_roi, val, dst1_roi, mask_roi);
        cv::ocl::add(gsrc1_roi, val, gdst1_roi, gmask_roi);
        Near(1e-5);
    }
}

//////////////////////////////// Sub /////////////////////////////////////////////////

typedef ArithmTestBase Sub;

OCL_TEST_P(Sub, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::subtract(src1_roi, src2_roi, dst1_roi);
        cv::ocl::subtract(gsrc1_roi, gsrc2_roi, gdst1_roi);

        Near(0);
    }
}

OCL_TEST_P(Sub, Mat_Mask)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::subtract(src1_roi, src2_roi, dst1_roi, mask_roi);
        cv::ocl::subtract(gsrc1_roi, gsrc2_roi, gdst1_roi, gmask_roi);
        Near(0);
    }
}

OCL_TEST_P(Sub, Scalar)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::subtract(src1_roi, val, dst1_roi);
        cv::ocl::subtract(gsrc1_roi, val, gdst1_roi);

        Near(1e-5);
    }
}

OCL_TEST_P(Sub, Scalar_Mask)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::subtract(src1_roi, val, dst1_roi, mask_roi);
        cv::ocl::subtract(gsrc1_roi, val, gdst1_roi, gmask_roi);
        Near(1e-5);
    }
}

//////////////////////////////// Mul /////////////////////////////////////////////////

typedef ArithmTestBase Mul;

OCL_TEST_P(Mul, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::multiply(src1_roi, src2_roi, dst1_roi);
        cv::ocl::multiply(gsrc1_roi, gsrc2_roi, gdst1_roi);
        Near(0);
    }
}

OCL_TEST_P(Mul, Scalar)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::multiply(val[0], src1_roi, dst1_roi);
        cv::ocl::multiply(val[0], gsrc1_roi, gdst1_roi);

        Near(gdst1_roi.depth() >= CV_32F ? 1e-3 : 1);
    }
}

OCL_TEST_P(Mul, Mat_Scalar)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::multiply(src1_roi, src2_roi, dst1_roi, val[0]);
        cv::ocl::multiply(gsrc1_roi, gsrc2_roi, gdst1_roi, val[0]);

        Near(gdst1_roi.depth() >= CV_32F ? 1e-3 : 1);
    }
}

//////////////////////////////// Div /////////////////////////////////////////////////

typedef ArithmTestBase Div;

OCL_TEST_P(Div, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::divide(src1_roi, src2_roi, dst1_roi);
        cv::ocl::divide(gsrc1_roi, gsrc2_roi, gdst1_roi);
        Near(1);
    }
}

OCL_TEST_P(Div, Scalar)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::divide(val[0], src1_roi, dst1_roi);
        cv::ocl::divide(val[0], gsrc1_roi, gdst1_roi);

        Near(gdst1_roi.depth() >= CV_32F ? 1e-3 : 1);
    }
}

OCL_TEST_P(Div, Mat_Scalar)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::divide(src1_roi, src2_roi, dst1_roi, val[0]);
        cv::ocl::divide(gsrc1_roi, gsrc2_roi, gdst1_roi, val[0]);

        Near(gdst1_roi.depth() >= CV_32F ? 4e-3 : 1);
    }
}

//////////////////////////////// Absdiff /////////////////////////////////////////////////

typedef ArithmTestBase Min;

OCL_TEST_P(Min, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        dst1_roi = cv::min(src1_roi, src2_roi);
        cv::ocl::min(gsrc1_roi, gsrc2_roi, gdst1_roi);
        Near(0);
    }
}

typedef ArithmTestBase Max;

OCL_TEST_P(Max, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        dst1_roi = cv::min(src1_roi, src2_roi);
        cv::ocl::min(gsrc1_roi, gsrc2_roi, gdst1_roi);
        Near(0);
    }
}

//////////////////////////////// Abs /////////////////////////////////////////////////////

typedef ArithmTestBase Abs;

OCL_TEST_P(Abs, Abs)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        dst1_roi = cv::abs(src1_roi);
        cv::ocl::abs(gsrc1_roi, gdst1_roi);
        Near(0);
    }
}

//////////////////////////////// Absdiff /////////////////////////////////////////////////

typedef ArithmTestBase Absdiff;

OCL_TEST_P(Absdiff, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::absdiff(src1_roi, src2_roi, dst1_roi);
        cv::ocl::absdiff(gsrc1_roi, gsrc2_roi, gdst1_roi);
        Near(0);
    }
}

OCL_TEST_P(Absdiff, Scalar)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::absdiff(src1_roi, val, dst1_roi);
        cv::ocl::absdiff(gsrc1_roi, val, gdst1_roi);
        Near(1e-5);
    }
}

//////////////////////////////// CartToPolar /////////////////////////////////////////////////

typedef ArithmTestBase CartToPolar;

OCL_TEST_P(CartToPolar, angleInDegree)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::cartToPolar(src1_roi, src2_roi, dst1_roi, dst2_roi, true);
        cv::ocl::cartToPolar(gsrc1_roi, gsrc2_roi, gdst1_roi, gdst2_roi, true);
        Near(.5);
        Near1(.5);
    }
}

OCL_TEST_P(CartToPolar, angleInRadians)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::cartToPolar(src1_roi, src2_roi, dst1_roi, dst2_roi);
        cv::ocl::cartToPolar(gsrc1_roi, gsrc2_roi, gdst1_roi, gdst2_roi);
        Near(.5);
        Near1(.5);
    }
}

//////////////////////////////// PolarToCart /////////////////////////////////////////////////

typedef ArithmTestBase PolarToCart;

OCL_TEST_P(PolarToCart, angleInDegree)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::polarToCart(src1_roi, src2_roi, dst1_roi, dst2_roi, true);
        cv::ocl::polarToCart(gsrc1_roi, gsrc2_roi, gdst1_roi, gdst2_roi, true);

        Near(.5);
        Near1(.5);
    }
}

OCL_TEST_P(PolarToCart, angleInRadians)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::polarToCart(src1_roi, src2_roi, dst1_roi, dst2_roi);
        cv::ocl::polarToCart(gsrc1_roi, gsrc2_roi, gdst1_roi, gdst2_roi);

        Near(.5);
        Near1(.5);
    }
}

//////////////////////////////// Magnitude /////////////////////////////////////////////////

typedef ArithmTestBase Magnitude;

OCL_TEST_P(Magnitude, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::magnitude(src1_roi, src2_roi, dst1_roi);
        cv::ocl::magnitude(gsrc1_roi, gsrc2_roi, gdst1_roi);
        Near(depth == CV_64F ? 1e-5 : 1e-2);
    }
}

//////////////////////////////// Transpose /////////////////////////////////////////////////

typedef ArithmTestBase Transpose;

OCL_TEST_P(Transpose, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::transpose(src1_roi, dst1_roi);
        cv::ocl::transpose(gsrc1_roi, gdst1_roi);

        Near(1e-5);
    }
}

OCL_TEST_P(Transpose, SquareInplace)
{
    const int type = CV_MAKE_TYPE(depth, cn);

    for (int j = 0; j < LOOP_TIMES; j++)
    {
        Size roiSize = randomSize(1, MAX_VALUE);
        roiSize.height = roiSize.width; // make it square

        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src1, src1_roi, roiSize, srcBorder, type, 5, 16);

        generateOclMat(gsrc1_whole, gsrc1_roi, src1, roiSize, srcBorder);

        cv::transpose(src1_roi, src1_roi);
        cv::ocl::transpose(gsrc1_roi, gsrc1_roi);

        EXPECT_MAT_NEAR(src1, Mat(gsrc1_whole), 0.0);
        EXPECT_MAT_NEAR(src1_roi, Mat(gsrc1_roi), 0.0);
    }
}

//////////////////////////////// Flip /////////////////////////////////////////////////

typedef ArithmTestBase Flip;

OCL_TEST_P(Flip, X)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::flip(src1_roi, dst1_roi, 0);
        cv::ocl::flip(gsrc1_roi, gdst1_roi, 0);
        Near(1e-5);
    }
}

OCL_TEST_P(Flip, Y)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::flip(src1_roi, dst1_roi, 1);
        cv::ocl::flip(gsrc1_roi, gdst1_roi, 1);
        Near(1e-5);
    }
}

OCL_TEST_P(Flip, BOTH)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::flip(src1_roi, dst1_roi, -1);
        cv::ocl::flip(gsrc1_roi, gdst1_roi, -1);
        Near(1e-5);
    }
}

//////////////////////////////// MinMax /////////////////////////////////////////////////

typedef ArithmTestBase MinMax;

OCL_TEST_P(MinMax, MAT)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        double minVal, maxVal;

        if (src1.depth() != CV_8S)
            cv::minMaxIdx(src1_roi, &minVal, &maxVal, NULL, NULL);
        else
        {
            minVal = std::numeric_limits<double>::max();
            maxVal = -std::numeric_limits<double>::max();
            for (int i = 0; i < src1_roi.rows; ++i)
                for (int j = 0; j < src1_roi.cols; ++j)
                {
                    signed char val = src1_roi.at<signed char>(i, j);
                    if (val < minVal) minVal = val;
                    if (val > maxVal) maxVal = val;
                }
        }

        double minVal_, maxVal_;
        cv::ocl::minMax(gsrc1_roi, &minVal_, &maxVal_);

        EXPECT_DOUBLE_EQ(minVal_, minVal);
        EXPECT_DOUBLE_EQ(maxVal_, maxVal);
    }
}

OCL_TEST_P(MinMax, MASK)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;

        if (src1.depth() != CV_8S)
            cv::minMaxLoc(src1_roi, &minVal, &maxVal, &minLoc, &maxLoc, mask_roi);
        else
        {
            minVal = std::numeric_limits<double>::max();
            maxVal = -std::numeric_limits<double>::max();
            for (int i = 0; i < src1_roi.rows; ++i)
                for (int j = 0; j < src1_roi.cols; ++j)
                {
                    signed char val = src1_roi.at<signed char>(i, j);
                    unsigned char m = mask_roi.at<unsigned char>(i, j);
                    if (val < minVal && m) minVal = val;
                    if (val > maxVal && m) maxVal = val;
                }
        }

        double minVal_, maxVal_;
        cv::ocl::minMax(gsrc1_roi, &minVal_, &maxVal_, gmask_roi);

        EXPECT_DOUBLE_EQ(minVal, minVal_);
        EXPECT_DOUBLE_EQ(maxVal, maxVal_);
    }
}

//////////////////////////////// MinMaxLoc /////////////////////////////////////////////////

typedef ArithmTestBase MinMaxLoc;

OCL_TEST_P(MinMaxLoc, MAT)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        int depth = src1.depth();

        if (depth != CV_8S)
            cv::minMaxLoc(src1_roi, &minVal, &maxVal, &minLoc, &maxLoc);
        else
        {
            minVal = std::numeric_limits<double>::max();
            maxVal = -std::numeric_limits<double>::max();
            for (int i = 0; i < src1_roi.rows; ++i)
                for (int j = 0; j < src1_roi.cols; ++j)
                {
                    signed char val = src1_roi.at<signed char>(i, j);
                    if (val < minVal)
                    {
                        minVal = val;
                        minLoc.x = j;
                        minLoc.y = i;
                    }
                    if (val > maxVal)
                    {
                        maxVal = val;
                        maxLoc.x = j;
                        maxLoc.y = i;
                    }
                }
        }

        double minVal_, maxVal_;
        cv::Point minLoc_, maxLoc_;
        cv::ocl::minMaxLoc(gsrc1_roi, &minVal_, &maxVal_, &minLoc_, &maxLoc_, cv::ocl::oclMat());

        double error0 = 0., error1 = 0., minlocVal = 0., minlocVal_ = 0., maxlocVal = 0., maxlocVal_ = 0.;
        if (depth == 0)
        {
            minlocVal = src1_roi.at<unsigned char>(minLoc);
            minlocVal_ = src1_roi.at<unsigned char>(minLoc_);
            maxlocVal = src1_roi.at<unsigned char>(maxLoc);
            maxlocVal_ = src1_roi.at<unsigned char>(maxLoc_);
            error0 = ::abs(src1_roi.at<unsigned char>(minLoc_) - src1_roi.at<unsigned char>(minLoc));
            error1 = ::abs(src1_roi.at<unsigned char>(maxLoc_) - src1_roi.at<unsigned char>(maxLoc));
        }
        if (depth == 1)
        {
            minlocVal = src1_roi.at<signed char>(minLoc);
            minlocVal_ = src1_roi.at<signed char>(minLoc_);
            maxlocVal = src1_roi.at<signed char>(maxLoc);
            maxlocVal_ = src1_roi.at<signed char>(maxLoc_);
            error0 = ::abs(src1_roi.at<signed char>(minLoc_) - src1_roi.at<signed char>(minLoc));
            error1 = ::abs(src1_roi.at<signed char>(maxLoc_) - src1_roi.at<signed char>(maxLoc));
        }
        if (depth == 2)
        {
            minlocVal = src1_roi.at<unsigned short>(minLoc);
            minlocVal_ = src1_roi.at<unsigned short>(minLoc_);
            maxlocVal = src1_roi.at<unsigned short>(maxLoc);
            maxlocVal_ = src1_roi.at<unsigned short>(maxLoc_);
            error0 = ::abs(src1_roi.at<unsigned short>(minLoc_) - src1_roi.at<unsigned short>(minLoc));
            error1 = ::abs(src1_roi.at<unsigned short>(maxLoc_) - src1_roi.at<unsigned short>(maxLoc));
        }
        if (depth == 3)
        {
            minlocVal = src1_roi.at<signed short>(minLoc);
            minlocVal_ = src1_roi.at<signed short>(minLoc_);
            maxlocVal = src1_roi.at<signed short>(maxLoc);
            maxlocVal_ = src1_roi.at<signed short>(maxLoc_);
            error0 = ::abs(src1_roi.at<signed short>(minLoc_) - src1_roi.at<signed short>(minLoc));
            error1 = ::abs(src1_roi.at<signed short>(maxLoc_) - src1_roi.at<signed short>(maxLoc));
        }
        if (depth == 4)
        {
            minlocVal = src1_roi.at<int>(minLoc);
            minlocVal_ = src1_roi.at<int>(minLoc_);
            maxlocVal = src1_roi.at<int>(maxLoc);
            maxlocVal_ = src1_roi.at<int>(maxLoc_);
            error0 = ::abs(src1_roi.at<int>(minLoc_) - src1_roi.at<int>(minLoc));
            error1 = ::abs(src1_roi.at<int>(maxLoc_) - src1_roi.at<int>(maxLoc));
        }
        if (depth == 5)
        {
            minlocVal = src1_roi.at<float>(minLoc);
            minlocVal_ = src1_roi.at<float>(minLoc_);
            maxlocVal = src1_roi.at<float>(maxLoc);
            maxlocVal_ = src1_roi.at<float>(maxLoc_);
            error0 = ::abs(src1_roi.at<float>(minLoc_) - src1_roi.at<float>(minLoc));
            error1 = ::abs(src1_roi.at<float>(maxLoc_) - src1_roi.at<float>(maxLoc));
        }
        if (depth == 6)
        {
            minlocVal = src1_roi.at<double>(minLoc);
            minlocVal_ = src1_roi.at<double>(minLoc_);
            maxlocVal = src1_roi.at<double>(maxLoc);
            maxlocVal_ = src1_roi.at<double>(maxLoc_);
            error0 = ::abs(src1_roi.at<double>(minLoc_) - src1_roi.at<double>(minLoc));
            error1 = ::abs(src1_roi.at<double>(maxLoc_) - src1_roi.at<double>(maxLoc));
        }

        EXPECT_DOUBLE_EQ(minVal_, minVal);
        EXPECT_DOUBLE_EQ(maxVal_, maxVal);
        EXPECT_DOUBLE_EQ(minlocVal_, minlocVal);
        EXPECT_DOUBLE_EQ(maxlocVal_, maxlocVal);

        EXPECT_DOUBLE_EQ(error0, 0.0);
        EXPECT_DOUBLE_EQ(error1, 0.0);
    }
}

OCL_TEST_P(MinMaxLoc, MASK)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        int depth = src1.depth();
        if (depth != CV_8S)
            cv::minMaxLoc(src1_roi, &minVal, &maxVal, &minLoc, &maxLoc, mask_roi);
        else
        {
            minVal = std::numeric_limits<double>::max();
            maxVal = -std::numeric_limits<double>::max();
            for (int i = 0; i < src1_roi.rows; ++i)
                for (int j = 0; j < src1_roi.cols; ++j)
                {
                    signed char val = src1_roi.at<signed char>(i, j);
                    unsigned char m = mask_roi.at<unsigned char>(i , j);
                    if (val < minVal && m)
                    {
                        minVal = val;
                        minLoc.x = j;
                        minLoc.y = i;
                    }
                    if (val > maxVal && m)
                    {
                        maxVal = val;
                        maxLoc.x = j;
                        maxLoc.y = i;
                    }
                }
        }

        double minVal_, maxVal_;
        cv::Point minLoc_, maxLoc_;
        cv::ocl::minMaxLoc(gsrc1_roi, &minVal_, &maxVal_, &minLoc_, &maxLoc_, gmask_roi);

        double error0 = 0., error1 = 0., minlocVal = 0., minlocVal_ = 0., maxlocVal = 0., maxlocVal_ = 0.;
        if (minLoc_.x == -1 || minLoc_.y == -1 || maxLoc_.x == -1 || maxLoc_.y == -1) continue;
        if (depth == 0)
        {
            minlocVal = src1_roi.at<unsigned char>(minLoc);
            minlocVal_ = src1_roi.at<unsigned char>(minLoc_);
            maxlocVal = src1_roi.at<unsigned char>(maxLoc);
            maxlocVal_ = src1_roi.at<unsigned char>(maxLoc_);
            error0 = ::abs(src1_roi.at<unsigned char>(minLoc_) - src1_roi.at<unsigned char>(minLoc));
            error1 = ::abs(src1_roi.at<unsigned char>(maxLoc_) - src1_roi.at<unsigned char>(maxLoc));
        }
        if (depth == 1)
        {
            minlocVal = src1_roi.at<signed char>(minLoc);
            minlocVal_ = src1_roi.at<signed char>(minLoc_);
            maxlocVal = src1_roi.at<signed char>(maxLoc);
            maxlocVal_ = src1_roi.at<signed char>(maxLoc_);
            error0 = ::abs(src1_roi.at<signed char>(minLoc_) - src1_roi.at<signed char>(minLoc));
            error1 = ::abs(src1_roi.at<signed char>(maxLoc_) - src1_roi.at<signed char>(maxLoc));
        }
        if (depth == 2)
        {
            minlocVal = src1_roi.at<unsigned short>(minLoc);
            minlocVal_ = src1_roi.at<unsigned short>(minLoc_);
            maxlocVal = src1_roi.at<unsigned short>(maxLoc);
            maxlocVal_ = src1_roi.at<unsigned short>(maxLoc_);
            error0 = ::abs(src1_roi.at<unsigned short>(minLoc_) - src1_roi.at<unsigned short>(minLoc));
            error1 = ::abs(src1_roi.at<unsigned short>(maxLoc_) - src1_roi.at<unsigned short>(maxLoc));
        }
        if (depth == 3)
        {
            minlocVal = src1_roi.at<signed short>(minLoc);
            minlocVal_ = src1_roi.at<signed short>(minLoc_);
            maxlocVal = src1_roi.at<signed short>(maxLoc);
            maxlocVal_ = src1_roi.at<signed short>(maxLoc_);
            error0 = ::abs(src1_roi.at<signed short>(minLoc_) - src1_roi.at<signed short>(minLoc));
            error1 = ::abs(src1_roi.at<signed short>(maxLoc_) - src1_roi.at<signed short>(maxLoc));
        }
        if (depth == 4)
        {
            minlocVal = src1_roi.at<int>(minLoc);
            minlocVal_ = src1_roi.at<int>(minLoc_);
            maxlocVal = src1_roi.at<int>(maxLoc);
            maxlocVal_ = src1_roi.at<int>(maxLoc_);
            error0 = ::abs(src1_roi.at<int>(minLoc_) - src1_roi.at<int>(minLoc));
            error1 = ::abs(src1_roi.at<int>(maxLoc_) - src1_roi.at<int>(maxLoc));
        }
        if (depth == 5)
        {
            minlocVal = src1_roi.at<float>(minLoc);
            minlocVal_ = src1_roi.at<float>(minLoc_);
            maxlocVal = src1_roi.at<float>(maxLoc);
            maxlocVal_ = src1_roi.at<float>(maxLoc_);
            error0 = ::abs(src1_roi.at<float>(minLoc_) - src1_roi.at<float>(minLoc));
            error1 = ::abs(src1_roi.at<float>(maxLoc_) - src1_roi.at<float>(maxLoc));
        }
        if (depth == 6)
        {
            minlocVal = src1_roi.at<double>(minLoc);
            minlocVal_ = src1_roi.at<double>(minLoc_);
            maxlocVal = src1_roi.at<double>(maxLoc);
            maxlocVal_ = src1_roi.at<double>(maxLoc_);
            error0 = ::abs(src1_roi.at<double>(minLoc_) - src1_roi.at<double>(minLoc));
            error1 = ::abs(src1_roi.at<double>(maxLoc_) - src1_roi.at<double>(maxLoc));
        }

        EXPECT_DOUBLE_EQ(minVal_, minVal);
        EXPECT_DOUBLE_EQ(maxVal_, maxVal);
        EXPECT_DOUBLE_EQ(minlocVal_, minlocVal);
        EXPECT_DOUBLE_EQ(maxlocVal_, maxlocVal);

        EXPECT_DOUBLE_EQ(error0, 0.0);
        EXPECT_DOUBLE_EQ(error1, 0.0);
    }
}

//////////////////////////////// Sum /////////////////////////////////////////////////

typedef ArithmTestBase Sum;

OCL_TEST_P(Sum, MAT)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        Scalar cpures = cv::sum(src1_roi);
        Scalar gpures = cv::ocl::sum(gsrc1_roi);

        // check results
        EXPECT_NEAR(cpures[0], gpures[0], 0.1);
        EXPECT_NEAR(cpures[1], gpures[1], 0.1);
        EXPECT_NEAR(cpures[2], gpures[2], 0.1);
        EXPECT_NEAR(cpures[3], gpures[3], 0.1);
    }
}

typedef ArithmTestBase SqrSum;

template <typename T, typename WT>
static Scalar sqrSum(const Mat & src)
{
    Scalar sum = Scalar::all(0);
    int cn = src.channels();
    WT data[4] = { 0, 0, 0, 0 };

    int cols = src.cols * cn;
    for (int y = 0; y < src.rows; ++y)
    {
        const T * const sdata = src.ptr<T>(y);
        for (int x = 0; x < cols; )
            for (int i = 0; i < cn; ++i, ++x)
            {
                WT t = static_cast<WT>(sdata[x]);
                data[i] += t * t;
            }
    }

    for (int i = 0; i < cn; ++i)
        sum[i] = static_cast<double>(data[i]);

    return sum;
}

typedef Scalar (*sumFunc)(const Mat &);

OCL_TEST_P(SqrSum, MAT)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        static sumFunc funcs[] = { sqrSum<uchar, int>,
                                 sqrSum<char, int>,
                                 sqrSum<ushort, int>,
                                 sqrSum<short, int>,
                                 sqrSum<int, int>,
                                 sqrSum<float, double>,
                                 sqrSum<double, double>,
                                 0 };

        sumFunc func = funcs[src1_roi.depth()];
        CV_Assert(func != 0);

        Scalar cpures = func(src1_roi);
        Scalar gpures = cv::ocl::sqrSum(gsrc1_roi);

        // check results
        EXPECT_NEAR(cpures[0], gpures[0], 1.0);
        EXPECT_NEAR(cpures[1], gpures[1], 1.0);
        EXPECT_NEAR(cpures[2], gpures[2], 1.0);
        EXPECT_NEAR(cpures[3], gpures[3], 1.0);
    }
}

typedef ArithmTestBase AbsSum;

template <typename T, typename WT>
static Scalar absSum(const Mat & src)
{
    Scalar sum = Scalar::all(0);
    int cn = src.channels();
    WT data[4] = { 0, 0, 0, 0 };

    int cols = src.cols * cn;
    for (int y = 0; y < src.rows; ++y)
    {
        const T * const sdata = src.ptr<T>(y);
        for (int x = 0; x < cols; )
            for (int i = 0; i < cn; ++i, ++x)
            {
                WT t = static_cast<WT>(sdata[x]);
                data[i] += t >= 0 ? t : -t;
            }
    }

    for (int i = 0; i < cn; ++i)
        sum[i] = static_cast<double>(data[i]);

    return sum;
}

OCL_TEST_P(AbsSum, MAT)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        static sumFunc funcs[] = { absSum<uchar, int>,
                                 absSum<char, int>,
                                 absSum<ushort, int>,
                                 absSum<short, int>,
                                 absSum<int, int>,
                                 absSum<float, double>,
                                 absSum<double, double>,
                                 0 };

        sumFunc func = funcs[src1_roi.depth()];
        CV_Assert(func != 0);

        Scalar cpures = func(src1_roi);
        Scalar gpures = cv::ocl::absSum(gsrc1_roi);

        // check results
        EXPECT_NEAR(cpures[0], gpures[0], 0.1);
        EXPECT_NEAR(cpures[1], gpures[1], 0.1);
        EXPECT_NEAR(cpures[2], gpures[2], 0.1);
        EXPECT_NEAR(cpures[3], gpures[3], 0.1);
    }
}

//////////////////////////////// CountNonZero /////////////////////////////////////////////////

typedef ArithmTestBase CountNonZero;

OCL_TEST_P(CountNonZero, MAT)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();
        int cpures = cv::countNonZero(src1_roi);
        int gpures = cv::ocl::countNonZero(gsrc1_roi);

        EXPECT_DOUBLE_EQ((double)cpures, (double)gpures);
    }
}

//////////////////////////////// Phase /////////////////////////////////////////////////

typedef ArithmTestBase Phase;

OCL_TEST_P(Phase, angleInDegrees)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();
        cv::phase(src1_roi, src2_roi, dst1_roi, true);
        cv::ocl::phase(gsrc1_roi, gsrc2_roi, gdst1_roi, true);

        Near(1e-2);
    }
}

OCL_TEST_P(Phase, angleInRadians)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();
        cv::phase(src1_roi, src2_roi, dst1_roi);
        cv::ocl::phase(gsrc1_roi, gsrc2_roi, gdst1_roi);

        Near(1e-2);
    }
}

//////////////////////////////// Bitwise_and /////////////////////////////////////////////////

typedef ArithmTestBase Bitwise_and;

OCL_TEST_P(Bitwise_and, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::bitwise_and(src1_roi, src2_roi, dst1_roi);
        cv::ocl::bitwise_and(gsrc1_roi, gsrc2_roi, gdst1_roi);
        Near(0);
    }
}

OCL_TEST_P(Bitwise_and, Mat_Mask)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::bitwise_and(src1_roi, src2_roi, dst1_roi, mask_roi);
        cv::ocl::bitwise_and(gsrc1_roi, gsrc2_roi, gdst1_roi, gmask_roi);
        Near(0);
    }
}

OCL_TEST_P(Bitwise_and, Scalar)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::bitwise_and(src1_roi, val, dst1_roi);
        cv::ocl::bitwise_and(gsrc1_roi, val, gdst1_roi);
        Near(1e-5);
    }
}

OCL_TEST_P(Bitwise_and, Scalar_Mask)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::bitwise_and(src1_roi, val, dst1_roi, mask_roi);
        cv::ocl::bitwise_and(gsrc1_roi, val, gdst1_roi, gmask_roi);
        Near(1e-5);
    }
}

//////////////////////////////// Bitwise_or /////////////////////////////////////////////////

typedef ArithmTestBase Bitwise_or;

OCL_TEST_P(Bitwise_or, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::bitwise_or(src1_roi, src2_roi, dst1_roi);
        cv::ocl::bitwise_or(gsrc1_roi, gsrc2_roi, gdst1_roi);
        Near(0);
    }
}

OCL_TEST_P(Bitwise_or, Mat_Mask)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::bitwise_or(src1_roi, src2_roi, dst1_roi, mask_roi);
        cv::ocl::bitwise_or(gsrc1_roi, gsrc2_roi, gdst1_roi, gmask_roi);
        Near(0);
    }
}

OCL_TEST_P(Bitwise_or, Scalar)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::bitwise_or(src1_roi, val, dst1_roi);
        cv::ocl::bitwise_or(gsrc1_roi, val, gdst1_roi);
        Near(1e-5);
    }
}

OCL_TEST_P(Bitwise_or, Scalar_Mask)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::bitwise_or(src1_roi, val, dst1_roi, mask_roi);
        cv::ocl::bitwise_or(gsrc1_roi, val, gdst1_roi, gmask_roi);
        Near(1e-5);
    }
}

//////////////////////////////// Bitwise_xor /////////////////////////////////////////////////

typedef ArithmTestBase Bitwise_xor;

OCL_TEST_P(Bitwise_xor, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::bitwise_xor(src1_roi, src2_roi, dst1_roi);
        cv::ocl::bitwise_xor(gsrc1_roi, gsrc2_roi, gdst1_roi);
        Near(0);
    }
}

OCL_TEST_P(Bitwise_xor, Mat_Mask)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::bitwise_xor(src1_roi, src2_roi, dst1_roi, mask_roi);
        cv::ocl::bitwise_xor(gsrc1_roi, gsrc2_roi, gdst1_roi, gmask_roi);
        Near(0);
    }
}

OCL_TEST_P(Bitwise_xor, Scalar)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::bitwise_xor(src1_roi, val, dst1_roi);
        cv::ocl::bitwise_xor(gsrc1_roi, val, gdst1_roi);
        Near(1e-5);
    }
}

OCL_TEST_P(Bitwise_xor, Scalar_Mask)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::bitwise_xor(src1_roi, val, dst1_roi, mask_roi);
        cv::ocl::bitwise_xor(gsrc1_roi, val, gdst1_roi, gmask_roi);
        Near(1e-5);
    }
}

//////////////////////////////// Bitwise_not /////////////////////////////////////////////////

typedef ArithmTestBase Bitwise_not;

OCL_TEST_P(Bitwise_not, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::bitwise_not(src1_roi, dst1_roi);
        cv::ocl::bitwise_not(gsrc1_roi, gdst1_roi);
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
        for (int j = 0; j < LOOP_TIMES; j++)
        {
            random_roi();

            cv::compare(src1_roi, src2_roi, dst1_roi, cmp_codes[i]);
            cv::ocl::compare(gsrc1_roi, gsrc2_roi, gdst1_roi, cmp_codes[i]);

            Near(0);
        }
}

//////////////////////////////// Pow /////////////////////////////////////////////////

typedef ArithmTestBase Pow;

OCL_TEST_P(Pow, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();
        double p = 4.5;
        cv::pow(src1_roi, p, dst1_roi);
        cv::ocl::pow(gsrc1_roi, p, gdst1_roi);
        Near(1);
    }
}

//////////////////////////////// AddWeighted /////////////////////////////////////////////////

typedef ArithmTestBase AddWeighted;

OCL_TEST_P(AddWeighted, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        const double alpha = 2.0, beta = 1.0, gama = 3.0;

        cv::addWeighted(src1_roi, alpha, src2_roi, beta, gama, dst1_roi);
        cv::ocl::addWeighted(gsrc1_roi, alpha, gsrc2_roi, beta, gama, gdst1_roi);

        Near(3e-4);
    }
}

//////////////////////////////// setIdentity /////////////////////////////////////////////////

typedef ArithmTestBase SetIdentity;

OCL_TEST_P(SetIdentity, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::setIdentity(dst1_roi, val);
        cv::ocl::setIdentity(gdst1_roi, val);

        Near(0);
    }
}

//////////////////////////////// meanStdDev /////////////////////////////////////////////////

typedef ArithmTestBase MeanStdDev;

OCL_TEST_P(MeanStdDev, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        Scalar cpu_mean, cpu_stddev;
        Scalar gpu_mean, gpu_stddev;

        cv::meanStdDev(src1_roi, cpu_mean, cpu_stddev);
        cv::ocl::meanStdDev(gsrc1_roi, gpu_mean, gpu_stddev);

        for (int i = 0; i < 4; ++i)
        {
            EXPECT_NEAR(cpu_mean[i], gpu_mean[i], 0.1);
            EXPECT_NEAR(cpu_stddev[i], gpu_stddev[i], 0.1);
        }
    }
}

//////////////////////////////// Norm /////////////////////////////////////////////////

typedef ArithmTestBase Norm;

OCL_TEST_P(Norm, NORM_INF)
{
    for (int relative = 0; relative < 2; ++relative)
        for (int j = 0; j < LOOP_TIMES; j++)
        {
            random_roi();

            int type = NORM_INF;
            if (relative == 1)
                type |= NORM_RELATIVE;

            const double cpuRes = cv::norm(src1_roi, src2_roi, type);
            const double gpuRes = cv::ocl::norm(gsrc1_roi, gsrc2_roi, type);

            EXPECT_NEAR(cpuRes, gpuRes, 0.1);
        }
}

OCL_TEST_P(Norm, NORM_L1)
{
    for (int relative = 0; relative < 2; ++relative)
        for (int j = 0; j < LOOP_TIMES; j++)
        {
            random_roi();

            int type = NORM_L1;
            if (relative == 1)
                type |= NORM_RELATIVE;

            const double cpuRes = cv::norm(src1_roi, src2_roi, type);
            const double gpuRes = cv::ocl::norm(gsrc1_roi, gsrc2_roi, type);

            EXPECT_NEAR(cpuRes, gpuRes, 0.1);
        }
}

OCL_TEST_P(Norm, NORM_L2)
{
    for (int relative = 0; relative < 2; ++relative)
        for (int j = 0; j < LOOP_TIMES; j++)
        {
            random_roi();

            int type = NORM_L2;
            if (relative == 1)
                type |= NORM_RELATIVE;

            const double cpuRes = cv::norm(src1_roi, src2_roi, type);
            const double gpuRes = cv::ocl::norm(gsrc1_roi, gsrc2_roi, type);

            EXPECT_NEAR(cpuRes, gpuRes, 0.1);
        }
}

//////////////////////////////////////// Instantiation /////////////////////////////////////////

INSTANTIATE_TEST_CASE_P(Arithm, Lut, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool(), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Exp, Combine(testing::Values(CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Log, Combine(testing::Values(CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Add, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Sub, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Mul, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Div, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Min, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Max, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Abs, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Absdiff, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, CartToPolar, Combine(Values(CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, PolarToCart, Combine(Values(CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Magnitude, Combine(Values(CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Transpose, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Flip, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, MinMax, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(Channels(1)), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, MinMaxLoc, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(Channels(1)), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Sum, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, SqrSum, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, AbsSum, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, CountNonZero, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(Channels(1)), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Phase, Combine(Values(CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Bitwise_and, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Bitwise_or, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Bitwise_xor, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Bitwise_not, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Compare, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(Channels(1)), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Pow, Combine(Values(CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, AddWeighted, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, SetIdentity, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, MeanStdDev, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));
INSTANTIATE_TEST_CASE_P(Arithm, Norm, Combine(Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F), Values(1, 2, 3, 4), Bool()));

#endif // HAVE_OPENCL
