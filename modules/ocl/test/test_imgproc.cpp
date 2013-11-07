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
//    Niko Li, newlife20080214@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Shengen Yan, yanshengen@gmail.com
//    Jiang Liyuan, lyuan001.good@163.com
//    Rock Li, Rock.Li@amd.com
//    Wu Zailong, bullet@yeah.net
//    Xu Pang, pangxu010@163.com
//    Sen Liu, swjtuls1987@126.com
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

#ifdef HAVE_OPENCL

using namespace testing;
using namespace std;
using namespace cv;

///////////////////////////////////////////////////////////////////////////////

PARAM_TEST_CASE(ImgprocTestBase, MatType,
                int, // blockSize
                int, // border type
                bool) // roi or not
{
    int type, borderType, blockSize;
    bool useRoi;

    Mat src, dst_whole, src_roi, dst_roi;
    ocl::oclMat gsrc_whole, gsrc_roi, gdst_whole, gdst_roi;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        blockSize = GET_PARAM(1);
        borderType = GET_PARAM(2);
        useRoi = GET_PARAM(3);
    }

    virtual void random_roi()
    {
        Size roiSize = randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, 5, 256);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst_whole, dst_roi, roiSize, dstBorder, type, 5, 16);

        generateOclMat(gsrc_whole, gsrc_roi, src, roiSize, srcBorder);
        generateOclMat(gdst_whole, gdst_roi, dst_whole, roiSize, dstBorder);
    }

    void Near(double threshold = 0.0, bool relative = false)
    {
        Mat roi, whole;
        gdst_whole.download(whole);
        gdst_roi.download(roi);

        if (relative)
        {
            EXPECT_MAT_NEAR_RELATIVE(dst_whole, whole, threshold);
            EXPECT_MAT_NEAR_RELATIVE(dst_roi, roi, threshold);
        }
        else
        {
            EXPECT_MAT_NEAR(dst_whole, whole, threshold);
            EXPECT_MAT_NEAR(dst_roi, roi, threshold);
        }
    }
};

////////////////////////////////copyMakeBorder////////////////////////////////////////////

PARAM_TEST_CASE(CopyMakeBorder, MatDepth, // depth
                Channels, // channels
                bool, // isolated or not
                Border, // border type
                bool) // roi or not
{
    int type, borderType;
    bool useRoi;

    Border border;
    Scalar val;

    Mat src, dst_whole, src_roi, dst_roi;
    ocl::oclMat gsrc_whole, gsrc_roi, gdst_whole, gdst_roi;

    virtual void SetUp()
    {
        type = CV_MAKE_TYPE(GET_PARAM(0), GET_PARAM(1));
        borderType = GET_PARAM(3);

        if (GET_PARAM(2))
            borderType |= BORDER_ISOLATED;

        useRoi = GET_PARAM(4);
    }

    void random_roi()
    {
        border = randomBorder(0, MAX_VALUE << 2);
        val = randomScalar(-MAX_VALUE, MAX_VALUE);

        Size roiSize = randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, -MAX_VALUE, MAX_VALUE);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        dstBorder.top += border.top;
        dstBorder.lef += border.lef;
        dstBorder.rig += border.rig;
        dstBorder.bot += border.bot;

        randomSubMat(dst_whole, dst_roi, roiSize, dstBorder, type, -MAX_VALUE, MAX_VALUE);

        generateOclMat(gsrc_whole, gsrc_roi, src, roiSize, srcBorder);
        generateOclMat(gdst_whole, gdst_roi, dst_whole, roiSize, dstBorder);
    }

    void Near(double threshold = 0.0)
    {
        Mat whole, roi;
        gdst_whole.download(whole);
        gdst_roi.download(roi);

        EXPECT_MAT_NEAR(dst_whole, whole, threshold);
        EXPECT_MAT_NEAR(dst_roi, roi, threshold);
    }
};

OCL_TEST_P(CopyMakeBorder, Mat)
{
    for (int i = 0; i < LOOP_TIMES; ++i)
    {
        random_roi();

        cv::copyMakeBorder(src_roi, dst_roi, border.top, border.bot, border.lef, border.rig, borderType, val);
        ocl::copyMakeBorder(gsrc_roi, gdst_roi, border.top, border.bot, border.lef, border.rig, borderType, val);

        Near();
    }
}

////////////////////////////////equalizeHist//////////////////////////////////////////////

typedef ImgprocTestBase EqualizeHist;

OCL_TEST_P(EqualizeHist, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        equalizeHist(src_roi, dst_roi);
        ocl::equalizeHist(gsrc_roi, gdst_roi);

        Near(1.1);
    }
}

////////////////////////////////cornerMinEigenVal//////////////////////////////////////////

struct CornerTestBase :
        public ImgprocTestBase
{
    virtual void random_roi()
    {
        Mat image = readImageType("gpu/stereobm/aloe-L.png", type);
        ASSERT_FALSE(image.empty());

        bool isFP = CV_MAT_DEPTH(type) >= CV_32F;
        float val = 255.0f;
        if (isFP)
        {
            image.convertTo(image, -1, 1.0 / 255);
            val /= 255.0f;
        }

        Size roiSize = image.size();
        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);

        Size wholeSize = Size(roiSize.width + srcBorder.lef + srcBorder.rig, roiSize.height + srcBorder.top + srcBorder.bot);
        src = randomMat(wholeSize, type, -val, val, false);
        src_roi = src(Rect(srcBorder.lef, srcBorder.top, roiSize.width, roiSize.height));
        image.copyTo(src_roi);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst_whole, dst_roi, roiSize, dstBorder, CV_32FC1, 5, 16);

        generateOclMat(gsrc_whole, gsrc_roi, src, roiSize, srcBorder);
        generateOclMat(gdst_whole, gdst_roi, dst_whole, roiSize, dstBorder);
    }
};

typedef CornerTestBase CornerMinEigenVal;

OCL_TEST_P(CornerMinEigenVal, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        int apertureSize = 3;

        cornerMinEigenVal(src_roi, dst_roi, blockSize, apertureSize, borderType);
        ocl::cornerMinEigenVal(gsrc_roi, gdst_roi, blockSize, apertureSize, borderType);

        Near(1e-5, true);
    }
}

////////////////////////////////cornerHarris//////////////////////////////////////////
struct CornerHarris :
    public ImgprocTestBase
{
    void Near(double threshold = 0.0)
    {
        Mat whole, roi;
        gdst_whole.download(whole);
        gdst_roi.download(roi);

        absdiff(whole, dst_whole, whole);
        absdiff(roi, dst_roi, roi);

        divide(whole, dst_whole, whole);
        divide(roi, dst_roi, roi);

        absdiff(dst_whole, dst_whole, dst_whole);
        absdiff(dst_roi, dst_roi, dst_roi);

        EXPECT_MAT_NEAR(dst_whole, whole, threshold);
        EXPECT_MAT_NEAR(dst_roi, roi, threshold);
    }
};

OCL_TEST_P(CornerHarris, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        int apertureSize = 3;
        double k = randomDouble(0.01, 0.9);

        cornerHarris(src_roi, dst_roi, blockSize, apertureSize, k, borderType);
        ocl::cornerHarris(gsrc_roi, gdst_roi, blockSize, apertureSize, k, borderType);

        Near(1e-5);
    }
}

//////////////////////////////////integral/////////////////////////////////////////////////

typedef ImgprocTestBase Integral;

OCL_TEST_P(Integral, Mat1)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        ocl::integral(gsrc_roi, gdst_roi);
        integral(src_roi, dst_roi);

        Near();
    }
}

// TODO wrong output type
OCL_TEST_P(Integral, DISABLED_Mat2)
{
    Mat dst1;
    ocl::oclMat gdst1;

    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        integral(src_roi, dst1, dst_roi);
        ocl::integral(gsrc_roi, gdst1, gdst_roi);

        Near();
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//// threshold

struct Threshold :
        public ImgprocTestBase
{
    int thresholdType;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        blockSize = GET_PARAM(1);
        thresholdType = GET_PARAM(2);
        useRoi = GET_PARAM(3);
    }
};

OCL_TEST_P(Threshold, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        double maxVal = randomDouble(20.0, 127.0);
        double thresh = randomDouble(0.0, maxVal);

        threshold(src_roi, dst_roi, thresh, maxVal, thresholdType);
        ocl::threshold(gsrc_roi, gdst_roi, thresh, maxVal, thresholdType);

        Near(1);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////
// calcHist

static void calcHistGold(const Mat &src, Mat &hist)
{
    hist = Mat(1, 256, CV_32SC1, Scalar::all(0));

    int * const hist_row = hist.ptr<int>();
    for (int y = 0; y < src.rows; ++y)
    {
        const uchar * const src_row = src.ptr(y);

        for (int x = 0; x < src.cols; ++x)
            ++hist_row[src_row[x]];
    }
}

typedef ImgprocTestBase CalcHist;

OCL_TEST_P(CalcHist, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        calcHistGold(src_roi, dst_roi);
        ocl::calcHist(gsrc_roi, gdst_roi);

        Near();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//// CLAHE

PARAM_TEST_CASE(CLAHETest, Size, double, bool)
{
    Size gridSize;
    double clipLimit;
    bool useRoi;

    Mat src, dst_whole, src_roi, dst_roi;
    ocl::oclMat gsrc_whole, gsrc_roi, gdst_whole, gdst_roi;

    virtual void SetUp()
    {
        gridSize = GET_PARAM(0);
        clipLimit = GET_PARAM(1);
        useRoi = GET_PARAM(2);
    }

    void random_roi()
    {
        Size roiSize = randomSize(std::max(gridSize.height, gridSize.width), MAX_VALUE);
        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, CV_8UC1, 5, 256);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst_whole, dst_roi, roiSize, dstBorder, CV_8UC1, 5, 16);

        generateOclMat(gsrc_whole, gsrc_roi, src, roiSize, srcBorder);
        generateOclMat(gdst_whole, gdst_roi, dst_whole, roiSize, dstBorder);
    }

    void Near(double threshold = 0.0)
    {
        Mat whole, roi;
        gdst_whole.download(whole);
        gdst_roi.download(roi);

        EXPECT_MAT_NEAR(dst_whole, whole, threshold);
        EXPECT_MAT_NEAR(dst_roi, roi, threshold);
    }
};

OCL_TEST_P(CLAHETest, Accuracy)
{
    for (int i = 0; i < LOOP_TIMES; ++i)
    {
        random_roi();

        Ptr<CLAHE> clahe = ocl::createCLAHE(clipLimit, gridSize);
        clahe->apply(gsrc_roi, gdst_roi);

        Ptr<CLAHE> clahe_gold = createCLAHE(clipLimit, gridSize);
        clahe_gold->apply(src_roi, dst_roi);

        Near(1.0);
    }
}

/////////////////////////////Convolve//////////////////////////////////

static void convolve_gold(const Mat & src, const Mat & kernel, Mat & dst)
{
    for (int i = 0; i < src.rows; i++)
    {
        float * const dstptr = dst.ptr<float>(i);

        for (int j = 0; j < src.cols; j++)
        {
            float temp = 0;

            for (int m = 0; m < kernel.rows; m++)
            {
                const float * const kptr = kernel.ptr<float>(m);
                for (int n = 0; n < kernel.cols; n++)
                {
                    int r = clipInt(i - kernel.rows / 2 + m, 0, src.rows - 1);
                    int c = clipInt(j - kernel.cols / 2 + n, 0, src.cols - 1);

                    temp += src.ptr<float>(r)[c] * kptr[n];
                }
            }

            dstptr[j] = temp;
        }
    }
}

typedef ImgprocTestBase Convolve;

OCL_TEST_P(Convolve, Mat)
{
    Mat kernel, kernel_roi;
    ocl::oclMat gkernel, gkernel_roi;
    const Size roiSize(7, 7);

    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        Border kernelBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(kernel, kernel_roi, roiSize, kernelBorder, type, 5, 16);
        generateOclMat(gkernel, gkernel_roi, kernel, roiSize, kernelBorder);

        convolve_gold(src_roi, kernel_roi, dst_roi);
        ocl::convolve(gsrc_roi, gkernel_roi, gdst_roi);

        Near(1);
    }
}

////////////////////////////////// ColumnSum //////////////////////////////////////

static void columnSum_gold(const Mat & src, Mat & dst)
{
    float * prevdptr = dst.ptr<float>(0);
    const float * sptr = src.ptr<float>(0);

    for (int x = 0; x < src.cols; ++x)
        prevdptr[x] = sptr[x];

    for (int y = 1; y < src.rows; ++y)
    {
        sptr = src.ptr<float>(y);
        float * const dptr = dst.ptr<float>(y);

        for (int x = 0; x < src.cols; ++x)
            dptr[x] = prevdptr[x] + sptr[x];

        prevdptr = dptr;
    }
}

typedef ImgprocTestBase ColumnSum;

OCL_TEST_P(ColumnSum, Accuracy)
{
    for (int i = 0; i < LOOP_TIMES; ++i)
    {
        random_roi();

        columnSum_gold(src_roi, dst_roi);
        ocl::columnSum(gsrc_roi, gdst_roi);

        Near(1e-5);
    }
}

/////////////////////////////////////////////////////////////////////////////////////

INSTANTIATE_TEST_CASE_P(Imgproc, EqualizeHist, Combine(
                            Values((MatType)CV_8UC1),
                            Values(0), // not used
                            Values(0), // not used
                            Bool()));

INSTANTIATE_TEST_CASE_P(Imgproc, CornerMinEigenVal, Combine(
                            Values((MatType)CV_8UC1, (MatType)CV_32FC1),
                            Values(3, 5),
                            Values((int)BORDER_CONSTANT, (int)BORDER_REPLICATE, (int)BORDER_REFLECT, (int)BORDER_REFLECT101),
                            Bool()));

INSTANTIATE_TEST_CASE_P(Imgproc, CornerHarris, Combine(
                            Values((MatType)CV_8UC1, CV_32FC1),
                            Values(3, 5),
                            Values( (int)BORDER_CONSTANT, (int)BORDER_REPLICATE, (int)BORDER_REFLECT, (int)BORDER_REFLECT_101),
                            Bool()));

INSTANTIATE_TEST_CASE_P(Imgproc, Integral, Combine(
                            Values((MatType)CV_8UC1), // TODO does not work with CV_32F, CV_64F
                            Values(0), // not used
                            Values(0), // not used
                            Bool()));

INSTANTIATE_TEST_CASE_P(Imgproc, Threshold, Combine(
                            Values(CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4,
                                   CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
                                   CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4),
                            Values(0),
                            Values(ThreshOp(THRESH_BINARY),
                                   ThreshOp(THRESH_BINARY_INV), ThreshOp(THRESH_TRUNC),
                                   ThreshOp(THRESH_TOZERO), ThreshOp(THRESH_TOZERO_INV)),
                            Bool()));

INSTANTIATE_TEST_CASE_P(Imgproc, CalcHist, Combine(
                            Values((MatType)CV_8UC1),
                            Values(0), // not used
                            Values(0), // not used
                            Bool()));

INSTANTIATE_TEST_CASE_P(Imgproc, CLAHETest, Combine(
                            Values(Size(4, 4), Size(32, 8), Size(8, 64)),
                            Values(0.0, 10.0, 62.0, 300.0),
                            Bool()));

INSTANTIATE_TEST_CASE_P(Imgproc, Convolve, Combine(
                            Values((MatType)CV_32FC1),
                            Values(0), // not used
                            Values(0), // not used
                            Bool()));

INSTANTIATE_TEST_CASE_P(Imgproc, ColumnSum, Combine(
                            Values(MatType(CV_32FC1)),
                            Values(0), // not used
                            Values(0), // not used
                            Bool()));

INSTANTIATE_TEST_CASE_P(ImgprocTestBase, CopyMakeBorder, Combine(
                            testing::Values((MatDepth)CV_8U, (MatDepth)CV_16S, (MatDepth)CV_32S, (MatDepth)CV_32F),
                            testing::Values(Channels(1), Channels(3), (Channels)4),
                            Bool(), // border isolated or not
                            Values((Border)BORDER_REPLICATE, (Border)BORDER_REFLECT,
                                   (Border)BORDER_WRAP, (Border)BORDER_REFLECT_101),
                            Bool()));

#endif // HAVE_OPENCL
