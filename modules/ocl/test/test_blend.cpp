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
//    Nathan, liujun@multicorewareinc.com
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
#include "test_precomp.hpp"
#include <iomanip>

using namespace cv;
using namespace cv::ocl;
using namespace testing;
using namespace std;

template <typename T>
static void blendLinearGold(const Mat &img1, const Mat &img2,
                            const Mat &weights1, const Mat &weights2,
                            Mat &result_gold)
{
    CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
    CV_Assert(weights1.size() == weights2.size() && weights1.size() == img1.size() &&
              weights1.type() == CV_32FC1 && weights2.type() == CV_32FC1);

    result_gold.create(img1.size(), img1.type());

    int cn = img1.channels();
    int step1 = img1.cols * img1.channels();

    for (int y = 0; y < img1.rows; ++y)
    {
        const float * const weights1_row = weights1.ptr<float>(y);
        const float * const weights2_row = weights2.ptr<float>(y);
        const T * const img1_row = img1.ptr<T>(y);
        const T * const img2_row = img2.ptr<T>(y);
        T * const result_gold_row = result_gold.ptr<T>(y);

        for (int x = 0; x < step1; ++x)
        {
            int x1 = x / cn;
            float w1 = weights1_row[x1], w2 = weights2_row[x1];
            result_gold_row[x] = saturate_cast<T>(((float)img1_row[x] * w1
                                                 + (float)img2_row[x] * w2) / (w1 + w2 + 1e-5f));
        }
    }
}

PARAM_TEST_CASE(Blend, MatDepth, int, bool)
{
    int depth, channels;
    bool useRoi;

    Mat src1, src2, weights1, weights2, dst;
    Mat src1_roi, src2_roi, weights1_roi, weights2_roi, dst_roi;
    oclMat gsrc1, gsrc2, gweights1, gweights2, gdst, gst;
    oclMat gsrc1_roi, gsrc2_roi, gweights1_roi, gweights2_roi, gdst_roi;

    virtual void SetUp()
    {
        depth = GET_PARAM(0);
        channels = GET_PARAM(1);
        useRoi = GET_PARAM(2);
    }

    void random_roi()
    {
        const int type = CV_MAKE_TYPE(depth, channels);

        const double upValue = 256;
        const double sumMinValue = 0.01; // we don't want to divide by "zero"

        Size roiSize = randomSize(1, 20);
        Border src1Border = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src1, src1_roi, roiSize, src1Border, type, -upValue, upValue);

        Border src2Border = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src2, src2_roi, roiSize, src2Border, type, -upValue, upValue);

        Border weights1Border = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(weights1, weights1_roi, roiSize, weights1Border, CV_32FC1, -upValue, upValue);

        Border weights2Border = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(weights2, weights2_roi, roiSize, weights2Border, CV_32FC1, sumMinValue, upValue); // fill it as a (w1 + w12)

        weights2_roi = weights2_roi - weights1_roi;
        // check that weights2_roi is still a part of weights2 (not a new matrix)
        CV_Assert(checkNorm(weights2_roi,
            weights2(Rect(weights2Border.lef, weights2Border.top, roiSize.width, roiSize.height))) < 1e-6);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, type, 5, 16);

        generateOclMat(gsrc1, gsrc1_roi, src1, roiSize, src1Border);
        generateOclMat(gsrc2, gsrc2_roi, src2, roiSize, src2Border);
        generateOclMat(gweights1, gweights1_roi, weights1, roiSize, weights1Border);
        generateOclMat(gweights2, gweights2_roi, weights2, roiSize, weights2Border);
        generateOclMat(gdst, gdst_roi, dst, roiSize, dstBorder);
    }

    void Near(double eps = 0.0)
    {
        Mat whole, roi;
        gdst.download(whole);
        gdst_roi.download(roi);

        EXPECT_MAT_NEAR(dst, whole, eps);
        EXPECT_MAT_NEAR(dst_roi, roi, eps);
    }
};

typedef void (*blendLinearFunc)(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &weights1, const cv::Mat &weights2, cv::Mat &result_gold);

OCL_TEST_P(Blend, Accuracy)
{
    for (int i = 0; i < LOOP_TIMES; ++i)
    {
        random_roi();

        cv::ocl::blendLinear(gsrc1_roi, gsrc2_roi, gweights1_roi, gweights2_roi, gdst_roi);

        static blendLinearFunc funcs[] = {
            blendLinearGold<uchar>,
            blendLinearGold<schar>,
            blendLinearGold<ushort>,
            blendLinearGold<short>,
            blendLinearGold<int>,
            blendLinearGold<float>,
        };

        blendLinearFunc func = funcs[depth];
        func(src1_roi, src2_roi, weights1_roi, weights2_roi, dst_roi);

        Near(depth <= CV_32S ? 1.0 : 0.2);
    }
}

INSTANTIATE_TEST_CASE_P(OCL_ImgProc, Blend,
                        Combine(testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F),
                                testing::Range(1, 5), Bool()));
