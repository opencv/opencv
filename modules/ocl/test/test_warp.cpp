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

using namespace cv;
using namespace testing;
using namespace std;

static MatType noType = -1;

/////////////////////////////////////////////////////////////////////////////////////////////////
// warpAffine  & warpPerspective

PARAM_TEST_CASE(WarpTestBase, MatType, Interpolation, bool, bool)
{
    int type, interpolation;
    Size dsize;
    bool useRoi, mapInverse;

    Mat src, dst_whole, src_roi, dst_roi;
    ocl::oclMat gsrc_whole, gsrc_roi, gdst_whole, gdst_roi;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        interpolation = GET_PARAM(1);
        mapInverse = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        if (mapInverse)
            interpolation |= WARP_INVERSE_MAP;
    }

    void random_roi()
    {
        Size roiSize = randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, -MAX_VALUE, MAX_VALUE);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst_whole, dst_roi, roiSize, dstBorder, type, -MAX_VALUE, MAX_VALUE);

        generateOclMat(gsrc_whole, gsrc_roi, src, roiSize, srcBorder);
        generateOclMat(gdst_whole, gdst_roi, dst_whole, roiSize, dstBorder);

        dsize = randomSize(1, MAX_VALUE);
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

/////warpAffine

typedef WarpTestBase WarpAffine;

OCL_TEST_P(WarpAffine, Mat)
{
    static const double coeffs[2][3] =
    {
        { cos(CV_PI / 6), -sin(CV_PI / 6),  100.0 },
        { sin(CV_PI / 6),  cos(CV_PI / 6), -100.0 }
    };

    static Mat M(2, 3, CV_64FC1, (void *)coeffs);

    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        warpAffine(src_roi, dst_roi, M, dsize, interpolation);
        ocl::warpAffine(gsrc_roi, gdst_roi, M, dsize, interpolation);

        Near(1.0);
    }
}

// warpPerspective

typedef WarpTestBase WarpPerspective;

OCL_TEST_P(WarpPerspective, Mat)
{
    static const double coeffs[3][3] =
    {
        { cos(CV_PI / 6), -sin(CV_PI / 6),  100.0 },
        { sin(CV_PI / 6),  cos(CV_PI / 6), -100.0 },
        { 0.0,             0.0,             1.0   }
    };

    static Mat M(3, 3, CV_64FC1, (void *)coeffs);

    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        warpPerspective(src_roi, dst_roi, M, dsize, interpolation);
        ocl::warpPerspective(gsrc_roi, gdst_roi, M, dsize, interpolation);

        Near(1.0);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// remap

PARAM_TEST_CASE(Remap, MatDepth, Channels, pair<MatType, MatType>, Border, bool)
{
    int srcType, map1Type, map2Type;
    int borderType;
    bool useRoi;

    Scalar val;

    Mat src, src_roi;
    Mat dst, dst_roi;
    Mat map1, map1_roi;
    Mat map2, map2_roi;

    // ocl mat with roi
    ocl::oclMat gsrc, gsrc_roi;
    ocl::oclMat gdst, gdst_roi;
    ocl::oclMat gmap1, gmap1_roi;
    ocl::oclMat gmap2, gmap2_roi;

    virtual void SetUp()
    {
        srcType = CV_MAKE_TYPE(GET_PARAM(0), GET_PARAM(1));
        map1Type = GET_PARAM(2).first;
        map2Type = GET_PARAM(2).second;
        borderType = GET_PARAM(3);
        useRoi = GET_PARAM(4);
    }

    void random_roi()
    {
        val = randomScalar(-MAX_VALUE, MAX_VALUE);
        Size srcROISize = randomSize(1, MAX_VALUE);
        Size dstROISize = randomSize(1, MAX_VALUE);

        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, srcROISize, srcBorder, srcType, 5, 256);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, dstROISize, dstBorder, srcType, -MAX_VALUE, MAX_VALUE);

        int mapMaxValue = MAX_VALUE << 2;
        Border map1Border = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(map1, map1_roi, dstROISize, map1Border, map1Type, -mapMaxValue, mapMaxValue);

        Border map2Border = randomBorder(0, useRoi ? MAX_VALUE : 0);
        if (map2Type != noType)
            randomSubMat(map2, map2_roi, dstROISize, map2Border, map2Type, -mapMaxValue, mapMaxValue);

        generateOclMat(gsrc, gsrc_roi, src, srcROISize, srcBorder);
        generateOclMat(gdst, gdst_roi, dst, dstROISize, dstBorder);
        generateOclMat(gmap1, gmap1_roi, map1, dstROISize, map1Border);
        if (noType != map2Type)
            generateOclMat(gmap2, gmap2_roi, map2, dstROISize, map2Border);
    }

    void Near(double threshold = 0.0)
    {
        Mat whole, roi;
        gdst.download(whole);
        gdst_roi.download(roi);

        EXPECT_MAT_NEAR(dst, whole, threshold);
        EXPECT_MAT_NEAR(dst_roi, roi, threshold);
    }
};

typedef Remap Remap_INTER_NEAREST;

OCL_TEST_P(Remap_INTER_NEAREST, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        remap(src_roi, dst_roi, map1_roi, map2_roi, INTER_NEAREST, borderType, val);
        ocl::remap(gsrc_roi, gdst_roi, gmap1_roi, gmap2_roi, INTER_NEAREST, borderType, val);

        Near(1.0);
    }
}

typedef Remap Remap_INTER_LINEAR;

OCL_TEST_P(Remap_INTER_LINEAR, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::remap(src_roi, dst_roi, map1_roi, map2_roi, INTER_LINEAR, borderType, val);
        ocl::remap(gsrc_roi, gdst_roi, gmap1_roi, gmap2_roi, INTER_LINEAR, borderType, val);

        Near(2.0);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// resize

PARAM_TEST_CASE(Resize, MatType, double, double, Interpolation, bool)
{
    int type, interpolation;
    double fx, fy;
    bool useRoi;

    Mat src, dst_whole, src_roi, dst_roi;
    ocl::oclMat gsrc_whole, gsrc_roi, gdst_whole, gdst_roi;

    virtual void SetUp()
    {
        type = GET_PARAM(0);
        fx = GET_PARAM(1);
        fy = GET_PARAM(2);
        interpolation = GET_PARAM(3);
        useRoi = GET_PARAM(4);
    }

    void random_roi()
    {
        CV_Assert(fx > 0 && fy > 0);

        Size srcRoiSize = randomSize(1, MAX_VALUE), dstRoiSize;
        dstRoiSize.width = cvRound(srcRoiSize.width * fx);
        dstRoiSize.height = cvRound(srcRoiSize.height * fy);

        if (dstRoiSize.area() == 0)
        {
            random_roi();
            return;
        }

        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, srcRoiSize, srcBorder, type, -MAX_VALUE, MAX_VALUE);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst_whole, dst_roi, dstRoiSize, dstBorder, type, -MAX_VALUE, MAX_VALUE);

        generateOclMat(gsrc_whole, gsrc_roi, src, srcRoiSize, srcBorder);
        generateOclMat(gdst_whole, gdst_roi, dst_whole, dstRoiSize, dstBorder);
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

OCL_TEST_P(Resize, Mat)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        cv::resize(src_roi, dst_roi, Size(), fx, fy, interpolation);
        ocl::resize(gsrc_roi, gdst_roi, Size(), fx, fy, interpolation);

        Near(1.0);
    }
}

/////////////////////////////////////////////////////////////////////////////////////

INSTANTIATE_TEST_CASE_P(ImgprocWarp, WarpAffine, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                            Values((Interpolation)INTER_NEAREST, (Interpolation)INTER_LINEAR, (Interpolation)INTER_CUBIC),
                            Bool(),
                            Bool()));

INSTANTIATE_TEST_CASE_P(ImgprocWarp, WarpPerspective, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                            Values((Interpolation)INTER_NEAREST, (Interpolation)INTER_LINEAR, (Interpolation)INTER_CUBIC),
                            Bool(),
                            Bool()));

INSTANTIATE_TEST_CASE_P(ImgprocWarp, Remap_INTER_LINEAR, Combine(
                            Values(CV_8U, CV_16U, CV_16S, CV_32F, CV_64F),
                            Values(1, 2, 3, 4),
                            Values(pair<MatType, MatType>((MatType)CV_32FC1, (MatType)CV_32FC1),
                                   pair<MatType, MatType>((MatType)CV_32FC2, noType)),
                            Values((Border)BORDER_CONSTANT,
                                   (Border)BORDER_REPLICATE,
                                   (Border)BORDER_WRAP,
                                   (Border)BORDER_REFLECT,
                                   (Border)BORDER_REFLECT_101),
                            Bool()));

INSTANTIATE_TEST_CASE_P(ImgprocWarp, Remap_INTER_NEAREST, Combine(
                            Values(CV_8U, CV_16U, CV_16S, CV_32F, CV_64F),
                            Values(1, 2, 3, 4),
                            Values(pair<MatType, MatType>((MatType)CV_32FC1, (MatType)CV_32FC1),
                                   pair<MatType, MatType>((MatType)CV_32FC2, noType),
                                   pair<MatType, MatType>((MatType)CV_16SC2, noType)),
                            Values((Border)BORDER_CONSTANT,
                                   (Border)BORDER_REPLICATE,
                                   (Border)BORDER_WRAP,
                                   (Border)BORDER_REFLECT,
                                   (Border)BORDER_REFLECT_101),
                            Bool()));

INSTANTIATE_TEST_CASE_P(ImgprocWarp, Resize, Combine(
                            Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                            Values(0.5, 1.5, 2.0),
                            Values(0.5, 1.5, 2.0),
                            Values((Interpolation)INTER_NEAREST, (Interpolation)INTER_LINEAR),
                            Bool()));

#endif // HAVE_OPENCL
