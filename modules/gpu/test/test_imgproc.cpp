/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#ifdef HAVE_CUDA

using namespace cvtest;
using namespace testing;

///////////////////////////////////////////////////////////////////////////////////////////////////////
// threshold

PARAM_TEST_CASE(Threshold, cv::gpu::DeviceInfo, MatType, ThreshOp, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int threshOp;
    bool useRoi;

    cv::Size size;
    cv::Mat src;
    double maxVal;
    double thresh;

    cv::Mat dst_gold;
    
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        threshOp = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(20, 150), rng.uniform(20, 150));

        src = randomMat(rng, size, type, 0.0, 127.0, false);

        maxVal = rng.uniform(20.0, 127.0);
        thresh = rng.uniform(0.0, maxVal);

        cv::threshold(src, dst_gold, thresh, maxVal, threshOp);
    }
};

TEST_P(Threshold, Accuracy)
{
    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::threshold(loadMat(src, useRoi), gpuRes, thresh, maxVal, threshOp);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, Threshold, Combine(
                        ALL_DEVICES, 
                        Values(CV_8UC1, CV_32FC1), 
                        Values((int)cv::THRESH_BINARY, (int)cv::THRESH_BINARY_INV, (int)cv::THRESH_TRUNC, (int)cv::THRESH_TOZERO, (int)cv::THRESH_TOZERO_INV),
                        USE_ROI));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// resize

PARAM_TEST_CASE(Resize, cv::gpu::DeviceInfo, MatType, Interpolation, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int interpolation;
    bool useRoi;

    cv::Size size;
    cv::Mat src;

    cv::Mat dst_gold_up;
    cv::Mat dst_gold_down;
    
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        interpolation = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(20, 150), rng.uniform(20, 150));

        src = randomMat(rng, size, type, 0.0, CV_MAT_DEPTH(type) == CV_32F ? 1.0 : 255.0, false);

        cv::resize(src, dst_gold_up, cv::Size(), 2.0, 2.0, interpolation);
        cv::resize(src, dst_gold_down, cv::Size(), 0.5, 0.5, interpolation);
    }
};

TEST_P(Resize, Up)
{
    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::resize(loadMat(src, useRoi), gpuRes, cv::Size(), 2.0, 2.0, interpolation);

    gpuRes.download(dst);

    EXPECT_MAT_SIMILAR(dst_gold_up, dst, 0.21);
}

TEST_P(Resize, Down)
{
    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::resize(loadMat(src, useRoi), gpuRes, cv::Size(), 0.5, 0.5, interpolation);

    gpuRes.download(dst);

    EXPECT_MAT_SIMILAR(dst_gold_down, dst, 0.22);
}

INSTANTIATE_TEST_CASE_P(ImgProc, Resize, Combine(
                        ALL_DEVICES, 
                        Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4), 
                        Values((int)cv::INTER_NEAREST, (int)cv::INTER_LINEAR, (int)cv::INTER_CUBIC),
                        USE_ROI));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// remap

PARAM_TEST_CASE(Remap, cv::gpu::DeviceInfo, MatType, Interpolation, Border, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int interpolation;
    int borderType;
    bool useRoi;

    cv::Size size;
    cv::Mat src;
    cv::Mat xmap;
    cv::Mat ymap;

    cv::Mat dst_gold;
    
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        interpolation = GET_PARAM(2);
        borderType = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        src = randomMat(rng, size, type, 0.0, 256.0, false);

        xmap = randomMat(rng, size, CV_32FC1, -20.0, src.cols + 20, false);
        ymap = randomMat(rng, size, CV_32FC1, -20.0, src.rows + 20, false);
        
        cv::remap(src, dst_gold, xmap, ymap, interpolation, borderType);
    }
};

TEST_P(Remap, Accuracy)
{
    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;
    
    cv::gpu::remap(loadMat(src, useRoi), gpuRes, loadMat(xmap, useRoi), loadMat(ymap, useRoi), interpolation, borderType);

    gpuRes.download(dst);

    EXPECT_MAT_SIMILAR(dst_gold, dst, 1e-1);
}

INSTANTIATE_TEST_CASE_P(ImgProc, Remap, Combine(
                        ALL_DEVICES, 
                        Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                        Values((int)cv::INTER_NEAREST, (int)cv::INTER_LINEAR, (int)cv::INTER_CUBIC),
                        Values((int)cv::BORDER_REFLECT101, (int)cv::BORDER_REPLICATE, (int)cv::BORDER_CONSTANT, (int)cv::BORDER_REFLECT, (int)cv::BORDER_WRAP),
                        USE_ROI));
                        
///////////////////////////////////////////////////////////////////////////////////////////////////////
// copyMakeBorder

PARAM_TEST_CASE(CopyMakeBorder, cv::gpu::DeviceInfo, MatType, Border, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int borderType;
    bool useRoi;

    cv::Size size;
    cv::Mat src;
    int top;
    int botton;
    int left;
    int right;
    cv::Scalar val;

    cv::Mat dst_gold;
    
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        borderType = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(20, 150), rng.uniform(20, 150));

        src = randomMat(rng, size, type, 0.0, 127.0, false);
        
        top = rng.uniform(1, 10);
        botton = rng.uniform(1, 10);
        left = rng.uniform(1, 10);
        right = rng.uniform(1, 10);
        val = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

        cv::copyMakeBorder(src, dst_gold, top, botton, left, right, borderType, val);
    }
};

TEST_P(CopyMakeBorder, Accuracy)
{
    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::copyMakeBorder(loadMat(src, useRoi), gpuRes, top, botton, left, right, borderType, val);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, CopyMakeBorder, Combine(
                        ALL_DEVICES, 
                        Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_16SC1, CV_16SC3, CV_16SC4, CV_32FC1, CV_32FC3, CV_32FC4),
                        Values((int)cv::BORDER_REFLECT101, (int)cv::BORDER_REPLICATE, (int)cv::BORDER_CONSTANT, (int)cv::BORDER_REFLECT, (int)cv::BORDER_WRAP),
                        USE_ROI));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// warpAffine & warpPerspective

PARAM_TEST_CASE(WarpAffine, cv::gpu::DeviceInfo, MatType, WarpFlags)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int flags;

    cv::Size size;
    cv::Mat src;
    cv::Mat M;

    cv::Mat dst_gold;
    
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        flags = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(20, 150), rng.uniform(20, 150));

        src = randomMat(rng, size, type, 0.0, 127.0, false);

        static double reflect[2][3] = { {-1,  0, 0},
                                        { 0, -1, 0}};
        reflect[0][2] = size.width;
        reflect[1][2] = size.height;
        M = cv::Mat(2, 3, CV_64F, (void*)reflect); 

        cv::warpAffine(src, dst_gold, M, src.size(), flags);       
    }
};

TEST_P(WarpAffine, Accuracy)
{
    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::warpAffine(loadMat(src), gpuRes, M, src.size(), flags);

    gpuRes.download(dst);

    // Check inner parts (ignoring 1 pixel width border)
    cv::Mat dst_gold_roi = dst_gold.rowRange(1, dst_gold.rows - 1).colRange(1, dst_gold.cols - 1);
    cv::Mat dst_roi = dst.rowRange(1, dst.rows - 1).colRange(1, dst.cols - 1);

    EXPECT_MAT_NEAR(dst_gold_roi, dst_roi, 1e-3);
}

INSTANTIATE_TEST_CASE_P(ImgProc, WarpAffine, Combine(
                        ALL_DEVICES, 
                        Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                        Values((int) cv::INTER_NEAREST, (int) cv::INTER_LINEAR, (int) cv::INTER_CUBIC, 
                               (int) (cv::INTER_NEAREST | cv::WARP_INVERSE_MAP), (int) (cv::INTER_LINEAR | cv::WARP_INVERSE_MAP), 
                               (int) (cv::INTER_CUBIC | cv::WARP_INVERSE_MAP))));

PARAM_TEST_CASE(WarpPerspective, cv::gpu::DeviceInfo, MatType, WarpFlags)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int flags;

    cv::Size size;
    cv::Mat src;
    cv::Mat M;

    cv::Mat dst_gold;
    
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        flags = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(20, 150), rng.uniform(20, 150));

        src = randomMat(rng, size, type, 0.0, 127.0, false);

        static double reflect[3][3] = { { -1, 0, 0},
                                        { 0, -1, 0},
                                        { 0,  0, 1}};
        reflect[0][2] = size.width;
        reflect[1][2] = size.height;
        M = cv::Mat(3, 3, CV_64F, (void*)reflect);

        cv::warpPerspective(src, dst_gold, M, src.size(), flags);       
    }
};

TEST_P(WarpPerspective, Accuracy)
{
    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::warpPerspective(loadMat(src), gpuRes, M, src.size(), flags);

    gpuRes.download(dst);

    // Check inner parts (ignoring 1 pixel width border)
    cv::Mat dst_gold_roi = dst_gold.rowRange(1, dst_gold.rows - 1).colRange(1, dst_gold.cols - 1);
    cv::Mat dst_roi = dst.rowRange(1, dst.rows - 1).colRange(1, dst.cols - 1);

    EXPECT_MAT_NEAR(dst_gold_roi, dst_roi, 1e-3);
}

INSTANTIATE_TEST_CASE_P(ImgProc, WarpPerspective, Combine(
                        ALL_DEVICES, 
                        Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                        Values((int) cv::INTER_NEAREST, (int) cv::INTER_LINEAR, (int) cv::INTER_CUBIC, 
                               (int) (cv::INTER_NEAREST | cv::WARP_INVERSE_MAP), (int) (cv::INTER_LINEAR | cv::WARP_INVERSE_MAP), 
                               (int) (cv::INTER_CUBIC | cv::WARP_INVERSE_MAP))));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// integral

PARAM_TEST_CASE(Integral, cv::gpu::DeviceInfo, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    bool useRoi;

    cv::Size size;
    cv::Mat src;

    cv::Mat dst_gold;
    
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        useRoi = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(20, 150), rng.uniform(20, 150));

        src = randomMat(rng, size, CV_8UC1, 0.0, 255.0, false); 

        cv::integral(src, dst_gold, CV_32S);     
    }
};

TEST_P(Integral, Accuracy)
{
    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::integral(loadMat(src, useRoi), gpuRes);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, Integral, Combine(
                        ALL_DEVICES, 
                        USE_ROI));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// cvtColor

PARAM_TEST_CASE(CvtColor, cv::gpu::DeviceInfo, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    bool useRoi;
    
    cv::Mat img;
    
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());
        
        cv::Mat imgBase = readImage("stereobm/aloe-L.png");
        ASSERT_FALSE(imgBase.empty());

        imgBase.convertTo(img, type, type == CV_32F ? 1.0 / 255.0 : 1.0);
    }
};

TEST_P(CvtColor, BGR2RGB)
{
    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2RGB);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR2RGB);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR2RGBA)
{
    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2RGBA);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR2RGBA);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR2BGRA)
{
    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2BGRA);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR2BGRA);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGRA2RGB)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGRA);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGRA2RGB);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGRA2RGB);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGRA2BGR)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGRA);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGRA2BGR);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGRA2BGR);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGRA2RGBA)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGRA);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGRA2RGBA);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGRA2RGBA);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR2GRAY)
{
    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2GRAY);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR2GRAY);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, RGB2GRAY)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGB);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2GRAY);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGB2GRAY);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, GRAY2BGR)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2GRAY);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_GRAY2BGR);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_GRAY2BGR);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, GRAY2BGRA)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2GRAY);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_GRAY2BGRA, 4);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_GRAY2BGRA, 4);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGRA2GRAY)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGRA);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGRA2GRAY);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGRA2GRAY);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, RGBA2GRAY)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGBA);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGBA2GRAY);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGBA2GRAY);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, BGR2BGR565)
{
    if (type != CV_8U)
        return;

    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2BGR565);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR2BGR565);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, RGB2BGR565)
{
    if (type != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGB);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2BGR565);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGB2BGR565);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5652BGR)
{
    if (type != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR565);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5652BGR);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR5652BGR);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5652RGB)
{
    if (type != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR565);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5652RGB);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR5652RGB);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGRA2BGR565)
{
    if (type != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGRA);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGRA2BGR565);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGRA2BGR565);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, RGBA2BGR565)
{
    if (type != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGBA);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGBA2BGR565);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGBA2BGR565);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5652BGRA)
{
    if (type != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR565);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5652BGRA, 4);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR5652BGRA, 4);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5652RGBA)
{
    if (type != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR565);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5652RGBA, 4);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR5652RGBA, 4);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, GRAY2BGR565)
{
    if (type != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2GRAY);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_GRAY2BGR565);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_GRAY2BGR565);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5652GRAY)
{
    if (type != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR565);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5652GRAY);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR5652GRAY);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR2BGR555)
{
    if (type != CV_8U)
        return;

    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2BGR555);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR2BGR555);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, RGB2BGR555)
{
    if (type != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGB);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2BGR555);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGB2BGR555);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5552BGR)
{
    if (type != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR555);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5552BGR);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR5552BGR);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5552RGB)
{
    if (type != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR555);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5552RGB);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR5552RGB);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGRA2BGR555)
{
    if (type != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGRA);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGRA2BGR555);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGRA2BGR555);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, RGBA2BGR555)
{
    if (type != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGBA);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGBA2BGR555);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGBA2BGR555);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5552BGRA)
{
    if (type != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR555);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5552BGRA, 4);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR5552BGRA, 4);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5552RGBA)
{
    if (type != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR555);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5552RGBA, 4);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR5552RGBA, 4);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, GRAY2BGR555)
{
    if (type != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2GRAY);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_GRAY2BGR555);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_GRAY2BGR555);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR5552GRAY)
{
    if (type != CV_8U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGR555);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR5552GRAY);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR5552GRAY);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR2XYZ)
{
    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2XYZ);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR2XYZ);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, RGB2XYZ)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGB);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2XYZ);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR2XYZ);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, BGR2XYZ4)
{
    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2XYZ);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR2XYZ, 4);

    gpuRes.download(dst);

    ASSERT_EQ(4, dst.channels());

    cv::Mat channels[4];
    cv::split(dst, channels);
    cv::merge(channels, 3, dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, BGRA2XYZ4)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2BGRA);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2XYZ);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR2XYZ, 4);

    gpuRes.download(dst);

    ASSERT_EQ(4, dst.channels());

    cv::Mat channels[4];
    cv::split(dst, channels);
    cv::merge(channels, 3, dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, XYZ2BGR)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2XYZ);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_XYZ2BGR);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_XYZ2BGR);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, XYZ2RGB)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2XYZ);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_XYZ2RGB);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_XYZ2RGB);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, XYZ42BGR)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2XYZ);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_XYZ2BGR);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), type, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_XYZ2BGR);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, XYZ42BGRA)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2XYZ);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_XYZ2BGR, 4);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), type, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_XYZ2BGR, 4);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, BGR2YCrCb)
{
    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2YCrCb);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR2YCrCb);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, RGB2YCrCb)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGB);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2YCrCb);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGB2YCrCb);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, BGR2YCrCb4)
{
    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2YCrCb);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR2YCrCb, 4);

    gpuRes.download(dst);

    ASSERT_EQ(4, dst.channels());

    cv::Mat channels[4];
    cv::split(dst, channels);
    cv::merge(channels, 3, dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, RGBA2YCrCb4)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGBA);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2YCrCb);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR2YCrCb, 4);

    gpuRes.download(dst);

    ASSERT_EQ(4, dst.channels());

    cv::Mat channels[4];
    cv::split(dst, channels);
    cv::merge(channels, 3, dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, YCrCb2BGR)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2YCrCb);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_YCrCb2BGR);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_YCrCb2BGR);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, YCrCb2RGB)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2YCrCb);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_YCrCb2RGB);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_YCrCb2RGB);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, YCrCb42RGB)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2YCrCb);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_YCrCb2RGB);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), type, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_YCrCb2RGB);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, YCrCb42RGBA)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2YCrCb);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_YCrCb2RGB, 4);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), type, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_YCrCb2RGB, 4);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, BGR2HSV)
{
    if (type == CV_16U)
        return;

    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2HSV);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR2HSV);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGB2HSV)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGB);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HSV);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGB2HSV);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGB2HSV4)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGB);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HSV);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGB2HSV, 4);

    gpuRes.download(dst);

    ASSERT_EQ(4, dst.channels());

    cv::Mat channels[4];
    cv::split(dst, channels);
    cv::merge(channels, 3, dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGBA2HSV4)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGBA);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HSV);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGB2HSV, 4);

    gpuRes.download(dst);

    ASSERT_EQ(4, dst.channels());

    cv::Mat channels[4];
    cv::split(dst, channels);
    cv::merge(channels, 3, dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, BGR2HLS)
{
    if (type == CV_16U)
        return;

    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2HLS);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR2HLS);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGB2HLS)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGB);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HLS);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGB2HLS);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGB2HLS4)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGB);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HLS);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGB2HLS, 4);

    gpuRes.download(dst);

    ASSERT_EQ(4, dst.channels());

    cv::Mat channels[4];
    cv::split(dst, channels);
    cv::merge(channels, 3, dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGBA2HLS4)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGBA);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HLS);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGB2HLS, 4);

    gpuRes.download(dst);

    ASSERT_EQ(4, dst.channels());

    cv::Mat channels[4];
    cv::split(dst, channels);
    cv::merge(channels, 3, dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HSV2BGR)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HSV);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HSV2BGR);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_HSV2BGR);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HSV2RGB)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HSV);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HSV2RGB);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_HSV2RGB);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HSV42BGR)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HSV);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HSV2BGR);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), type, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_HSV2BGR);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HSV42BGRA)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HSV);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HSV2BGR, 4);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), type, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_HSV2BGR, 4);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HLS2BGR)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HLS);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HLS2BGR);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_HLS2BGR);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HLS2RGB)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HLS);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HLS2RGB);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_HLS2RGB);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HLS42RGB)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HLS);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HLS2RGB);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), type, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_HLS2RGB);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HLS42RGBA)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HLS);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HLS2RGB, 4);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), type, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_HLS2RGB, 4);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, BGR2HSV_FULL)
{
    if (type == CV_16U)
        return;

    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2HSV_FULL);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR2HSV_FULL);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGB2HSV_FULL)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGB);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HSV_FULL);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGB2HSV_FULL);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGB2HSV4_FULL)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGB);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HSV_FULL);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGB2HSV_FULL, 4);

    gpuRes.download(dst);

    ASSERT_EQ(4, dst.channels());

    cv::Mat channels[4];
    cv::split(dst, channels);
    cv::merge(channels, 3, dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGBA2HSV4_FULL)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGBA);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HSV_FULL);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGB2HSV_FULL, 4);

    gpuRes.download(dst);

    ASSERT_EQ(4, dst.channels());

    cv::Mat channels[4];
    cv::split(dst, channels);
    cv::merge(channels, 3, dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, BGR2HLS_FULL)
{
    if (type == CV_16U)
        return;

    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2HLS_FULL);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR2HLS_FULL);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGB2HLS_FULL)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGB);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HLS_FULL);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGB2HLS_FULL);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGB2HLS4_FULL)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGB);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HLS_FULL);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGB2HLS_FULL, 4);

    gpuRes.download(dst);

    ASSERT_EQ(4, dst.channels());

    cv::Mat channels[4];
    cv::split(dst, channels);
    cv::merge(channels, 3, dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, RGBA2HLS4_FULL)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGBA);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2HLS_FULL);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGB2HLS_FULL, 4);

    gpuRes.download(dst);

    ASSERT_EQ(4, dst.channels());

    cv::Mat channels[4];
    cv::split(dst, channels);
    cv::merge(channels, 3, dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HSV2BGR_FULL)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HSV_FULL);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HSV2BGR_FULL);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_HSV2BGR_FULL);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HSV2RGB_FULL)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HSV_FULL);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HSV2RGB_FULL);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_HSV2RGB_FULL);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HSV42RGB_FULL)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HSV_FULL);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HSV2RGB_FULL);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), type, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_HSV2RGB_FULL);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HSV42RGBA_FULL)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HSV_FULL);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HSV2RGB_FULL, 4);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), type, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_HSV2RGB_FULL, 4);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HLS2BGR_FULL)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HLS_FULL);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HLS2BGR_FULL);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_HLS2BGR_FULL);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HLS2RGB_FULL)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HLS_FULL);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HLS2RGB_FULL);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_HLS2RGB_FULL);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HLS42RGB_FULL)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HLS_FULL);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HLS2RGB_FULL);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), type, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_HLS2RGB_FULL);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HLS42RGBA_FULL)
{
    if (type == CV_16U)
        return;

    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2HLS_FULL);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_HLS2RGB_FULL, 4);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), type, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_HLS2RGB_FULL, 4);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, BGR2YUV)
{
    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2YUV);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR2YUV);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, RGB2YUV)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGB);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2YUV);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGB2YUV);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, YUV2BGR)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2YUV);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_YUV2BGR);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_YUV2BGR);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, YUV42BGR)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2YUV);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_YUV2BGR);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), type, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_YUV2BGR);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, YUV42BGRA)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2YUV);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_YUV2BGR, 4);

    cv::Mat channels[4];
    cv::split(src, channels);
    channels[3] = cv::Mat(src.size(), type, cv::Scalar::all(0));
    cv::merge(channels, 4, src);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_YUV2BGR, 4);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, YUV2RGB)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_RGB2YUV);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_YUV2RGB);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_YUV2RGB);

    gpuRes.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, BGR2YUV4)
{
    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_BGR2YUV);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_BGR2YUV, 4);

    gpuRes.download(dst);

    ASSERT_EQ(4, dst.channels());

    cv::Mat channels[4];
    cv::split(dst, channels);
    cv::merge(channels, 3, dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, RGBA2YUV4)
{
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGBA);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, cv::COLOR_RGB2YUV);

    cv::Mat dst;

    cv::gpu::GpuMat gpuRes;

    cv::gpu::cvtColor(loadMat(src, useRoi), gpuRes, cv::COLOR_RGB2YUV, 4);

    gpuRes.download(dst);

    ASSERT_EQ(4, dst.channels());

    cv::Mat channels[4];
    cv::split(dst, channels);
    cv::merge(channels, 3, dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

INSTANTIATE_TEST_CASE_P(ImgProc, CvtColor, Combine(
                        ALL_DEVICES, 
                        Values(CV_8U, CV_16U, CV_32F),
                        USE_ROI));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// swapChannels

PARAM_TEST_CASE(SwapChannels, cv::gpu::DeviceInfo, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    bool useRoi;
    
    cv::Mat img;
    
    cv::Mat dst_gold;
    
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        useRoi = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());
        
        cv::Mat imgBase = readImage("stereobm/aloe-L.png");
        ASSERT_FALSE(imgBase.empty());

        cv::cvtColor(imgBase, img, cv::COLOR_BGR2BGRA);

        cv::cvtColor(img, dst_gold, cv::COLOR_BGRA2RGBA);
    }
};

TEST_P(SwapChannels, Accuracy)
{
    cv::gpu::GpuMat gpuImage = loadMat(img, useRoi);

    const int dstOrder[] = {2, 1, 0, 3};
    cv::gpu::swapChannels(gpuImage, dstOrder);

    cv::Mat dst;
    gpuImage.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, SwapChannels, Combine(ALL_DEVICES, USE_ROI));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// histograms

struct HistEven : TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;
    
    cv::Mat hsv;
    
    int hbins;
    float hranges[2];

    cv::Mat hist_gold;
    
    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());
        
        cv::Mat img = readImage("stereobm/aloe-L.png");
        ASSERT_FALSE(img.empty());
        
        cv::cvtColor(img, hsv, CV_BGR2HSV);

        hbins = 30;

        hranges[0] = 0;
        hranges[1] = 180;

        int histSize[] = {hbins};
        const float* ranges[] = {hranges};

        cv::MatND histnd;

        int channels[] = {0};
        cv::calcHist(&hsv, 1, channels, cv::Mat(), histnd, 1, histSize, ranges);

        hist_gold = histnd;
        hist_gold = hist_gold.t();
        hist_gold.convertTo(hist_gold, CV_32S);
    }
};

TEST_P(HistEven, Accuracy)
{
    cv::Mat hist;
    
    std::vector<cv::gpu::GpuMat> srcs;
    cv::gpu::split(loadMat(hsv), srcs);

    cv::gpu::GpuMat gpuHist;

    cv::gpu::histEven(srcs[0], gpuHist, hbins, (int)hranges[0], (int)hranges[1]);

    gpuHist.download(hist);

    EXPECT_MAT_NEAR(hist_gold, hist, 0.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, HistEven, ALL_DEVICES);

struct CalcHist : TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Size size;
    cv::Mat src;
    cv::Mat hist_gold;
    
    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));
        
        src = randomMat(rng, size, CV_8UC1, 0, 255, false);

        hist_gold.create(1, 256, CV_32SC1);
        hist_gold.setTo(cv::Scalar::all(0));

        int* hist = hist_gold.ptr<int>();
        for (int y = 0; y < src.rows; ++y)
        {
            const uchar* src_row = src.ptr(y);

            for (int x = 0; x < src.cols; ++x)
                ++hist[src_row[x]];
        }
    }
};

TEST_P(CalcHist, Accuracy)
{
    cv::Mat hist;
    
    cv::gpu::GpuMat gpuHist;

    cv::gpu::calcHist(loadMat(src), gpuHist);

    gpuHist.download(hist);

    EXPECT_MAT_NEAR(hist_gold, hist, 0.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, CalcHist, ALL_DEVICES);

struct EqualizeHist : TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Size size;
    cv::Mat src;
    cv::Mat dst_gold;
    
    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));
        
        src = randomMat(rng, size, CV_8UC1, 0, 255, false);

        cv::equalizeHist(src, dst_gold);
    }
};

TEST_P(EqualizeHist, Accuracy)
{
    cv::Mat dst;
    
    cv::gpu::GpuMat gpuDst;

    cv::gpu::equalizeHist(loadMat(src), gpuDst);

    gpuDst.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 3.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, EqualizeHist, ALL_DEVICES);

///////////////////////////////////////////////////////////////////////////////////////////////////////
// cornerHarris

PARAM_TEST_CASE(CornerHarris, cv::gpu::DeviceInfo, MatType, Border, int, int)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int borderType;
    int blockSize;
    int apertureSize;

    cv::Mat src;
    double k;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        borderType = GET_PARAM(2);
        blockSize = GET_PARAM(3);
        apertureSize = GET_PARAM(4); 

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        cv::Mat img = readImage("stereobm/aloe-L.png", CV_LOAD_IMAGE_GRAYSCALE);
        ASSERT_FALSE(img.empty());

        img.convertTo(src, type, type == CV_32F ? 1.0 / 255.0 : 1.0);

        k = rng.uniform(0.1, 0.9);

        cv::cornerHarris(src, dst_gold, blockSize, apertureSize, k, borderType);
    }
};

TEST_P(CornerHarris, Accuracy)
{
    cv::Mat dst;

    cv::gpu::GpuMat dev_dst;

    cv::gpu::cornerHarris(loadMat(src), dev_dst, blockSize, apertureSize, k, borderType);

    dev_dst.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.02);
}

INSTANTIATE_TEST_CASE_P(ImgProc, CornerHarris, Combine(
                        ALL_DEVICES, 
                        Values(CV_8UC1, CV_32FC1), 
                        Values((int) cv::BORDER_REFLECT101, (int) cv::BORDER_REPLICATE, (int) cv::BORDER_REFLECT),
                        Values(3, 5, 7),
                        Values(0, 3, 5, 7)));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// cornerMinEigen

PARAM_TEST_CASE(CornerMinEigen, cv::gpu::DeviceInfo, MatType, Border, int, int)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int borderType;
    int blockSize;
    int apertureSize;

    cv::Mat src;

    cv::Mat dst_gold;
    
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        borderType = GET_PARAM(2);
        blockSize = GET_PARAM(3);
        apertureSize = GET_PARAM(4); 

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        cv::Mat img = readImage("stereobm/aloe-L.png", CV_LOAD_IMAGE_GRAYSCALE);
        ASSERT_FALSE(img.empty());

        img.convertTo(src, type, type == CV_32F ? 1.0 / 255.0 : 1.0);

        cv::cornerMinEigenVal(src, dst_gold, blockSize, apertureSize, borderType);
    }
};

TEST_P(CornerMinEigen, Accuracy)
{
    cv::Mat dst;

    cv::gpu::GpuMat dev_dst;

    cv::gpu::cornerMinEigenVal(loadMat(src), dev_dst, blockSize, apertureSize, borderType);

    dev_dst.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.02);
}

INSTANTIATE_TEST_CASE_P(ImgProc, CornerMinEigen, Combine(
                        ALL_DEVICES, 
                        Values(CV_8UC1, CV_32FC1), 
                        Values((int) cv::BORDER_REFLECT101, (int) cv::BORDER_REPLICATE, (int) cv::BORDER_REFLECT),
                        Values(3, 5, 7),
                        Values(0, 3, 5, 7)));

////////////////////////////////////////////////////////////////////////
// ColumnSum

struct ColumnSum : TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Size size;
    cv::Mat src;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());
    
        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 400), rng.uniform(100, 400));

        src = randomMat(rng, size, CV_32F, 0.0, 1.0, false);
    }
};

TEST_P(ColumnSum, Accuracy)
{
    cv::Mat dst;
    
    cv::gpu::GpuMat dev_dst;

    cv::gpu::columnSum(loadMat(src), dev_dst);

    dev_dst.download(dst);

    for (int j = 0; j < src.cols; ++j)
    {
        float gold = src.at<float>(0, j);
        float res = dst.at<float>(0, j);
        ASSERT_NEAR(res, gold, 0.5);
    }

    for (int i = 1; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            float gold = src.at<float>(i, j) += src.at<float>(i - 1, j);
            float res = dst.at<float>(i, j);
            ASSERT_NEAR(res, gold, 0.5);
        }
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, ColumnSum, ALL_DEVICES);

////////////////////////////////////////////////////////////////////////
// Norm

PARAM_TEST_CASE(Norm, cv::gpu::DeviceInfo, MatType, NormCode, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int normType;
    bool useRoi;

    cv::Size size;
    cv::Mat src;

    double gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        normType = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    
        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 400), rng.uniform(100, 400));

        src = randomMat(rng, size, type, 0.0, 10.0, false);

        gold = cv::norm(src, normType);
    }
};

TEST_P(Norm, Accuracy)
{
    double res = cv::gpu::norm(loadMat(src, useRoi), normType);

    ASSERT_NEAR(res, gold, 0.5);
}

INSTANTIATE_TEST_CASE_P(ImgProc, Norm, Combine(
                        ALL_DEVICES, 
                        TYPES(CV_8U, CV_32F, 1, 1),
                        Values((int) cv::NORM_INF, (int) cv::NORM_L1, (int) cv::NORM_L2),
                        USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// reprojectImageTo3D

PARAM_TEST_CASE(ReprojectImageTo3D, cv::gpu::DeviceInfo, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    bool useRoi;

    cv::Size size;
    cv::Mat disp;
    cv::Mat Q;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        useRoi = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());
    
        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 500), rng.uniform(100, 500));

        disp = randomMat(rng, size, CV_8UC1, 5.0, 30.0, false);

        Q = randomMat(rng, cv::Size(4, 4), CV_32FC1, 0.1, 1.0, false);

        cv::reprojectImageTo3D(disp, dst_gold, Q, false);
    }
};

TEST_P(ReprojectImageTo3D, Accuracy)
{
    cv::Mat dst;

    cv::gpu::GpuMat gpures;

    cv::gpu::reprojectImageTo3D(loadMat(disp, useRoi), gpures, Q);

    gpures.download(dst);

    ASSERT_EQ(dst_gold.size(), dst.size());

    for (int y = 0; y < dst_gold.rows; ++y)
    {
        const cv::Vec3f* cpu_row = dst_gold.ptr<cv::Vec3f>(y);
        const cv::Vec4f* gpu_row = dst.ptr<cv::Vec4f>(y);

        for (int x = 0; x < dst_gold.cols; ++x)
        {
            cv::Vec3f gold = cpu_row[x];
            cv::Vec4f res = gpu_row[x];

            ASSERT_NEAR(res[0], gold[0], 1e-5);
            ASSERT_NEAR(res[1], gold[1], 1e-5);
            ASSERT_NEAR(res[2], gold[2], 1e-5);
        }
    }
}

INSTANTIATE_TEST_CASE_P(ImgProc, ReprojectImageTo3D, Combine(ALL_DEVICES, USE_ROI));

////////////////////////////////////////////////////////////////////////////////
// meanShift

struct MeanShift : TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;
    
    cv::Mat rgba;

    int spatialRad;
    int colorRad;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());
        
        cv::Mat img = readImage("meanshift/cones.png");
        ASSERT_FALSE(img.empty());
        
        cv::cvtColor(img, rgba, CV_BGR2BGRA);

        spatialRad = 30;
        colorRad = 30;
    }
};

TEST_P(MeanShift, Filtering)
{
    cv::Mat img_template;
    
    if (supportFeature(devInfo, cv::gpu::FEATURE_SET_COMPUTE_20))
        img_template = readImage("meanshift/con_result.png");
    else
        img_template = readImage("meanshift/con_result_CC1X.png");

    ASSERT_FALSE(img_template.empty());

    cv::Mat dst;

    cv::gpu::GpuMat dev_dst;

    cv::gpu::meanShiftFiltering(loadMat(rgba), dev_dst, spatialRad, colorRad);

    dev_dst.download(dst);

    ASSERT_EQ(CV_8UC4, dst.type());

    cv::Mat result;
    cv::cvtColor(dst, result, CV_BGRA2BGR);

    EXPECT_MAT_NEAR(img_template, result, 0.0);
}

TEST_P(MeanShift, Proc)
{
    cv::Mat spmap_template;
    cv::FileStorage fs;

    if (supportFeature(devInfo, cv::gpu::FEATURE_SET_COMPUTE_20))
        fs.open(std::string(cvtest::TS::ptr()->get_data_path()) + "meanshift/spmap.yaml", cv::FileStorage::READ);
    else
        fs.open(std::string(cvtest::TS::ptr()->get_data_path()) + "meanshift/spmap_CC1X.yaml", cv::FileStorage::READ);

    ASSERT_TRUE(fs.isOpened());

    fs["spmap"] >> spmap_template;

    ASSERT_TRUE(!rgba.empty() && !spmap_template.empty());

    cv::Mat rmap_filtered;
    cv::Mat rmap;
    cv::Mat spmap;

    cv::gpu::GpuMat d_rmap_filtered;
    cv::gpu::meanShiftFiltering(loadMat(rgba), d_rmap_filtered, spatialRad, colorRad);

    cv::gpu::GpuMat d_rmap;
    cv::gpu::GpuMat d_spmap;
    cv::gpu::meanShiftProc(loadMat(rgba), d_rmap, d_spmap, spatialRad, colorRad);

    d_rmap_filtered.download(rmap_filtered);
    d_rmap.download(rmap);
    d_spmap.download(spmap);

    ASSERT_EQ(CV_8UC4, rmap.type());
    
    EXPECT_MAT_NEAR(rmap_filtered, rmap, 0.0);    
    EXPECT_MAT_NEAR(spmap_template, spmap, 0.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, MeanShift, ALL_DEVICES);

PARAM_TEST_CASE(MeanShiftSegmentation, cv::gpu::DeviceInfo, int)
{
    cv::gpu::DeviceInfo devInfo;
    int minsize;
    
    cv::Mat rgba;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        minsize = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());
        
        cv::Mat img = readImage("meanshift/cones.png");
        ASSERT_FALSE(img.empty());
        
        cv::cvtColor(img, rgba, CV_BGR2BGRA);

        std::ostringstream path;
        path << "meanshift/cones_segmented_sp10_sr10_minsize" << minsize;
        if (supportFeature(devInfo, cv::gpu::FEATURE_SET_COMPUTE_20))
            path << ".png";
        else
            path << "_CC1X.png";

        dst_gold = readImage(path.str());
        ASSERT_FALSE(dst_gold.empty());
    }
};

TEST_P(MeanShiftSegmentation, Regression)
{
    cv::Mat dst;

    cv::gpu::meanShiftSegmentation(loadMat(rgba), dst, 10, 10, minsize);

    cv::Mat dst_rgb;
    cv::cvtColor(dst, dst_rgb, CV_BGRA2BGR);

    EXPECT_MAT_SIMILAR(dst_gold, dst_rgb, 1e-3);
}

INSTANTIATE_TEST_CASE_P(ImgProc, MeanShiftSegmentation, Combine(
                        ALL_DEVICES,
                        Values(0, 4, 20, 84, 340, 1364)));

////////////////////////////////////////////////////////////////////////////////
// matchTemplate

PARAM_TEST_CASE(MatchTemplate8U, cv::gpu::DeviceInfo, int, TemplateMethod)
{
    cv::gpu::DeviceInfo devInfo;
    int cn;
    int method;

    int n, m, h, w;
    cv::Mat image, templ;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        cn = GET_PARAM(1);
        method = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        n = rng.uniform(30, 100);
        m = rng.uniform(30, 100);
        h = rng.uniform(5, n - 1);
        w = rng.uniform(5, m - 1);

        image = randomMat(rng, cv::Size(m, n), CV_MAKETYPE(CV_8U, cn), 1.0, 10.0, false);
        templ = randomMat(rng, cv::Size(w, h), CV_MAKETYPE(CV_8U, cn), 1.0, 10.0, false);

        cv::matchTemplate(image, templ, dst_gold, method);
    }
};

TEST_P(MatchTemplate8U, Regression)
{
    cv::Mat dst;

    cv::gpu::GpuMat dev_dst;

    cv::gpu::matchTemplate(loadMat(image), loadMat(templ), dev_dst, method);

    dev_dst.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 5 * h * w * 1e-4);
}

INSTANTIATE_TEST_CASE_P(ImgProc, MatchTemplate8U, Combine(
                        ALL_DEVICES,
                        Range(1, 5), 
                        Values((int)cv::TM_SQDIFF, (int) cv::TM_SQDIFF_NORMED, (int) cv::TM_CCORR, (int) cv::TM_CCORR_NORMED, (int) cv::TM_CCOEFF, (int) cv::TM_CCOEFF_NORMED)));


PARAM_TEST_CASE(MatchTemplate32F, cv::gpu::DeviceInfo, int, TemplateMethod)
{
    cv::gpu::DeviceInfo devInfo;
    int cn;
    int method;

    int n, m, h, w;
    cv::Mat image, templ;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        cn = GET_PARAM(1);
        method = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        n = rng.uniform(30, 100);
        m = rng.uniform(30, 100);
        h = rng.uniform(5, n - 1);
        w = rng.uniform(5, m - 1);

        image = randomMat(rng, cv::Size(m, n), CV_MAKETYPE(CV_32F, cn), 0.001, 1.0, false);
        templ = randomMat(rng, cv::Size(w, h), CV_MAKETYPE(CV_32F, cn), 0.001, 1.0, false);

        cv::matchTemplate(image, templ, dst_gold, method);
    }
};

TEST_P(MatchTemplate32F, Regression)
{
    cv::Mat dst;

    cv::gpu::GpuMat dev_dst;

    cv::gpu::matchTemplate(loadMat(image), loadMat(templ), dev_dst, method);

    dev_dst.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.25 * h * w * 1e-4);
}

INSTANTIATE_TEST_CASE_P(ImgProc, MatchTemplate32F, Combine(
                        ALL_DEVICES, 
                        Range(1, 5), 
                        Values((int) cv::TM_SQDIFF, (int) cv::TM_CCORR)));


PARAM_TEST_CASE(MatchTemplateBlackSource, cv::gpu::DeviceInfo, TemplateMethod)
{
    cv::gpu::DeviceInfo devInfo;
    int method;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        method = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(MatchTemplateBlackSource, Accuracy)
{
    cv::Mat image = readImage("matchtemplate/black.png");
    ASSERT_FALSE(image.empty());

    cv::Mat pattern = readImage("matchtemplate/cat.png");
    ASSERT_FALSE(pattern.empty());

    cv::Point maxLocGold = cv::Point(284, 12);

    cv::Mat dst;

    cv::gpu::GpuMat dev_dst;

    cv::gpu::matchTemplate(loadMat(image), loadMat(pattern), dev_dst, method);

    dev_dst.download(dst);

    double maxValue;
    cv::Point maxLoc;
    cv::minMaxLoc(dst, NULL, &maxValue, NULL, &maxLoc);

    ASSERT_EQ(maxLocGold, maxLoc);
}

INSTANTIATE_TEST_CASE_P(ImgProc, MatchTemplateBlackSource, Combine(
                        ALL_DEVICES,
                        Values((int) cv::TM_CCOEFF_NORMED, (int) cv::TM_CCORR_NORMED)));


PARAM_TEST_CASE(MatchTemplate_CCOEF_NORMED, cv::gpu::DeviceInfo, std::pair<std::string, std::string>)
{
    cv::gpu::DeviceInfo devInfo;
    std::string imageName;
    std::string patternName;

    cv::Mat image, pattern;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        imageName = GET_PARAM(1).first;
        patternName = GET_PARAM(1).second;

        image = readImage(imageName);
        ASSERT_FALSE(image.empty());

        pattern = readImage(patternName);
        ASSERT_FALSE(pattern.empty());
    }
};

TEST_P(MatchTemplate_CCOEF_NORMED, Accuracy)
{
    cv::Mat dstGold;
    cv::matchTemplate(image, pattern, dstGold, CV_TM_CCOEFF_NORMED);

    double minValGold, maxValGold;
    cv::Point minLocGold, maxLocGold;
    cv::minMaxLoc(dstGold, &minValGold, &maxValGold, &minLocGold, &maxLocGold);

    cv::Mat dst;

    cv::gpu::GpuMat dev_dst;

    cv::gpu::matchTemplate(loadMat(image), loadMat(pattern), dev_dst, CV_TM_CCOEFF_NORMED);

    dev_dst.download(dst);

    cv::Point minLoc, maxLoc;
    double minVal, maxVal;
    cv::minMaxLoc(dst, &minVal, &maxVal, &minLoc, &maxLoc);

    ASSERT_EQ(minLocGold, minLoc);
    ASSERT_EQ(maxLocGold, maxLoc);
    ASSERT_LE(maxVal, 1.);
    ASSERT_GE(minVal, -1.);
}

INSTANTIATE_TEST_CASE_P(ImgProc, MatchTemplate_CCOEF_NORMED, Combine(
                        ALL_DEVICES,
                        Values(std::make_pair(std::string("matchtemplate/source-0.png"), std::string("matchtemplate/target-0.png")))));

////////////////////////////////////////////////////////////////////////////
// MulSpectrums

PARAM_TEST_CASE(MulSpectrums, cv::gpu::DeviceInfo, DftFlags)
{
    cv::gpu::DeviceInfo devInfo;
    int flag;

    cv::Mat a, b; 

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        flag = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        a = randomMat(rng, cv::Size(rng.uniform(100, 200), rng.uniform(100, 200)), CV_32FC2, 0.0, 10.0, false);
        b = randomMat(rng, a.size(), CV_32FC2, 0.0, 10.0, false);
    }
};

TEST_P(MulSpectrums, Simple)
{
    cv::Mat c_gold;
    cv::mulSpectrums(a, b, c_gold, flag, false);
    
    cv::Mat c;

    cv::gpu::GpuMat d_c;

    cv::gpu::mulSpectrums(loadMat(a), loadMat(b), d_c, flag, false);

    d_c.download(c);

    EXPECT_MAT_NEAR(c_gold, c, 1e-4);
}

TEST_P(MulSpectrums, Scaled)
{
    float scale = 1.f / a.size().area();

    cv::Mat c_gold;
    cv::mulSpectrums(a, b, c_gold, flag, false);
    c_gold.convertTo(c_gold, c_gold.type(), scale);

    cv::Mat c;

    cv::gpu::GpuMat d_c;

    cv::gpu::mulAndScaleSpectrums(loadMat(a), loadMat(b), d_c, flag, scale, false);

    d_c.download(c);

    EXPECT_MAT_NEAR(c_gold, c, 1e-4);
}

INSTANTIATE_TEST_CASE_P(ImgProc, MulSpectrums, Combine(
                        ALL_DEVICES, 
                        Values(0, (int) cv::DFT_ROWS)));

////////////////////////////////////////////////////////////////////////////
// Dft

struct Dft : TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    virtual void SetUp() 
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());
    }
};


void testC2C(const std::string& hint, int cols, int rows, int flags, bool inplace)
{
    SCOPED_TRACE(hint);

    cv::RNG& rng = cvtest::TS::ptr()->get_rng();

    cv::Mat a = randomMat(rng, cv::Size(cols, rows), CV_32FC2, 0.0, 10.0, false);

    cv::Mat b_gold;
    cv::dft(a, b_gold, flags);

    cv::gpu::GpuMat d_b;
    cv::gpu::GpuMat d_b_data;
    if (inplace)
    {
        d_b_data.create(1, a.size().area(), CV_32FC2);
        d_b = cv::gpu::GpuMat(a.rows, a.cols, CV_32FC2, d_b_data.ptr(), a.cols * d_b_data.elemSize());
    }
    cv::gpu::dft(loadMat(a), d_b, cv::Size(cols, rows), flags);

    EXPECT_TRUE(!inplace || d_b.ptr() == d_b_data.ptr());
    ASSERT_EQ(CV_32F, d_b.depth());
    ASSERT_EQ(2, d_b.channels());
    EXPECT_MAT_NEAR(b_gold, cv::Mat(d_b), rows * cols * 1e-4);
}

TEST_P(Dft, C2C)
{
    cv::RNG& rng = cvtest::TS::ptr()->get_rng();

    int cols = 2 + rng.next() % 100, rows = 2 + rng.next() % 100;

    for (int i = 0; i < 2; ++i)
    {
        bool inplace = i != 0;

        testC2C("no flags", cols, rows, 0, inplace);
        testC2C("no flags 0 1", cols, rows + 1, 0, inplace);
        testC2C("no flags 1 0", cols, rows + 1, 0, inplace);
        testC2C("no flags 1 1", cols + 1, rows, 0, inplace);
        testC2C("DFT_INVERSE", cols, rows, cv::DFT_INVERSE, inplace);
        testC2C("DFT_ROWS", cols, rows, cv::DFT_ROWS, inplace);
        testC2C("single col", 1, rows, 0, inplace);
        testC2C("single row", cols, 1, 0, inplace);
        testC2C("single col inversed", 1, rows, cv::DFT_INVERSE, inplace);
        testC2C("single row inversed", cols, 1, cv::DFT_INVERSE, inplace);
        testC2C("single row DFT_ROWS", cols, 1, cv::DFT_ROWS, inplace);
        testC2C("size 1 2", 1, 2, 0, inplace);
        testC2C("size 2 1", 2, 1, 0, inplace);
    }
}

void testR2CThenC2R(const std::string& hint, int cols, int rows, bool inplace)
{
    SCOPED_TRACE(hint);
    
    cv::RNG& rng = TS::ptr()->get_rng();

    cv::Mat a = randomMat(rng, cv::Size(cols, rows), CV_32FC1, 0.0, 10.0, false);

    cv::gpu::GpuMat d_b, d_c;
    cv::gpu::GpuMat d_b_data, d_c_data;
    if (inplace)
    {
        if (a.cols == 1)
        {
            d_b_data.create(1, (a.rows / 2 + 1) * a.cols, CV_32FC2);
            d_b = cv::gpu::GpuMat(a.rows / 2 + 1, a.cols, CV_32FC2, d_b_data.ptr(), a.cols * d_b_data.elemSize());
        }
        else
        {
            d_b_data.create(1, a.rows * (a.cols / 2 + 1), CV_32FC2);
            d_b = cv::gpu::GpuMat(a.rows, a.cols / 2 + 1, CV_32FC2, d_b_data.ptr(), (a.cols / 2 + 1) * d_b_data.elemSize());
        }
        d_c_data.create(1, a.size().area(), CV_32F);
        d_c = cv::gpu::GpuMat(a.rows, a.cols, CV_32F, d_c_data.ptr(), a.cols * d_c_data.elemSize());
    }

    cv::gpu::dft(loadMat(a), d_b, cv::Size(cols, rows), 0);
    cv::gpu::dft(d_b, d_c, cv::Size(cols, rows), cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    
    EXPECT_TRUE(!inplace || d_b.ptr() == d_b_data.ptr());
    EXPECT_TRUE(!inplace || d_c.ptr() == d_c_data.ptr());
    ASSERT_EQ(CV_32F, d_c.depth());
    ASSERT_EQ(1, d_c.channels());

    cv::Mat c(d_c);
    EXPECT_MAT_NEAR(a, c, rows * cols * 1e-5);
}

TEST_P(Dft, R2CThenC2R)
{
    cv::RNG& rng = TS::ptr()->get_rng();

    int cols = 2 + rng.next() % 100, rows = 2 + rng.next() % 100;

    testR2CThenC2R("sanity", cols, rows, false);
    testR2CThenC2R("sanity 0 1", cols, rows + 1, false);
    testR2CThenC2R("sanity 1 0", cols + 1, rows, false);
    testR2CThenC2R("sanity 1 1", cols + 1, rows + 1, false);
    testR2CThenC2R("single col", 1, rows, false);
    testR2CThenC2R("single col 1", 1, rows + 1, false);
    testR2CThenC2R("single row", cols, 1, false);
    testR2CThenC2R("single row 1", cols + 1, 1, false);

    testR2CThenC2R("sanity", cols, rows, true);
    testR2CThenC2R("sanity 0 1", cols, rows + 1, true);
    testR2CThenC2R("sanity 1 0", cols + 1, rows, true);
    testR2CThenC2R("sanity 1 1", cols + 1, rows + 1, true);
    testR2CThenC2R("single row", cols, 1, true);
    testR2CThenC2R("single row 1", cols + 1, 1, true);
}

INSTANTIATE_TEST_CASE_P(ImgProc, Dft, ALL_DEVICES);

////////////////////////////////////////////////////////////////////////////
// blend

template <typename T> 
void blendLinearGold(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& weights1, const cv::Mat& weights2, cv::Mat& result_gold)
{
    result_gold.create(img1.size(), img1.type());

    int cn = img1.channels();

    for (int y = 0; y < img1.rows; ++y)
    {
        const float* weights1_row = weights1.ptr<float>(y);
        const float* weights2_row = weights2.ptr<float>(y);
        const T* img1_row = img1.ptr<T>(y);
        const T* img2_row = img2.ptr<T>(y);
        T* result_gold_row = result_gold.ptr<T>(y);

        for (int x = 0; x < img1.cols * cn; ++x)
        {
            float w1 = weights1_row[x / cn];
            float w2 = weights2_row[x / cn];
            result_gold_row[x] = static_cast<T>((img1_row[x] * w1 + img2_row[x] * w2) / (w1 + w2 + 1e-5f));
        }
    }
}

PARAM_TEST_CASE(Blend, cv::gpu::DeviceInfo, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    bool useRoi;

    cv::Size size;
    cv::Mat img1;
    cv::Mat img2;
    cv::Mat weights1;
    cv::Mat weights2;

    cv::Mat result_gold;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(200 + randInt(rng) % 1000, 200 + randInt(rng) % 1000);

        int depth = CV_MAT_DEPTH(type);

        img1 = randomMat(rng, size, type, 0.0, depth == CV_8U ? 255.0 : 1.0, false);
        img2 = randomMat(rng, size, type, 0.0, depth == CV_8U ? 255.0 : 1.0, false);
        weights1 = randomMat(rng, size, CV_32F, 0, 1, false);
        weights2 = randomMat(rng, size, CV_32F, 0, 1, false);
        
        if (depth == CV_8U)
            blendLinearGold<uchar>(img1, img2, weights1, weights2, result_gold);
        else
            blendLinearGold<float>(img1, img2, weights1, weights2, result_gold);
    }
};

TEST_P(Blend, Accuracy)
{
    cv::Mat result;

    cv::gpu::GpuMat d_result;

    cv::gpu::blendLinear(loadMat(img1, useRoi), loadMat(img2, useRoi), loadMat(weights1, useRoi), loadMat(weights2, useRoi), d_result);

    d_result.download(result);

    EXPECT_MAT_NEAR(result_gold, result, CV_MAT_DEPTH(type) == CV_8U ? 1.0 : 1e-5);
}

INSTANTIATE_TEST_CASE_P(ImgProc, Blend, Combine(
                        ALL_DEVICES,
                        testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                        USE_ROI));

////////////////////////////////////////////////////////
// pyrDown

PARAM_TEST_CASE(PyrDown, cv::gpu::DeviceInfo, MatType, UseRoi)
{    
    cv::gpu::DeviceInfo devInfo;
    int type;
    bool useRoi;

    cv::Mat src;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = TS::ptr()->get_rng();

        cv::Size size(rng.uniform(100, 200), rng.uniform(100, 200));

        src = randomMat(rng, size, type, 0.0, 255.0, false);
        
        cv::pyrDown(src, dst_gold);
    }
};

TEST_P(PyrDown, Accuracy)
{    
    cv::Mat dst;

    cv::gpu::GpuMat d_dst;
    
    cv::gpu::pyrDown(loadMat(src, useRoi), d_dst);
    
    d_dst.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() == CV_32F ? 1e-4 : 1.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, PyrDown, Combine(
                        ALL_DEVICES, 
                        Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                        USE_ROI));

////////////////////////////////////////////////////////
// pyrUp

PARAM_TEST_CASE(PyrUp, cv::gpu::DeviceInfo, MatType, UseRoi)
{    
    cv::gpu::DeviceInfo devInfo;
    int type;
    bool useRoi;
    
    cv::Size size;    
    cv::Mat src;
    
    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());
        
        cv::RNG& rng = TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));

        src = randomMat(rng, size, type, 0.0, 255.0, false);
        
        cv::pyrUp(src, dst_gold);
    }
};

TEST_P(PyrUp, Accuracy)
{    
    cv::Mat dst;

    cv::gpu::GpuMat d_dst;
    
    cv::gpu::pyrUp(loadMat(src, useRoi), d_dst);
    
    d_dst.download(dst);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() == CV_32F ? 1e-4 : 1.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, PyrUp, Combine(
                        ALL_DEVICES, 
                        Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                        USE_ROI));

////////////////////////////////////////////////////////
// Canny

PARAM_TEST_CASE(Canny, cv::gpu::DeviceInfo, int, bool, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int apperture_size;
    bool L2gradient;
    bool useRoi;
    
    cv::Mat img;

    double low_thresh;
    double high_thresh;

    cv::Mat edges_gold;

    virtual void SetUp() 
    {
        devInfo = GET_PARAM(0);
        apperture_size = GET_PARAM(1);
        L2gradient = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
        
        img = readImage("stereobm/aloe-L.png", CV_LOAD_IMAGE_GRAYSCALE);
        ASSERT_FALSE(img.empty()); 

        low_thresh = 50.0;
        high_thresh = 100.0;
        
        cv::Canny(img, edges_gold, low_thresh, high_thresh, apperture_size, L2gradient);
    }
};

TEST_P(Canny, Accuracy)
{
    cv::Mat edges;

    cv::gpu::GpuMat d_edges;

    cv::gpu::Canny(loadMat(img, useRoi), d_edges, low_thresh, high_thresh, apperture_size, L2gradient);

    d_edges.download(edges);

    EXPECT_MAT_SIMILAR(edges_gold, edges, 1.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, Canny, testing::Combine(
                        DEVICES(cv::gpu::SHARED_ATOMICS),
                        Values(3, 5),
                        Values(false, true),
                        USE_ROI));

////////////////////////////////////////////////////////
// convolve

namespace
{
    void convolveDFT(const cv::Mat& A, const cv::Mat& B, cv::Mat& C, bool ccorr = false)
    {
        // reallocate the output array if needed
        C.create(std::abs(A.rows - B.rows) + 1, std::abs(A.cols - B.cols) + 1, A.type());
        Size dftSize;

        // compute the size of DFT transform
        dftSize.width = cv::getOptimalDFTSize(A.cols + B.cols - 1);
        dftSize.height = cv::getOptimalDFTSize(A.rows + B.rows - 1);

        // allocate temporary buffers and initialize them with 0’s
        cv::Mat tempA(dftSize, A.type(), cv::Scalar::all(0));
        cv::Mat tempB(dftSize, B.type(), cv::Scalar::all(0));

        // copy A and B to the top-left corners of tempA and tempB, respectively
        cv::Mat roiA(tempA, cv::Rect(0, 0, A.cols, A.rows));
        A.copyTo(roiA);
        cv::Mat roiB(tempB, cv::Rect(0, 0, B.cols, B.rows));
        B.copyTo(roiB);

        // now transform the padded A & B in-place;
        // use "nonzeroRows" hint for faster processing
        cv::dft(tempA, tempA, 0, A.rows);
        cv::dft(tempB, tempB, 0, B.rows);

        // multiply the spectrums;
        // the function handles packed spectrum representations well
        cv::mulSpectrums(tempA, tempB, tempA, 0, ccorr);

        // transform the product back from the frequency domain.
        // Even though all the result rows will be non-zero,
        // you need only the first C.rows of them, and thus you
        // pass nonzeroRows == C.rows
        cv::dft(tempA, tempA, cv::DFT_INVERSE + cv::DFT_SCALE, C.rows);

        // now copy the result back to C.
        tempA(cv::Rect(0, 0, C.cols, C.rows)).copyTo(C);
    }
}

PARAM_TEST_CASE(Convolve, cv::gpu::DeviceInfo, int, bool)
{    
    cv::gpu::DeviceInfo devInfo;
    int ksize;
    bool ccorr;
    
    cv::Mat src;
    cv::Mat kernel;
    
    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        ksize = GET_PARAM(1);
        ccorr = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());
        
        cv::RNG& rng = TS::ptr()->get_rng();

        cv::Size size(rng.uniform(200, 400), rng.uniform(200, 400));

        src = randomMat(rng, size, CV_32FC1, 0.0, 100.0, false);
        kernel = randomMat(rng, cv::Size(ksize, ksize), CV_32FC1, 0.0, 1.0, false);
        
        convolveDFT(src, kernel, dst_gold, ccorr);
    }
};

TEST_P(Convolve, Accuracy)
{
    cv::Mat dst;

    cv::gpu::GpuMat d_dst;

    cv::gpu::convolve(loadMat(src), loadMat(kernel), d_dst, ccorr);

    d_dst.download(dst);

    EXPECT_MAT_NEAR(dst, dst_gold, 1e-1);
}


INSTANTIATE_TEST_CASE_P(ImgProc, Convolve, Combine(
                        ALL_DEVICES, 
                        Values(3, 7, 11, 17, 19, 23, 45),
                        Bool()));

#endif // HAVE_CUDA
