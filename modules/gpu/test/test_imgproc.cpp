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

///////////////////////////////////////////////////////////////////////////////////////////////////////
// threshold

struct Threshold : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int threshOp;

    cv::Size size;
    cv::Mat src;
    double maxVal;
    double thresh;

    cv::Mat dst_gold;
    
    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());
        threshOp = std::tr1::get<2>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(20, 150), rng.uniform(20, 150));

        src = cvtest::randomMat(rng, size, type, 0.0, 127.0, false);

        maxVal = rng.uniform(20.0, 127.0);
        thresh = rng.uniform(0.0, maxVal);

        cv::threshold(src, dst_gold, thresh, maxVal, threshOp);
    }
};

TEST_P(Threshold, Accuracy)
{
    static const char* ops[] = {"THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_TRUNC", "THRESH_TOZERO", "THRESH_TOZERO_INV"};
    const char* threshOpStr = ops[threshOp];

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);
    PRINT_PARAM(threshOpStr);
    PRINT_PARAM(maxVal);
    PRINT_PARAM(thresh);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::threshold(cv::gpu::GpuMat(src), gpuRes, thresh, maxVal, threshOp);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, Threshold, testing::Combine(
                        testing::ValuesIn(devices()), 
                        testing::Values(CV_8U, CV_32F), 
                        testing::Values((int)cv::THRESH_BINARY, (int)cv::THRESH_BINARY_INV, (int)cv::THRESH_TRUNC, (int)cv::THRESH_TOZERO, (int)cv::THRESH_TOZERO_INV)));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// resize

struct Resize : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int interpolation;

    cv::Size size;
    cv::Mat src;

    cv::Mat dst_gold1;
    cv::Mat dst_gold2;
    
    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());
        interpolation = std::tr1::get<2>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(20, 150), rng.uniform(20, 150));

        src = cvtest::randomMat(rng, size, type, 0.0, 127.0, false);

        cv::resize(src, dst_gold1, cv::Size(), 2.0, 2.0, interpolation);
        cv::resize(src, dst_gold2, cv::Size(), 0.5, 0.5, interpolation);
    }
};

TEST_P(Resize, Accuracy)
{
    static const char* interpolations[] = {"INTER_NEAREST", "INTER_LINEAR"};
    const char* interpolationStr = interpolations[interpolation];

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);
    PRINT_PARAM(interpolationStr);

    cv::Mat dst1;
    cv::Mat dst2;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_src(src);
        cv::gpu::GpuMat gpuRes1;
        cv::gpu::GpuMat gpuRes2;

        cv::gpu::resize(dev_src, gpuRes1, cv::Size(), 2.0, 2.0, interpolation);
        cv::gpu::resize(dev_src, gpuRes2, cv::Size(), 0.5, 0.5, interpolation);

        gpuRes1.download(dst1);
        gpuRes2.download(dst2);
    );

    EXPECT_MAT_SIMILAR(dst_gold1, dst1, 0.5);
    EXPECT_MAT_SIMILAR(dst_gold2, dst2, 0.5);
}

INSTANTIATE_TEST_CASE_P(ImgProc, Resize, testing::Combine(
                        testing::ValuesIn(devices()), 
                        testing::Values(CV_8UC1, CV_8UC4), 
                        testing::Values((int)cv::INTER_NEAREST, (int)cv::INTER_LINEAR)));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// remap

struct Remap : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;

    cv::Size size;
    cv::Mat src;
    cv::Mat xmap;
    cv::Mat ymap;

    cv::Mat dst_gold;
    
    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(20, 150), rng.uniform(20, 150));

        src = cvtest::randomMat(rng, size, type, 0.0, 127.0, false);
        xmap = cvtest::randomMat(rng, size, CV_32FC1, 0.0, src.cols - 1, false);
        ymap = cvtest::randomMat(rng, size, CV_32FC1, 0.0, src.rows - 1, false);
        
        cv::remap(src, dst_gold, xmap, ymap, cv::INTER_LINEAR, cv::BORDER_WRAP);
    }
};

TEST_P(Remap, Accuracy)
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;
        
        cv::gpu::remap(cv::gpu::GpuMat(src), gpuRes, cv::gpu::GpuMat(xmap), cv::gpu::GpuMat(ymap));

        gpuRes.download(dst);
    );

    EXPECT_MAT_SIMILAR(dst_gold, dst, 0.5);
}

INSTANTIATE_TEST_CASE_P(ImgProc, Remap, testing::Combine(
                        testing::ValuesIn(devices()), 
                        testing::Values(CV_8UC1, CV_8UC3)));
                        
///////////////////////////////////////////////////////////////////////////////////////////////////////
// copyMakeBorder

struct CopyMakeBorder : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;

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
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(20, 150), rng.uniform(20, 150));

        src = cvtest::randomMat(rng, size, type, 0.0, 127.0, false);
        
        top = rng.uniform(1, 10);
        botton = rng.uniform(1, 10);
        left = rng.uniform(1, 10);
        right = rng.uniform(1, 10);
        val = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

        cv::copyMakeBorder(src, dst_gold, top, botton, left, right, cv::BORDER_CONSTANT, val);
    }
};

TEST_P(CopyMakeBorder, Accuracy)
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);
    PRINT_PARAM(top);
    PRINT_PARAM(botton);
    PRINT_PARAM(left);
    PRINT_PARAM(right);
    PRINT_PARAM(val);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::copyMakeBorder(cv::gpu::GpuMat(src), gpuRes, top, botton, left, right, val);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, CopyMakeBorder, testing::Combine(
                        testing::ValuesIn(devices()), 
                        testing::Values(CV_8UC1, CV_8UC4, CV_32SC1)));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// warpAffine & warpPerspective

static const int warpFlags[] = {cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC, cv::INTER_NEAREST | cv::WARP_INVERSE_MAP, cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::INTER_CUBIC | cv::WARP_INVERSE_MAP};
static const char* warpFlags_str[] = {"INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_NEAREST | WARP_INVERSE_MAP", "INTER_LINEAR | WARP_INVERSE_MAP", "INTER_CUBIC | WARP_INVERSE_MAP"};

struct WarpAffine : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int flagIdx;

    cv::Size size;
    cv::Mat src;
    cv::Mat M;

    cv::Mat dst_gold;
    
    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());
        flagIdx = std::tr1::get<2>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(20, 150), rng.uniform(20, 150));

        src = cvtest::randomMat(rng, size, type, 0.0, 127.0, false);

        static double reflect[2][3] = { {-1,  0, 0},
                                        { 0, -1, 0}};
        reflect[0][2] = size.width;
        reflect[1][2] = size.height;
        M = cv::Mat(2, 3, CV_64F, (void*)reflect); 

        cv::warpAffine(src, dst_gold, M, src.size(), warpFlags[flagIdx]);       
    }
};

TEST_P(WarpAffine, Accuracy)
{
    const char* warpFlagStr = warpFlags_str[flagIdx];

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);
    PRINT_PARAM(warpFlagStr);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::warpAffine(cv::gpu::GpuMat(src), gpuRes, M, src.size(), warpFlags[flagIdx]);

        gpuRes.download(dst);
    );

    // Check inner parts (ignoring 1 pixel width border)
    cv::Mat dst_gold_roi = dst_gold.rowRange(1, dst_gold.rows - 1).colRange(1, dst_gold.cols - 1);
    cv::Mat dst_roi = dst.rowRange(1, dst.rows - 1).colRange(1, dst.cols - 1);

    EXPECT_MAT_NEAR(dst_gold_roi, dst_roi, 1e-3);
}

struct WarpPerspective : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int flagIdx;

    cv::Size size;
    cv::Mat src;
    cv::Mat M;

    cv::Mat dst_gold;
    
    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());
        flagIdx = std::tr1::get<2>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(20, 150), rng.uniform(20, 150));

        src = cvtest::randomMat(rng, size, type, 0.0, 127.0, false);

        static double reflect[3][3] = { { -1, 0, 0},
                                        { 0, -1, 0},
                                        { 0,  0, 1}};
        reflect[0][2] = size.width;
        reflect[1][2] = size.height;
        M = cv::Mat(3, 3, CV_64F, (void*)reflect);

        cv::warpPerspective(src, dst_gold, M, src.size(), warpFlags[flagIdx]);       
    }
};

TEST_P(WarpPerspective, Accuracy)
{
    const char* warpFlagStr = warpFlags_str[flagIdx];

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);
    PRINT_PARAM(warpFlagStr);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::warpPerspective(cv::gpu::GpuMat(src), gpuRes, M, src.size(), warpFlags[flagIdx]);

        gpuRes.download(dst);
    );

    // Check inner parts (ignoring 1 pixel width border)
    cv::Mat dst_gold_roi = dst_gold.rowRange(1, dst_gold.rows - 1).colRange(1, dst_gold.cols - 1);
    cv::Mat dst_roi = dst.rowRange(1, dst.rows - 1).colRange(1, dst.cols - 1);

    EXPECT_MAT_NEAR(dst_gold_roi, dst_roi, 1e-3);
}

INSTANTIATE_TEST_CASE_P(ImgProc, WarpAffine, testing::Combine(
                        testing::ValuesIn(devices()), 
                        testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                        testing::Range(0, 6)));

INSTANTIATE_TEST_CASE_P(ImgProc, WarpPerspective, testing::Combine(
                        testing::ValuesIn(devices()), 
                        testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4),
                        testing::Range(0, 6)));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// integral

struct Integral : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Size size;
    cv::Mat src;

    cv::Mat dst_gold;
    
    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(20, 150), rng.uniform(20, 150));

        src = cvtest::randomMat(rng, size, CV_8UC1, 0.0, 255.0, false); 

        cv::integral(src, dst_gold, CV_32S);     
    }
};

TEST_P(Integral, Accuracy)
{
    PRINT_PARAM(devInfo);
    PRINT_PARAM(size);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::integral(cv::gpu::GpuMat(src), gpuRes);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, Integral, testing::ValuesIn(devices()));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// cvtColor

struct CvtColor : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    
    cv::Mat img;
    
    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());
        
        cv::Mat imgBase = readImage("stereobm/aloe-L.png");
        ASSERT_FALSE(imgBase.empty());

        imgBase.convertTo(img, type, type == CV_32F ? 1.0 / 255.0 : 1.0);
    }
};

TEST_P(CvtColor, BGR2RGB)
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);

    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, CV_BGR2RGB);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::cvtColor(cv::gpu::GpuMat(src), gpuRes, CV_BGR2RGB);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR2RGBA)
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);

    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, CV_BGR2RGBA);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::cvtColor(cv::gpu::GpuMat(src), gpuRes, CV_BGR2RGBA);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGRA2RGB)
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);

    cv::Mat src;
    cv::cvtColor(img, src, CV_BGR2BGRA);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, CV_BGRA2RGB);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::cvtColor(cv::gpu::GpuMat(src), gpuRes, CV_BGRA2RGB);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST_P(CvtColor, BGR2YCrCb)
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);

    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, CV_BGR2YCrCb);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::cvtColor(cv::gpu::GpuMat(src), gpuRes, CV_BGR2YCrCb);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, YCrCb2RGB)
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);

    cv::Mat src;
    cv::cvtColor(img, src, CV_BGR2YCrCb);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, CV_YCrCb2RGB);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::cvtColor(cv::gpu::GpuMat(src), gpuRes, CV_YCrCb2RGB);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, BGR2YUV)
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);

    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, CV_BGR2YUV);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::cvtColor(cv::gpu::GpuMat(src), gpuRes, CV_BGR2YUV);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, YUV2BGR)
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);

    cv::Mat src;
    cv::cvtColor(img, src, CV_BGR2YUV);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, CV_YUV2BGR);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::cvtColor(cv::gpu::GpuMat(src), gpuRes, CV_YUV2BGR);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, BGR2XYZ)
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);

    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, CV_BGR2XYZ);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::cvtColor(cv::gpu::GpuMat(src), gpuRes, CV_BGR2XYZ);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, XYZ2BGR)
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);

    cv::Mat src;
    cv::cvtColor(img, src, CV_BGR2XYZ);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, CV_XYZ2BGR);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::cvtColor(cv::gpu::GpuMat(src), gpuRes, CV_XYZ2BGR);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, BGR2HSV)
{
    if (type == CV_16U)
        return;

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);

    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, CV_BGR2HSV);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::cvtColor(cv::gpu::GpuMat(src), gpuRes, CV_BGR2HSV);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HSV2BGR)
{
    if (type == CV_16U)
        return;

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);

    cv::Mat src;
    cv::cvtColor(img, src, CV_BGR2HSV);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, CV_HSV2BGR);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::cvtColor(cv::gpu::GpuMat(src), gpuRes, CV_HSV2BGR);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, BGR2HSV_FULL)
{
    if (type == CV_16U)
        return;

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);

    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, CV_BGR2HSV_FULL);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::cvtColor(cv::gpu::GpuMat(src), gpuRes, CV_BGR2HSV_FULL);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HSV2BGR_FULL)
{
    if (type == CV_16U)
        return;

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);

    cv::Mat src;
    cv::cvtColor(img, src, CV_BGR2HSV_FULL);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, CV_HSV2BGR_FULL);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::cvtColor(cv::gpu::GpuMat(src), gpuRes, CV_HSV2BGR_FULL);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, BGR2HLS)
{
    if (type == CV_16U)
        return;

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);

    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, CV_BGR2HLS);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::cvtColor(cv::gpu::GpuMat(src), gpuRes, CV_BGR2HLS);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HLS2BGR)
{
    if (type == CV_16U)
        return;

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);

    cv::Mat src;
    cv::cvtColor(img, src, CV_BGR2HLS);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, CV_HLS2BGR);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::cvtColor(cv::gpu::GpuMat(src), gpuRes, CV_HLS2BGR);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, BGR2HLS_FULL)
{
    if (type == CV_16U)
        return;

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);

    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, CV_BGR2HLS_FULL);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::cvtColor(cv::gpu::GpuMat(src), gpuRes, CV_BGR2HLS_FULL);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, HLS2BGR_FULL)
{
    if (type == CV_16U)
        return;

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);

    cv::Mat src;
    cv::cvtColor(img, src, CV_BGR2HLS_FULL);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, CV_HLS2BGR_FULL);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::cvtColor(cv::gpu::GpuMat(src), gpuRes, CV_HLS2BGR_FULL);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, type == CV_32F ? 1e-2 : 1);
}

TEST_P(CvtColor, BGR2GRAY)
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);

    cv::Mat src = img;
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, CV_BGR2GRAY);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::cvtColor(cv::gpu::GpuMat(src), gpuRes, CV_BGR2GRAY);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-5);
}

TEST_P(CvtColor, GRAY2RGB)
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);

    cv::Mat src;
    cv::cvtColor(img, src, CV_BGR2GRAY);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, CV_GRAY2RGB);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuRes;

        cv::gpu::cvtColor(cv::gpu::GpuMat(src), gpuRes, CV_GRAY2RGB);

        gpuRes.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, CvtColor, testing::Combine(
                        testing::ValuesIn(devices()), 
                        testing::Values(CV_8U, CV_16U, CV_32F)));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// histograms

struct HistEven : testing::TestWithParam<cv::gpu::DeviceInfo>
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
    ASSERT_TRUE(!hsv.empty());

    PRINT_PARAM(devInfo);

    cv::Mat hist;
    
    ASSERT_NO_THROW(
        std::vector<cv::gpu::GpuMat> srcs;
        cv::gpu::split(cv::gpu::GpuMat(hsv), srcs);

        cv::gpu::GpuMat gpuHist;

        cv::gpu::histEven(srcs[0], gpuHist, hbins, (int)hranges[0], (int)hranges[1]);

        gpuHist.download(hist);
    );

    EXPECT_MAT_NEAR(hist_gold, hist, 0.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, HistEven, testing::ValuesIn(devices()));

struct CalcHist : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Size size;
    cv::Mat src;
    cv::Mat hist_gold;
    
    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));
        
        src = cvtest::randomMat(rng, size, CV_8UC1, 0, 255, false);

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
    PRINT_PARAM(devInfo);
    PRINT_PARAM(size);

    cv::Mat hist;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuHist;

        cv::gpu::calcHist(cv::gpu::GpuMat(src), gpuHist);

        gpuHist.download(hist);
    );

    EXPECT_MAT_NEAR(hist_gold, hist, 0.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, CalcHist, testing::ValuesIn(devices()));

struct EqualizeHist : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Size size;
    cv::Mat src;
    cv::Mat dst_gold;
    
    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 200), rng.uniform(100, 200));
        
        src = cvtest::randomMat(rng, size, CV_8UC1, 0, 255, false);

        cv::equalizeHist(src, dst_gold);
    }
};

TEST_P(EqualizeHist, Accuracy)
{
    PRINT_PARAM(devInfo);
    PRINT_PARAM(size);

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpuDst;

        cv::gpu::equalizeHist(cv::gpu::GpuMat(src), gpuDst);

        gpuDst.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 3.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, EqualizeHist, testing::ValuesIn(devices()));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// cornerHarris

static const int borderTypes[] = {cv::BORDER_REPLICATE, cv::BORDER_CONSTANT, cv::BORDER_REFLECT, cv::BORDER_WRAP, cv::BORDER_REFLECT101, cv::BORDER_TRANSPARENT};
static const char* borderTypes_str[] = {"BORDER_REPLICATE", "BORDER_CONSTANT", "BORDER_REFLECT", "BORDER_WRAP", "BORDER_REFLECT101", "BORDER_TRANSPARENT"};

struct CornerHarris : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int borderTypeIdx;

    cv::Mat src;
    int blockSize;
    int apertureSize;        
    double k;

    cv::Mat dst_gold;
    
    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());
        borderTypeIdx = std::tr1::get<2>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());
    
        cv::RNG& rng = cvtest::TS::ptr()->get_rng();
        
        cv::Mat img = readImage("stereobm/aloe-L.png", CV_LOAD_IMAGE_GRAYSCALE);
        ASSERT_FALSE(img.empty());
        
        img.convertTo(src, type, type == CV_32F ? 1.0 / 255.0 : 1.0);
        
        blockSize = 1 + rng.next() % 5;
        apertureSize = 1 + 2 * (rng.next() % 4);        
        k = rng.uniform(0.1, 0.9);

        cv::cornerHarris(src, dst_gold, blockSize, apertureSize, k, borderTypes[borderTypeIdx]);
    }
};

TEST_P(CornerHarris, Accuracy)
{
    const char* borderTypeStr = borderTypes_str[borderTypeIdx];
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(borderTypeStr);
    PRINT_PARAM(blockSize);
    PRINT_PARAM(apertureSize);
    PRINT_PARAM(k);

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst;
        cv::gpu::cornerHarris(cv::gpu::GpuMat(src), dev_dst, blockSize, apertureSize, k, borderTypes[borderTypeIdx]);
        dev_dst.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-3);
}

INSTANTIATE_TEST_CASE_P(ImgProc, CornerHarris, testing::Combine(
                        testing::ValuesIn(devices()), 
                        testing::Values(CV_8UC1, CV_32FC1), 
                        testing::Values(0, 4)));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// cornerMinEigen

struct CornerMinEigen : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int borderTypeIdx;

    cv::Mat src;
    int blockSize;
    int apertureSize;

    cv::Mat dst_gold;
    
    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());
        borderTypeIdx = std::tr1::get<2>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());        
    
        cv::RNG& rng = cvtest::TS::ptr()->get_rng();
        
        cv::Mat img = readImage("stereobm/aloe-L.png", CV_LOAD_IMAGE_GRAYSCALE);
        ASSERT_FALSE(img.empty());

        img.convertTo(src, type, type == CV_32F ? 1.0 / 255.0 : 1.0);
        
        blockSize = 1 + rng.next() % 5;
        apertureSize = 1 + 2 * (rng.next() % 4);

        cv::cornerMinEigenVal(src, dst_gold, blockSize, apertureSize, borderTypes[borderTypeIdx]);
    }
};

TEST_P(CornerMinEigen, Accuracy)
{
    const char* borderTypeStr = borderTypes_str[borderTypeIdx];
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(borderTypeStr);
    PRINT_PARAM(blockSize);
    PRINT_PARAM(apertureSize);

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst;
        cv::gpu::cornerMinEigenVal(cv::gpu::GpuMat(src), dev_dst, blockSize, apertureSize, borderTypes[borderTypeIdx]);
        dev_dst.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-2);
}

INSTANTIATE_TEST_CASE_P(ImgProc, CornerMinEigen, testing::Combine(
                        testing::ValuesIn(devices()), 
                        testing::Values(CV_8UC1, CV_32FC1), 
                        testing::Values(0, 4)));

////////////////////////////////////////////////////////////////////////
// ColumnSum

struct ColumnSum : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Size size;
    cv::Mat src;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());
    
        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 400), rng.uniform(100, 400));

        src = cvtest::randomMat(rng, size, CV_32F, 0.0, 1.0, false);
    }
};

TEST_P(ColumnSum, Accuracy)
{
    PRINT_PARAM(devInfo);
    PRINT_PARAM(size);

    cv::Mat dst;
    
    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst;
        cv::gpu::columnSum(cv::gpu::GpuMat(src), dev_dst);
        dev_dst.download(dst);
    );

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

INSTANTIATE_TEST_CASE_P(ImgProc, ColumnSum, testing::ValuesIn(devices()));

////////////////////////////////////////////////////////////////////////
// Norm

static const int normTypes[] = {cv::NORM_INF, cv::NORM_L1, cv::NORM_L2};
static const char* normTypes_str[] = {"NORM_INF", "NORM_L1", "NORM_L2"};

struct Norm : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int normTypeIdx;

    cv::Size size;
    cv::Mat src;

    double gold;

    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        type = std::tr1::get<1>(GetParam());
        normTypeIdx = std::tr1::get<2>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());
    
        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 400), rng.uniform(100, 400));

        src = cvtest::randomMat(rng, size, type, 0.0, 10.0, false);

        gold = cv::norm(src, normTypes[normTypeIdx]);
    }
};

TEST_P(Norm, Accuracy)
{
    const char* normTypeStr = normTypes_str[normTypeIdx];

    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);
    PRINT_PARAM(normTypeStr);

    double res;

    ASSERT_NO_THROW(
        res = cv::gpu::norm(cv::gpu::GpuMat(src), normTypes[normTypeIdx]);
    );

    ASSERT_NEAR(res, gold, 0.5);
}

INSTANTIATE_TEST_CASE_P(ImgProc, Norm, testing::Combine(
                        testing::ValuesIn(devices()), 
                        testing::ValuesIn(types(CV_8U, CV_32F, 1, 1)),
                        testing::Range(0, 3)));

////////////////////////////////////////////////////////////////////////////////
// reprojectImageTo3D

struct ReprojectImageTo3D : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Size size;
    cv::Mat disp;
    cv::Mat Q;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());
    
        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(rng.uniform(100, 500), rng.uniform(100, 500));

        disp = cvtest::randomMat(rng, size, CV_8UC1, 5.0, 30.0, false);

        Q = cvtest::randomMat(rng, cv::Size(4, 4), CV_32FC1, 0.1, 1.0, false);

        cv::reprojectImageTo3D(disp, dst_gold, Q, false);
    }
};

TEST_P(ReprojectImageTo3D, Accuracy)
{
    PRINT_PARAM(devInfo);
    PRINT_PARAM(size);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat gpures;
        cv::gpu::reprojectImageTo3D(cv::gpu::GpuMat(disp), gpures, Q);
        gpures.download(dst);
    );

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

INSTANTIATE_TEST_CASE_P(ImgProc, ReprojectImageTo3D, testing::ValuesIn(devices()));

////////////////////////////////////////////////////////////////////////////////
// meanShift

struct MeanShift : testing::TestWithParam<cv::gpu::DeviceInfo>
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

    PRINT_PARAM(devInfo);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst;
        cv::gpu::meanShiftFiltering(cv::gpu::GpuMat(rgba), dev_dst, spatialRad, colorRad);
        dev_dst.download(dst);
    );

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

    PRINT_PARAM(devInfo);

    cv::Mat rmap_filtered;
    cv::Mat rmap;
    cv::Mat spmap;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat d_rmap_filtered;
        cv::gpu::meanShiftFiltering(cv::gpu::GpuMat(rgba), d_rmap_filtered, spatialRad, colorRad);

        cv::gpu::GpuMat d_rmap;
        cv::gpu::GpuMat d_spmap;
        cv::gpu::meanShiftProc(cv::gpu::GpuMat(rgba), d_rmap, d_spmap, spatialRad, colorRad);

        d_rmap_filtered.download(rmap_filtered);
        d_rmap.download(rmap);
        d_spmap.download(spmap);
    );

    ASSERT_EQ(CV_8UC4, rmap.type());
    
    EXPECT_MAT_NEAR(rmap_filtered, rmap, 0.0);    
    EXPECT_MAT_NEAR(spmap_template, spmap, 0.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, MeanShift, testing::ValuesIn(devices()));

struct MeanShiftSegmentation : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int minsize;
    
    cv::Mat rgba;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        minsize = std::tr1::get<1>(GetParam());

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
    PRINT_PARAM(devInfo);
    PRINT_PARAM(minsize);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::meanShiftSegmentation(cv::gpu::GpuMat(rgba), dst, 10, 10, minsize);
    );

    cv::Mat dst_rgb;
    cv::cvtColor(dst, dst_rgb, CV_BGRA2BGR);

    EXPECT_MAT_SIMILAR(dst_gold, dst_rgb, 1e-3);
}

INSTANTIATE_TEST_CASE_P(ImgProc, MeanShiftSegmentation, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(0, 4, 20, 84, 340, 1364)));

////////////////////////////////////////////////////////////////////////////////
// matchTemplate

static const char* matchTemplateMethods[] = {"SQDIFF", "SQDIFF_NORMED", "CCORR", "CCORR_NORMED", "CCOEFF", "CCOEFF_NORMED"};

struct MatchTemplate8U : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int cn;
    int method;

    int n, m, h, w;
    cv::Mat image, templ;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        cn = std::tr1::get<1>(GetParam());
        method = std::tr1::get<2>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        n = rng.uniform(30, 100);
        m = rng.uniform(30, 100);
        h = rng.uniform(5, n - 1);
        w = rng.uniform(5, m - 1);

        image = cvtest::randomMat(rng, cv::Size(m, n), CV_MAKETYPE(CV_8U, cn), 1.0, 10.0, false);
        templ = cvtest::randomMat(rng, cv::Size(w, h), CV_MAKETYPE(CV_8U, cn), 1.0, 10.0, false);

        cv::matchTemplate(image, templ, dst_gold, method);
    }
};

TEST_P(MatchTemplate8U, Regression)
{
    const char* matchTemplateMethodStr = matchTemplateMethods[method];
    PRINT_PARAM(devInfo);
    PRINT_PARAM(cn);
    PRINT_PARAM(matchTemplateMethodStr);
    PRINT_PARAM(n);
    PRINT_PARAM(m);
    PRINT_PARAM(h);
    PRINT_PARAM(w);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst;
        cv::gpu::matchTemplate(cv::gpu::GpuMat(image), cv::gpu::GpuMat(templ), dev_dst, method);
        dev_dst.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 5 * h * w * 1e-4);
}

INSTANTIATE_TEST_CASE_P(ImgProc, MatchTemplate8U, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Range(1, 5), 
                        testing::Values((int)CV_TM_SQDIFF, (int)CV_TM_SQDIFF_NORMED, (int)CV_TM_CCORR, (int)CV_TM_CCORR_NORMED, (int)CV_TM_CCOEFF, (int)CV_TM_CCOEFF_NORMED)));

struct MatchTemplate32F : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int cn;
    int method;

    int n, m, h, w;
    cv::Mat image, templ;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        cn = std::tr1::get<1>(GetParam());
        method = std::tr1::get<2>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        n = rng.uniform(30, 100);
        m = rng.uniform(30, 100);
        h = rng.uniform(5, n - 1);
        w = rng.uniform(5, m - 1);

        image = cvtest::randomMat(rng, cv::Size(m, n), CV_MAKETYPE(CV_32F, cn), 0.001, 1.0, false);
        templ = cvtest::randomMat(rng, cv::Size(w, h), CV_MAKETYPE(CV_32F, cn), 0.001, 1.0, false);

        cv::matchTemplate(image, templ, dst_gold, method);
    }
};

TEST_P(MatchTemplate32F, Regression)
{
    const char* matchTemplateMethodStr = matchTemplateMethods[method];
    PRINT_PARAM(devInfo);
    PRINT_PARAM(cn);
    PRINT_PARAM(matchTemplateMethodStr);
    PRINT_PARAM(n);
    PRINT_PARAM(m);
    PRINT_PARAM(h);
    PRINT_PARAM(w);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst;
        cv::gpu::matchTemplate(cv::gpu::GpuMat(image), cv::gpu::GpuMat(templ), dev_dst, method);
        dev_dst.download(dst);
    );

    EXPECT_MAT_NEAR(dst_gold, dst, 0.25 * h * w * 1e-4);
}

INSTANTIATE_TEST_CASE_P(ImgProc, MatchTemplate32F, testing::Combine(
                        testing::ValuesIn(devices()), 
                        testing::Range(1, 5), 
                        testing::Values((int)CV_TM_SQDIFF, (int)CV_TM_CCORR)));

struct MatchTemplate : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::Mat image;
    cv::Mat pattern;

    cv::Point maxLocGold;

    cv::gpu::DeviceInfo devInfo;
    int method;

    virtual void SetUp()
    {
        devInfo = std::tr1::get<0>(GetParam());
        method = std::tr1::get<1>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());
        
        image = readImage("matchtemplate/black.png");
        ASSERT_FALSE(image.empty());
        
        pattern = readImage("matchtemplate/cat.png");
        ASSERT_FALSE(pattern.empty());

        maxLocGold = cv::Point(284, 12);
    }
};

TEST_P(MatchTemplate, FindPatternInBlack)
{
    const char* matchTemplateMethodStr = matchTemplateMethods[method];

    PRINT_PARAM(devInfo);
    PRINT_PARAM(matchTemplateMethodStr);

    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_dst;
        cv::gpu::matchTemplate(cv::gpu::GpuMat(image), cv::gpu::GpuMat(pattern), dev_dst, method);
        dev_dst.download(dst);
    );

    double maxValue;
    cv::Point maxLoc;
    cv::minMaxLoc(dst, NULL, &maxValue, NULL, &maxLoc);

    ASSERT_EQ(maxLocGold, maxLoc);
}

INSTANTIATE_TEST_CASE_P(ImgProc, MatchTemplate, testing::Combine(
                        testing::ValuesIn(devices()), 
                        testing::Values((int)CV_TM_CCOEFF_NORMED, (int)CV_TM_CCORR_NORMED)));

////////////////////////////////////////////////////////////////////////////
// MulSpectrums

struct MulSpectrums : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int flag;

    cv::Mat a, b; 

    virtual void SetUp() 
    {
        devInfo = std::tr1::get<0>(GetParam());
        flag = std::tr1::get<1>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        a = cvtest::randomMat(rng, cv::Size(rng.uniform(100, 200), rng.uniform(100, 200)), CV_32FC2, 0.0, 10.0, false);
        b = cvtest::randomMat(rng, a.size(), CV_32FC2, 0.0, 10.0, false);
    }
};

TEST_P(MulSpectrums, Simple)
{
    PRINT_PARAM(devInfo);
    PRINT_PARAM(flag);

    cv::Mat c_gold;
    cv::mulSpectrums(a, b, c_gold, flag, false);
    
    cv::Mat c;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat d_c;

        cv::gpu::mulSpectrums(cv::gpu::GpuMat(a), cv::gpu::GpuMat(b), d_c, flag, false);

        d_c.download(c);
    );

    EXPECT_MAT_NEAR(c_gold, c, 1e-4);
}

TEST_P(MulSpectrums, Scaled)
{
    PRINT_PARAM(devInfo);
    PRINT_PARAM(flag);

    float scale = 1.f / a.size().area();

    cv::Mat c_gold;
    cv::mulSpectrums(a, b, c_gold, flag, false);
    c_gold.convertTo(c_gold, c_gold.type(), scale);

    cv::Mat c;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat d_c;

        cv::gpu::mulAndScaleSpectrums(cv::gpu::GpuMat(a), cv::gpu::GpuMat(b), d_c, flag, scale, false);

        d_c.download(c);
    );

    EXPECT_MAT_NEAR(c_gold, c, 1e-4);
}

INSTANTIATE_TEST_CASE_P(ImgProc, MulSpectrums, testing::Combine(
                        testing::ValuesIn(devices()), 
                        testing::Values(0, (int)cv::DFT_ROWS)));

////////////////////////////////////////////////////////////////////////////
// Dft

struct Dft : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    virtual void SetUp() 
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());
    }
};


static void testC2C(const std::string& hint, int cols, int rows, int flags, bool inplace)
{
    PRINT_PARAM(hint);
    PRINT_PARAM(cols);
    PRINT_PARAM(rows);
    PRINT_PARAM(flags);
    PRINT_PARAM(inplace);

    cv::RNG& rng = cvtest::TS::ptr()->get_rng();

    cv::Mat a = cvtest::randomMat(rng, cv::Size(cols, rows), CV_32FC2, 0.0, 10.0, false);

    cv::Mat b_gold;
    cv::dft(a, b_gold, flags);

    cv::gpu::GpuMat d_b;
    cv::gpu::GpuMat d_b_data;
    if (inplace)
    {
        d_b_data.create(1, a.size().area(), CV_32FC2);
        d_b = cv::gpu::GpuMat(a.rows, a.cols, CV_32FC2, d_b_data.ptr(), a.cols * d_b_data.elemSize());
    }
    cv::gpu::dft(cv::gpu::GpuMat(a), d_b, cv::Size(cols, rows), flags);

    EXPECT_TRUE(!inplace || d_b.ptr() == d_b_data.ptr());
    ASSERT_EQ(CV_32F, d_b.depth());
    ASSERT_EQ(2, d_b.channels());
    EXPECT_MAT_NEAR(b_gold, d_b, rows * cols * 1e-4);
}

TEST_P(Dft, C2C)
{
    PRINT_PARAM(devInfo);

    cv::RNG& rng = cvtest::TS::ptr()->get_rng();

    int cols = 2 + rng.next() % 100, rows = 2 + rng.next() % 100;

    ASSERT_NO_THROW(
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
    );
}

static void testR2CThenC2R(const std::string& hint, int cols, int rows, bool inplace)
{
    PRINT_PARAM(hint);
    PRINT_PARAM(cols);
    PRINT_PARAM(rows);
    PRINT_PARAM(inplace);
    
    cv::RNG& rng = cvtest::TS::ptr()->get_rng();

    cv::Mat a = cvtest::randomMat(rng, cv::Size(cols, rows), CV_32FC1, 0.0, 10.0, false);

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

    cv::gpu::dft(cv::gpu::GpuMat(a), d_b, cv::Size(cols, rows), 0);
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
    PRINT_PARAM(devInfo);

    cv::RNG& rng = cvtest::TS::ptr()->get_rng();

    int cols = 2 + rng.next() % 100, rows = 2 + rng.next() % 100;

    ASSERT_NO_THROW(
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
    );
}

INSTANTIATE_TEST_CASE_P(ImgProc, Dft, testing::ValuesIn(devices()));

////////////////////////////////////////////////////////////////////////////
// blend

template <typename T> static void blendLinearGold(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& weights1, const cv::Mat& weights2, cv::Mat& result_gold)
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

struct Blend : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int, int> >
{
    cv::gpu::DeviceInfo devInfo;
    int depth;
    int cn;

    int type;
    cv::Size size;
    cv::Mat img1;
    cv::Mat img2;
    cv::Mat weights1;
    cv::Mat weights2;

    cv::Mat result_gold;

    virtual void SetUp() 
    {
        devInfo = std::tr1::get<0>(GetParam());
        depth = std::tr1::get<1>(GetParam());
        cn = std::tr1::get<2>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

        type = CV_MAKETYPE(depth, cn);

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        size = cv::Size(200 + cvtest::randInt(rng) % 1000, 200 + cvtest::randInt(rng) % 1000);

        img1 = cvtest::randomMat(rng, size, type, 0.0, depth == CV_8U ? 255.0 : 1.0, false);
        img2 = cvtest::randomMat(rng, size, type, 0.0, depth == CV_8U ? 255.0 : 1.0, false);
        weights1 = cvtest::randomMat(rng, size, CV_32F, 0, 1, false);
        weights2 = cvtest::randomMat(rng, size, CV_32F, 0, 1, false);
        
        if (depth == CV_8U)
            blendLinearGold<uchar>(img1, img2, weights1, weights2, result_gold);
        else
            blendLinearGold<float>(img1, img2, weights1, weights2, result_gold);
    }
};

TEST_P(Blend, Accuracy)
{
    PRINT_PARAM(devInfo);
    PRINT_TYPE(type);
    PRINT_PARAM(size);

    cv::Mat result;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat d_result;

        cv::gpu::blendLinear(cv::gpu::GpuMat(img1), cv::gpu::GpuMat(img2), cv::gpu::GpuMat(weights1), cv::gpu::GpuMat(weights2), d_result);

        d_result.download(result);
    );

    EXPECT_MAT_NEAR(result_gold, result, depth == CV_8U ? 1.0 : 1e-5);
}

INSTANTIATE_TEST_CASE_P(ImgProc, Blend, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(CV_8U, CV_32F),
                        testing::Range(1, 5)));

////////////////////////////////////////////////////////
// pyrDown

struct PyrDown : testing::TestWithParam<cv::gpu::DeviceInfo>
{    
    cv::gpu::DeviceInfo devInfo;
    
    cv::Mat src;
    
    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GetParam();
        cv::gpu::setDevice(devInfo.deviceID());
        
        cv::Mat img = readImage("stereobm/aloe-L.png");
        ASSERT_FALSE(img.empty());
        
        img.convertTo(src, CV_16S);
        
        cv::pyrDown(src, dst_gold);
    }
};

TEST_P(PyrDown, Accuracy)
{
    PRINT_PARAM(devInfo);
    
    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat d_dst;
        
        cv::gpu::pyrDown(cv::gpu::GpuMat(src), d_dst);
        
        d_dst.download(dst);
    );

    ASSERT_EQ(dst_gold.cols, dst.cols);
    ASSERT_EQ(dst_gold.rows, dst.rows);
    ASSERT_EQ(dst_gold.type(), dst.type());

    double err = cvtest::crossCorr(dst_gold, dst) /
            (cv::norm(dst_gold,cv::NORM_L2)*cv::norm(dst,cv::NORM_L2));
    ASSERT_NEAR(err, 1., 1e-2);
}

INSTANTIATE_TEST_CASE_P(ImgProc, PyrDown, testing::ValuesIn(devices()));

////////////////////////////////////////////////////////
// pyrUp

struct PyrUp: testing::TestWithParam<cv::gpu::DeviceInfo>
{    
    cv::gpu::DeviceInfo devInfo;
    
    cv::Mat src;
    
    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GetParam();
        cv::gpu::setDevice(devInfo.deviceID());
        
        cv::Mat img = readImage("stereobm/aloe-L.png");
        ASSERT_FALSE(img.empty());
        
        img.convertTo(src, CV_16S);
        
        cv::pyrUp(src, dst_gold);
    }
};

TEST_P(PyrUp, Accuracy)
{
    PRINT_PARAM(devInfo);
    
    cv::Mat dst;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat d_dst;
        
        cv::gpu::pyrUp(cv::gpu::GpuMat(src), d_dst);
        
        d_dst.download(dst);
    );

    ASSERT_EQ(dst_gold.cols, dst.cols);
    ASSERT_EQ(dst_gold.rows, dst.rows);
    ASSERT_EQ(dst_gold.type(), dst.type());

    double err = cvtest::crossCorr(dst_gold, dst) /
            (cv::norm(dst_gold,cv::NORM_L2)*cv::norm(dst,cv::NORM_L2));
    ASSERT_NEAR(err, 1., 1e-2);
}

INSTANTIATE_TEST_CASE_P(ImgProc, PyrUp, testing::ValuesIn(devices()));

////////////////////////////////////////////////////////
// Canny

struct Canny : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, int, bool> >
{
    cv::gpu::DeviceInfo devInfo;
    int apperture_size;
    bool L2gradient;
    
    cv::Mat img;

    double low_thresh;
    double high_thresh;

    cv::Mat edges_gold;

    virtual void SetUp() 
    {
        devInfo = std::tr1::get<0>(GetParam());
        apperture_size = std::tr1::get<1>(GetParam());
        L2gradient = std::tr1::get<2>(GetParam());

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
    PRINT_PARAM(devInfo);
    PRINT_PARAM(apperture_size);
    PRINT_PARAM(L2gradient);

    cv::Mat edges;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat d_edges;

        cv::gpu::Canny(cv::gpu::GpuMat(img), d_edges, low_thresh, high_thresh, apperture_size, L2gradient);

        d_edges.download(edges);
    );

    EXPECT_MAT_SIMILAR(edges_gold, edges, 1.0);
}

INSTANTIATE_TEST_CASE_P(ImgProc, Canny, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(3, 5),
                        testing::Values(false, true)));

#endif // HAVE_CUDA
