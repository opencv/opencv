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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

///////////////////////////////////////////////////////////////////////////////////////////////////////
// HistEven

struct HistEven : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(HistEven, Accuracy)
{
    cv::Mat img = readImage("stereobm/aloe-L.png");
    ASSERT_FALSE(img.empty());

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    int hbins = 30;
    float hranges[] = {0.0f, 180.0f};

    std::vector<cv::gpu::GpuMat> srcs;
    cv::gpu::split(loadMat(hsv), srcs);

    cv::gpu::GpuMat hist;
    cv::gpu::histEven(srcs[0], hist, hbins, (int)hranges[0], (int)hranges[1]);

    cv::MatND histnd;
    int histSize[] = {hbins};
    const float* ranges[] = {hranges};
    int channels[] = {0};
    cv::calcHist(&hsv, 1, channels, cv::Mat(), histnd, 1, histSize, ranges);

    cv::Mat hist_gold = histnd;
    hist_gold = hist_gold.t();
    hist_gold.convertTo(hist_gold, CV_32S);

    EXPECT_MAT_NEAR(hist_gold, hist, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, HistEven, ALL_DEVICES);

///////////////////////////////////////////////////////////////////////////////////////////////////////
// CalcHist

namespace
{
    void calcHistGold(const cv::Mat& src, cv::Mat& hist)
    {
        hist.create(1, 256, CV_32SC1);
        hist.setTo(cv::Scalar::all(0));

        int* hist_row = hist.ptr<int>();
        for (int y = 0; y < src.rows; ++y)
        {
            const uchar* src_row = src.ptr(y);

            for (int x = 0; x < src.cols; ++x)
                ++hist_row[src_row[x]];
        }
    }
}

PARAM_TEST_CASE(CalcHist, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo;

    cv::Size size;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(CalcHist, Accuracy)
{
    cv::Mat src = randomMat(size, CV_8UC1);

    cv::gpu::GpuMat hist;
    cv::gpu::calcHist(loadMat(src), hist);

    cv::Mat hist_gold;
    calcHistGold(src, hist_gold);

    EXPECT_MAT_NEAR(hist_gold, hist, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, CalcHist, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// EqualizeHist

PARAM_TEST_CASE(EqualizeHist, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(EqualizeHist, Accuracy)
{
    cv::Mat src = randomMat(size, CV_8UC1);

    cv::gpu::GpuMat dst;
    cv::gpu::equalizeHist(loadMat(src), dst);

    cv::Mat dst_gold;
    cv::equalizeHist(src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 3.0);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, EqualizeHist, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// CLAHE

namespace
{
    IMPLEMENT_PARAM_CLASS(ClipLimit, double)
}

PARAM_TEST_CASE(CLAHE, cv::gpu::DeviceInfo, cv::Size, ClipLimit)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    double clipLimit;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        clipLimit = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(CLAHE, Accuracy)
{
    cv::Mat src = randomMat(size, CV_8UC1);

    cv::Ptr<cv::gpu::CLAHE> clahe = cv::gpu::createCLAHE(clipLimit);
    cv::gpu::GpuMat dst;
    clahe->apply(loadMat(src), dst);

    cv::Ptr<cv::CLAHE> clahe_gold = cv::createCLAHE(clipLimit);
    cv::Mat dst_gold;
    clahe_gold->apply(src, dst_gold);

    ASSERT_MAT_NEAR(dst_gold, dst, 1.0);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, CLAHE, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(0.0, 40.0)));

////////////////////////////////////////////////////////
// Canny

namespace
{
    IMPLEMENT_PARAM_CLASS(AppertureSize, int);
    IMPLEMENT_PARAM_CLASS(L2gradient, bool);
}

PARAM_TEST_CASE(Canny, cv::gpu::DeviceInfo, AppertureSize, L2gradient, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int apperture_size;
    bool useL2gradient;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        apperture_size = GET_PARAM(1);
        useL2gradient = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(Canny, Accuracy)
{
    cv::Mat img = readImage("stereobm/aloe-L.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    double low_thresh = 50.0;
    double high_thresh = 100.0;

    if (!supportFeature(devInfo, cv::gpu::SHARED_ATOMICS))
    {
        try
        {
        cv::gpu::GpuMat edges;
        cv::gpu::Canny(loadMat(img), edges, low_thresh, high_thresh, apperture_size, useL2gradient);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsNotImplemented, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat edges;
        cv::gpu::Canny(loadMat(img, useRoi), edges, low_thresh, high_thresh, apperture_size, useL2gradient);

        cv::Mat edges_gold;
        cv::Canny(img, edges_gold, low_thresh, high_thresh, apperture_size, useL2gradient);

        EXPECT_MAT_SIMILAR(edges_gold, edges, 2e-2);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, Canny, testing::Combine(
    ALL_DEVICES,
    testing::Values(AppertureSize(3), AppertureSize(5)),
    testing::Values(L2gradient(false), L2gradient(true)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// MeanShift

struct MeanShift : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Mat img;

    int spatialRad;
    int colorRad;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        img = readImageType("meanshift/cones.png", CV_8UC4);
        ASSERT_FALSE(img.empty());

        spatialRad = 30;
        colorRad = 30;
    }
};

GPU_TEST_P(MeanShift, Filtering)
{
    cv::Mat img_template;
    if (supportFeature(devInfo, cv::gpu::FEATURE_SET_COMPUTE_20))
        img_template = readImage("meanshift/con_result.png");
    else
        img_template = readImage("meanshift/con_result_CC1X.png");
    ASSERT_FALSE(img_template.empty());

    cv::gpu::GpuMat d_dst;
    cv::gpu::meanShiftFiltering(loadMat(img), d_dst, spatialRad, colorRad);

    ASSERT_EQ(CV_8UC4, d_dst.type());

    cv::Mat dst(d_dst);

    cv::Mat result;
    cv::cvtColor(dst, result, cv::COLOR_BGRA2BGR);

    EXPECT_MAT_NEAR(img_template, result, 0.0);
}

GPU_TEST_P(MeanShift, Proc)
{
    cv::FileStorage fs;
    if (supportFeature(devInfo, cv::gpu::FEATURE_SET_COMPUTE_20))
        fs.open(std::string(cvtest::TS::ptr()->get_data_path()) + "meanshift/spmap.yaml", cv::FileStorage::READ);
    else
        fs.open(std::string(cvtest::TS::ptr()->get_data_path()) + "meanshift/spmap_CC1X.yaml", cv::FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());

    cv::Mat spmap_template;
    fs["spmap"] >> spmap_template;
    ASSERT_FALSE(spmap_template.empty());

    cv::gpu::GpuMat rmap_filtered;
    cv::gpu::meanShiftFiltering(loadMat(img), rmap_filtered, spatialRad, colorRad);

    cv::gpu::GpuMat rmap;
    cv::gpu::GpuMat spmap;
    cv::gpu::meanShiftProc(loadMat(img), rmap, spmap, spatialRad, colorRad);

    ASSERT_EQ(CV_8UC4, rmap.type());

    EXPECT_MAT_NEAR(rmap_filtered, rmap, 0.0);
    EXPECT_MAT_NEAR(spmap_template, spmap, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, MeanShift, ALL_DEVICES);

////////////////////////////////////////////////////////////////////////////////
// MeanShiftSegmentation

namespace
{
    IMPLEMENT_PARAM_CLASS(MinSize, int);
}

PARAM_TEST_CASE(MeanShiftSegmentation, cv::gpu::DeviceInfo, MinSize)
{
    cv::gpu::DeviceInfo devInfo;
    int minsize;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        minsize = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(MeanShiftSegmentation, Regression)
{
    cv::Mat img = readImageType("meanshift/cones.png", CV_8UC4);
    ASSERT_FALSE(img.empty());

    std::ostringstream path;
    path << "meanshift/cones_segmented_sp10_sr10_minsize" << minsize;
    if (supportFeature(devInfo, cv::gpu::FEATURE_SET_COMPUTE_20))
        path << ".png";
    else
        path << "_CC1X.png";
    cv::Mat dst_gold = readImage(path.str());
    ASSERT_FALSE(dst_gold.empty());

    cv::Mat dst;
    cv::gpu::meanShiftSegmentation(loadMat(img), dst, 10, 10, minsize);

    cv::Mat dst_rgb;
    cv::cvtColor(dst, dst_rgb, cv::COLOR_BGRA2BGR);

    EXPECT_MAT_SIMILAR(dst_gold, dst_rgb, 1e-3);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, MeanShiftSegmentation, testing::Combine(
    ALL_DEVICES,
    testing::Values(MinSize(0), MinSize(4), MinSize(20), MinSize(84), MinSize(340), MinSize(1364))));

////////////////////////////////////////////////////////////////////////////
// Blend

namespace
{
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
}

PARAM_TEST_CASE(Blend, cv::gpu::DeviceInfo, cv::Size, MatType, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int type;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(Blend, Accuracy)
{
    int depth = CV_MAT_DEPTH(type);

    cv::Mat img1 = randomMat(size, type, 0.0, depth == CV_8U ? 255.0 : 1.0);
    cv::Mat img2 = randomMat(size, type, 0.0, depth == CV_8U ? 255.0 : 1.0);
    cv::Mat weights1 = randomMat(size, CV_32F, 0, 1);
    cv::Mat weights2 = randomMat(size, CV_32F, 0, 1);

    cv::gpu::GpuMat result;
    cv::gpu::blendLinear(loadMat(img1, useRoi), loadMat(img2, useRoi), loadMat(weights1, useRoi), loadMat(weights2, useRoi), result);

    cv::Mat result_gold;
    if (depth == CV_8U)
        blendLinearGold<uchar>(img1, img2, weights1, weights2, result_gold);
    else
        blendLinearGold<float>(img1, img2, weights1, weights2, result_gold);

    EXPECT_MAT_NEAR(result_gold, result, CV_MAT_DEPTH(type) == CV_8U ? 1.0 : 1e-5);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, Blend, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4), MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate8U

CV_ENUM(TemplateMethod, cv::TM_SQDIFF, cv::TM_SQDIFF_NORMED, cv::TM_CCORR, cv::TM_CCORR_NORMED, cv::TM_CCOEFF, cv::TM_CCOEFF_NORMED)
#define ALL_TEMPLATE_METHODS testing::Values(TemplateMethod(cv::TM_SQDIFF), TemplateMethod(cv::TM_SQDIFF_NORMED), TemplateMethod(cv::TM_CCORR), TemplateMethod(cv::TM_CCORR_NORMED), TemplateMethod(cv::TM_CCOEFF), TemplateMethod(cv::TM_CCOEFF_NORMED))

namespace
{
    IMPLEMENT_PARAM_CLASS(TemplateSize, cv::Size);
}

PARAM_TEST_CASE(MatchTemplate8U, cv::gpu::DeviceInfo, cv::Size, TemplateSize, Channels, TemplateMethod)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    cv::Size templ_size;
    int cn;
    int method;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        templ_size = GET_PARAM(2);
        cn = GET_PARAM(3);
        method = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(MatchTemplate8U, Accuracy)
{
    cv::Mat image = randomMat(size, CV_MAKETYPE(CV_8U, cn));
    cv::Mat templ = randomMat(templ_size, CV_MAKETYPE(CV_8U, cn));

    cv::gpu::GpuMat dst;
    cv::gpu::matchTemplate(loadMat(image), loadMat(templ), dst, method);

    cv::Mat dst_gold;
    cv::matchTemplate(image, templ, dst_gold, method);

    EXPECT_MAT_NEAR(dst_gold, dst, templ_size.area() * 1e-1);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, MatchTemplate8U, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(TemplateSize(cv::Size(5, 5)), TemplateSize(cv::Size(16, 16)), TemplateSize(cv::Size(30, 30))),
    testing::Values(Channels(1), Channels(3), Channels(4)),
    ALL_TEMPLATE_METHODS));

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate32F

PARAM_TEST_CASE(MatchTemplate32F, cv::gpu::DeviceInfo, cv::Size, TemplateSize, Channels, TemplateMethod)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    cv::Size templ_size;
    int cn;
    int method;

    int n, m, h, w;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        templ_size = GET_PARAM(2);
        cn = GET_PARAM(3);
        method = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(MatchTemplate32F, Regression)
{
    cv::Mat image = randomMat(size, CV_MAKETYPE(CV_32F, cn));
    cv::Mat templ = randomMat(templ_size, CV_MAKETYPE(CV_32F, cn));

    cv::gpu::GpuMat dst;
    cv::gpu::matchTemplate(loadMat(image), loadMat(templ), dst, method);

    cv::Mat dst_gold;
    cv::matchTemplate(image, templ, dst_gold, method);

    EXPECT_MAT_NEAR(dst_gold, dst, templ_size.area() * 1e-1);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, MatchTemplate32F, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(TemplateSize(cv::Size(5, 5)), TemplateSize(cv::Size(16, 16)), TemplateSize(cv::Size(30, 30))),
    testing::Values(Channels(1), Channels(3), Channels(4)),
    testing::Values(TemplateMethod(cv::TM_SQDIFF), TemplateMethod(cv::TM_CCORR))));

////////////////////////////////////////////////////////////////////////////////
// MatchTemplateBlackSource

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

GPU_TEST_P(MatchTemplateBlackSource, Accuracy)
{
    cv::Mat image = readImage("matchtemplate/black.png");
    ASSERT_FALSE(image.empty());

    cv::Mat pattern = readImage("matchtemplate/cat.png");
    ASSERT_FALSE(pattern.empty());

    cv::gpu::GpuMat d_dst;
    cv::gpu::matchTemplate(loadMat(image), loadMat(pattern), d_dst, method);

    cv::Mat dst(d_dst);

    double maxValue;
    cv::Point maxLoc;
    cv::minMaxLoc(dst, NULL, &maxValue, NULL, &maxLoc);

    cv::Point maxLocGold = cv::Point(284, 12);

    ASSERT_EQ(maxLocGold, maxLoc);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, MatchTemplateBlackSource, testing::Combine(
    ALL_DEVICES,
    testing::Values(TemplateMethod(cv::TM_CCOEFF_NORMED), TemplateMethod(cv::TM_CCORR_NORMED))));

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate_CCOEF_NORMED

PARAM_TEST_CASE(MatchTemplate_CCOEF_NORMED, cv::gpu::DeviceInfo, std::pair<std::string, std::string>)
{
    cv::gpu::DeviceInfo devInfo;
    std::string imageName;
    std::string patternName;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        imageName = GET_PARAM(1).first;
        patternName = GET_PARAM(1).second;

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(MatchTemplate_CCOEF_NORMED, Accuracy)
{
    cv::Mat image = readImage(imageName);
    ASSERT_FALSE(image.empty());

    cv::Mat pattern = readImage(patternName);
    ASSERT_FALSE(pattern.empty());

    cv::gpu::GpuMat d_dst;
    cv::gpu::matchTemplate(loadMat(image), loadMat(pattern), d_dst, cv::TM_CCOEFF_NORMED);

    cv::Mat dst(d_dst);

    cv::Point minLoc, maxLoc;
    double minVal, maxVal;
    cv::minMaxLoc(dst, &minVal, &maxVal, &minLoc, &maxLoc);

    cv::Mat dstGold;
    cv::matchTemplate(image, pattern, dstGold, cv::TM_CCOEFF_NORMED);

    double minValGold, maxValGold;
    cv::Point minLocGold, maxLocGold;
    cv::minMaxLoc(dstGold, &minValGold, &maxValGold, &minLocGold, &maxLocGold);

    ASSERT_EQ(minLocGold, minLoc);
    ASSERT_EQ(maxLocGold, maxLoc);
    ASSERT_LE(maxVal, 1.0);
    ASSERT_GE(minVal, -1.0);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, MatchTemplate_CCOEF_NORMED, testing::Combine(
    ALL_DEVICES,
    testing::Values(std::make_pair(std::string("matchtemplate/source-0.png"), std::string("matchtemplate/target-0.png")))));

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate_CanFindBigTemplate

struct MatchTemplate_CanFindBigTemplate : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(MatchTemplate_CanFindBigTemplate, SQDIFF_NORMED)
{
    cv::Mat scene = readImage("matchtemplate/scene.png");
    ASSERT_FALSE(scene.empty());

    cv::Mat templ = readImage("matchtemplate/template.png");
    ASSERT_FALSE(templ.empty());

    cv::gpu::GpuMat d_result;
    cv::gpu::matchTemplate(loadMat(scene), loadMat(templ), d_result, cv::TM_SQDIFF_NORMED);

    cv::Mat result(d_result);

    double minVal;
    cv::Point minLoc;
    cv::minMaxLoc(result, &minVal, 0, &minLoc, 0);

    ASSERT_GE(minVal, 0);
    ASSERT_LT(minVal, 1e-3);
    ASSERT_EQ(344, minLoc.x);
    ASSERT_EQ(0, minLoc.y);
}

GPU_TEST_P(MatchTemplate_CanFindBigTemplate, SQDIFF)
{
    cv::Mat scene = readImage("matchtemplate/scene.png");
    ASSERT_FALSE(scene.empty());

    cv::Mat templ = readImage("matchtemplate/template.png");
    ASSERT_FALSE(templ.empty());

    cv::gpu::GpuMat d_result;
    cv::gpu::matchTemplate(loadMat(scene), loadMat(templ), d_result, cv::TM_SQDIFF);

    cv::Mat result(d_result);

    double minVal;
    cv::Point minLoc;
    cv::minMaxLoc(result, &minVal, 0, &minLoc, 0);

    ASSERT_GE(minVal, 0);
    ASSERT_EQ(344, minLoc.x);
    ASSERT_EQ(0, minLoc.y);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, MatchTemplate_CanFindBigTemplate, ALL_DEVICES);

///////////////////////////////////////////////////////////////////////////////////////////////////////
// CornerHarris

namespace
{
    IMPLEMENT_PARAM_CLASS(BlockSize, int);
    IMPLEMENT_PARAM_CLASS(ApertureSize, int);
}

PARAM_TEST_CASE(CornerHarris, cv::gpu::DeviceInfo, MatType, BorderType, BlockSize, ApertureSize)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int borderType;
    int blockSize;
    int apertureSize;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        borderType = GET_PARAM(2);
        blockSize = GET_PARAM(3);
        apertureSize = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(CornerHarris, Accuracy)
{
    cv::Mat src = readImageType("stereobm/aloe-L.png", type);
    ASSERT_FALSE(src.empty());

    double k = randomDouble(0.1, 0.9);

    cv::gpu::GpuMat dst;
    cv::gpu::cornerHarris(loadMat(src), dst, blockSize, apertureSize, k, borderType);

    cv::Mat dst_gold;
    cv::cornerHarris(src, dst_gold, blockSize, apertureSize, k, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.02);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, CornerHarris, testing::Combine(
    ALL_DEVICES,
    testing::Values(MatType(CV_8UC1), MatType(CV_32FC1)),
    testing::Values(BorderType(cv::BORDER_REFLECT101), BorderType(cv::BORDER_REPLICATE), BorderType(cv::BORDER_REFLECT)),
    testing::Values(BlockSize(3), BlockSize(5), BlockSize(7)),
    testing::Values(ApertureSize(0), ApertureSize(3), ApertureSize(5), ApertureSize(7))));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// cornerMinEigen

PARAM_TEST_CASE(CornerMinEigen, cv::gpu::DeviceInfo, MatType, BorderType, BlockSize, ApertureSize)
{
    cv::gpu::DeviceInfo devInfo;
    int type;
    int borderType;
    int blockSize;
    int apertureSize;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        borderType = GET_PARAM(2);
        blockSize = GET_PARAM(3);
        apertureSize = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

GPU_TEST_P(CornerMinEigen, Accuracy)
{
    cv::Mat src = readImageType("stereobm/aloe-L.png", type);
    ASSERT_FALSE(src.empty());

    cv::gpu::GpuMat dst;
    cv::gpu::cornerMinEigenVal(loadMat(src), dst, blockSize, apertureSize, borderType);

    cv::Mat dst_gold;
    cv::cornerMinEigenVal(src, dst_gold, blockSize, apertureSize, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.02);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, CornerMinEigen, testing::Combine(
    ALL_DEVICES,
    testing::Values(MatType(CV_8UC1), MatType(CV_32FC1)),
    testing::Values(BorderType(cv::BORDER_REFLECT101), BorderType(cv::BORDER_REPLICATE), BorderType(cv::BORDER_REFLECT)),
    testing::Values(BlockSize(3), BlockSize(5), BlockSize(7)),
    testing::Values(ApertureSize(0), ApertureSize(3), ApertureSize(5), ApertureSize(7))));

#endif // HAVE_CUDA
