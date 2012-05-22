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

#include "precomp.hpp"

namespace {

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Integral

PARAM_TEST_CASE(Integral, cv::gpu::DeviceInfo, cv::Size, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        useRoi = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Integral, Accuracy)
{
    cv::Mat src = randomMat(size, CV_8UC1);

    cv::gpu::GpuMat dst = createMat(cv::Size(src.cols + 1, src.rows + 1), CV_32SC1, useRoi);
    cv::gpu::integral(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::integral(src, dst_gold, CV_32S);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, Integral, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    WHOLE_SUBMAT));

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

TEST_P(HistEven, Accuracy)
{
    cv::Mat img = readImage("stereobm/aloe-L.png");
    ASSERT_FALSE(img.empty());

    cv::Mat hsv;
    cv::cvtColor(img, hsv, CV_BGR2HSV);

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

PARAM_TEST_CASE(CalcHist, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo;

    cv::Size size;
    cv::Mat src;
    cv::Mat hist_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(CalcHist, Accuracy)
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

TEST_P(EqualizeHist, Accuracy)
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

////////////////////////////////////////////////////////////////////////
// ColumnSum

PARAM_TEST_CASE(ColumnSum, cv::gpu::DeviceInfo, cv::Size)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;

    cv::Mat src;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(ColumnSum, Accuracy)
{
    cv::Mat src = randomMat(size, CV_32FC1);

    cv::gpu::GpuMat d_dst;
    cv::gpu::columnSum(loadMat(src), d_dst);

    cv::Mat dst(d_dst);

    for (int j = 0; j < src.cols; ++j)
    {
        float gold = src.at<float>(0, j);
        float res = dst.at<float>(0, j);
        ASSERT_NEAR(res, gold, 1e-5);
    }

    for (int i = 1; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            float gold = src.at<float>(i, j) += src.at<float>(i - 1, j);
            float res = dst.at<float>(i, j);
            ASSERT_NEAR(res, gold, 1e-5);
        }
    }
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, ColumnSum, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES));

////////////////////////////////////////////////////////
// Canny

IMPLEMENT_PARAM_CLASS(AppertureSize, int);
IMPLEMENT_PARAM_CLASS(L2gradient, bool);

PARAM_TEST_CASE(Canny, cv::gpu::DeviceInfo, AppertureSize, L2gradient, UseRoi)
{
    cv::gpu::DeviceInfo devInfo;
    int apperture_size;
    bool useL2gradient;
    bool useRoi;

    cv::Mat edges_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        apperture_size = GET_PARAM(1);
        useL2gradient = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Canny, Accuracy)
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
            ASSERT_EQ(CV_StsNotImplemented, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat edges;
        cv::gpu::Canny(loadMat(img, useRoi), edges, low_thresh, high_thresh, apperture_size, useL2gradient);

        cv::Mat edges_gold;
        cv::Canny(img, edges_gold, low_thresh, high_thresh, apperture_size, useL2gradient);

        EXPECT_MAT_SIMILAR(edges_gold, edges, 1e-2);
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

TEST_P(MeanShift, Filtering)
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
    cv::cvtColor(dst, result, CV_BGRA2BGR);

    EXPECT_MAT_NEAR(img_template, result, 0.0);
}

TEST_P(MeanShift, Proc)
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

IMPLEMENT_PARAM_CLASS(MinSize, int);

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

TEST_P(MeanShiftSegmentation, Regression)
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
    cv::cvtColor(dst, dst_rgb, CV_BGRA2BGR);

    EXPECT_MAT_SIMILAR(dst_gold, dst_rgb, 1e-3);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, MeanShiftSegmentation, testing::Combine(
    ALL_DEVICES,
    testing::Values(MinSize(0), MinSize(4), MinSize(20), MinSize(84), MinSize(340), MinSize(1364))));

////////////////////////////////////////////////////////////////////////////
// Blend

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

TEST_P(Blend, Accuracy)
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

////////////////////////////////////////////////////////
// Convolve

void convolveDFT(const cv::Mat& A, const cv::Mat& B, cv::Mat& C, bool ccorr = false)
{
    // reallocate the output array if needed
    C.create(std::abs(A.rows - B.rows) + 1, std::abs(A.cols - B.cols) + 1, A.type());
    cv::Size dftSize;

    // compute the size of DFT transform
    dftSize.width = cv::getOptimalDFTSize(A.cols + B.cols - 1);
    dftSize.height = cv::getOptimalDFTSize(A.rows + B.rows - 1);

    // allocate temporary buffers and initialize them with 0s
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

IMPLEMENT_PARAM_CLASS(KSize, int);
IMPLEMENT_PARAM_CLASS(Ccorr, bool);

PARAM_TEST_CASE(Convolve, cv::gpu::DeviceInfo, cv::Size, KSize, Ccorr)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int ksize;
    bool ccorr;

    cv::Mat src;
    cv::Mat kernel;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        ksize = GET_PARAM(2);
        ccorr = GET_PARAM(3);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(Convolve, Accuracy)
{
    cv::Mat src = randomMat(size, CV_32FC1, 0.0, 100.0);
    cv::Mat kernel = randomMat(cv::Size(ksize, ksize), CV_32FC1, 0.0, 1.0);

    cv::gpu::GpuMat dst;
    cv::gpu::convolve(loadMat(src), loadMat(kernel), dst, ccorr);
    
    cv::Mat dst_gold;
    convolveDFT(src, kernel, dst_gold, ccorr);

    EXPECT_MAT_NEAR(dst, dst_gold, 1e-1);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, Convolve, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(KSize(3), KSize(7), KSize(11), KSize(17), KSize(19), KSize(23), KSize(45)),
    testing::Values(Ccorr(false), Ccorr(true))));

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate8U

CV_ENUM(TemplateMethod, cv::TM_SQDIFF, cv::TM_SQDIFF_NORMED, cv::TM_CCORR, cv::TM_CCORR_NORMED, cv::TM_CCOEFF, cv::TM_CCOEFF_NORMED)
#define ALL_TEMPLATE_METHODS testing::Values(TemplateMethod(cv::TM_SQDIFF), TemplateMethod(cv::TM_SQDIFF_NORMED), TemplateMethod(cv::TM_CCORR), TemplateMethod(cv::TM_CCORR_NORMED), TemplateMethod(cv::TM_CCOEFF), TemplateMethod(cv::TM_CCOEFF_NORMED))

IMPLEMENT_PARAM_CLASS(TemplateSize, cv::Size);

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

TEST_P(MatchTemplate8U, Accuracy)
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
    cv::Mat image, templ;

    cv::Mat dst_gold;

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

TEST_P(MatchTemplate32F, Regression)
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

TEST_P(MatchTemplateBlackSource, Accuracy)
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

TEST_P(MatchTemplate_CCOEF_NORMED, Accuracy)
{
    cv::Mat image = readImage(imageName);
    ASSERT_FALSE(image.empty());

    cv::Mat pattern = readImage(patternName);
    ASSERT_FALSE(pattern.empty());

    cv::gpu::GpuMat d_dst;
    cv::gpu::matchTemplate(loadMat(image), loadMat(pattern), d_dst, CV_TM_CCOEFF_NORMED);

    cv::Mat dst(d_dst);

    cv::Point minLoc, maxLoc;
    double minVal, maxVal;
    cv::minMaxLoc(dst, &minVal, &maxVal, &minLoc, &maxLoc);

    cv::Mat dstGold;
    cv::matchTemplate(image, pattern, dstGold, CV_TM_CCOEFF_NORMED);

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

TEST_P(MatchTemplate_CanFindBigTemplate, SQDIFF_NORMED)
{
    cv::Mat scene = readImage("matchtemplate/scene.jpg");
    ASSERT_FALSE(scene.empty());

    cv::Mat templ = readImage("matchtemplate/template.jpg");
    ASSERT_FALSE(templ.empty());

    cv::gpu::GpuMat d_result;
    cv::gpu::matchTemplate(loadMat(scene), loadMat(templ), d_result, CV_TM_SQDIFF_NORMED);

    cv::Mat result(d_result);

    double minVal;
    cv::Point minLoc;
    cv::minMaxLoc(result, &minVal, 0, &minLoc, 0);

    ASSERT_GE(minVal, 0);
    ASSERT_LT(minVal, 1e-3);
    ASSERT_EQ(344, minLoc.x);
    ASSERT_EQ(0, minLoc.y);
}

TEST_P(MatchTemplate_CanFindBigTemplate, SQDIFF)
{
    cv::Mat scene = readImage("matchtemplate/scene.jpg");
    ASSERT_FALSE(scene.empty());

    cv::Mat templ = readImage("matchtemplate/template.jpg");
    ASSERT_FALSE(templ.empty());

    cv::gpu::GpuMat d_result;
    cv::gpu::matchTemplate(loadMat(scene), loadMat(templ), d_result, CV_TM_SQDIFF);

    cv::Mat result(d_result);

    double minVal;
    cv::Point minLoc;
    cv::minMaxLoc(result, &minVal, 0, &minLoc, 0);

    ASSERT_GE(minVal, 0);
    ASSERT_EQ(344, minLoc.x);
    ASSERT_EQ(0, minLoc.y);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, MatchTemplate_CanFindBigTemplate, ALL_DEVICES);

////////////////////////////////////////////////////////////////////////////
// MulSpectrums

CV_FLAGS(DftFlags, 0, cv::DFT_INVERSE, cv::DFT_SCALE, cv::DFT_ROWS, cv::DFT_COMPLEX_OUTPUT, cv::DFT_REAL_OUTPUT)

PARAM_TEST_CASE(MulSpectrums, cv::gpu::DeviceInfo, cv::Size, DftFlags)
{
    cv::gpu::DeviceInfo devInfo;
    cv::Size size;
    int flag;

    cv::Mat a, b;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        flag = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        a = randomMat(size, CV_32FC2);
        b = randomMat(size, CV_32FC2);
    }
};

TEST_P(MulSpectrums, Simple)
{
    cv::gpu::GpuMat c;
    cv::gpu::mulSpectrums(loadMat(a), loadMat(b), c, flag, false);

    cv::Mat c_gold;
    cv::mulSpectrums(a, b, c_gold, flag, false);

    EXPECT_MAT_NEAR(c_gold, c, 1e-2);
}

TEST_P(MulSpectrums, Scaled)
{
    float scale = 1.f / size.area();

    cv::gpu::GpuMat c;
    cv::gpu::mulAndScaleSpectrums(loadMat(a), loadMat(b), c, flag, scale, false);

    cv::Mat c_gold;
    cv::mulSpectrums(a, b, c_gold, flag, false);
    c_gold.convertTo(c_gold, c_gold.type(), scale);

    EXPECT_MAT_NEAR(c_gold, c, 1e-2);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, MulSpectrums, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(DftFlags(0), DftFlags(cv::DFT_ROWS))));

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

void testC2C(const std::string& hint, int cols, int rows, int flags, bool inplace)
{
    SCOPED_TRACE(hint);

    cv::Mat a = randomMat(cv::Size(cols, rows), CV_32FC2, 0.0, 10.0);

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
    int cols = randomInt(2, 100);
    int rows = randomInt(2, 100);

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

    cv::Mat a = randomMat(cv::Size(cols, rows), CV_32FC1, 0.0, 10.0);

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
    int cols = randomInt(2, 100);
    int rows = randomInt(2, 100);

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

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, Dft, ALL_DEVICES);

///////////////////////////////////////////////////////////////////////////////////////////////////////
// CornerHarris

IMPLEMENT_PARAM_CLASS(BlockSize, int);
IMPLEMENT_PARAM_CLASS(ApertureSize, int);

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

TEST_P(CornerHarris, Accuracy)
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

TEST_P(CornerMinEigen, Accuracy)
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

} // namespace
