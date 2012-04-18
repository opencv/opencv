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

//#define DUMP

/////////////////////////////////////////////////////////////////////////////////////////////////
// BroxOpticalFlow

#define BROX_OPTICAL_FLOW_DUMP_FILE            "opticalflow/brox_optical_flow.bin"
#define BROX_OPTICAL_FLOW_DUMP_FILE_CC20       "opticalflow/brox_optical_flow_cc20.bin"

struct BroxOpticalFlow : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(BroxOpticalFlow, Regression)
{
    cv::Mat frame0 = readImageType("opticalflow/frame0.png", CV_32FC1);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImageType("opticalflow/frame1.png", CV_32FC1);
    ASSERT_FALSE(frame1.empty());

    cv::gpu::BroxOpticalFlow brox(0.197f /*alpha*/, 50.0f /*gamma*/, 0.8f /*scale_factor*/,
                                  10 /*inner_iterations*/, 77 /*outer_iterations*/, 10 /*solver_iterations*/);

    cv::gpu::GpuMat u;
    cv::gpu::GpuMat v;
    brox(loadMat(frame0), loadMat(frame1), u, v);

#ifndef DUMP
    std::string fname(cvtest::TS::ptr()->get_data_path());
    if (devInfo.majorVersion() >= 2)
        fname += BROX_OPTICAL_FLOW_DUMP_FILE_CC20;
    else
        fname += BROX_OPTICAL_FLOW_DUMP_FILE;

    std::ifstream f(fname.c_str(), std::ios_base::binary);

    int rows, cols;

    f.read((char*)&rows, sizeof(rows));
    f.read((char*)&cols, sizeof(cols));

    cv::Mat u_gold(rows, cols, CV_32FC1);

    for (int i = 0; i < u_gold.rows; ++i)
        f.read(u_gold.ptr<char>(i), u_gold.cols * sizeof(float));

    cv::Mat v_gold(rows, cols, CV_32FC1);

    for (int i = 0; i < v_gold.rows; ++i)
        f.read(v_gold.ptr<char>(i), v_gold.cols * sizeof(float));

    EXPECT_MAT_NEAR(u_gold, u, 0);
    EXPECT_MAT_NEAR(v_gold, v, 0);
#else
    std::string fname(cvtest::TS::ptr()->get_data_path());
    if (devInfo.majorVersion() >= 2)
        fname += BROX_OPTICAL_FLOW_DUMP_FILE_CC20;
    else
        fname += BROX_OPTICAL_FLOW_DUMP_FILE;

    std::ofstream f(fname.c_str(), std::ios_base::binary);

    f.write((char*)&u.rows, sizeof(u.rows));
    f.write((char*)&u.cols, sizeof(u.cols));

    cv::Mat h_u(u);
    cv::Mat h_v(v);

    for (int i = 0; i < u.rows; ++i)
        f.write(h_u.ptr<char>(i), u.cols * sizeof(float));

    for (int i = 0; i < v.rows; ++i)
        f.write(h_v.ptr<char>(i), v.cols * sizeof(float));

#endif
}

INSTANTIATE_TEST_CASE_P(GPU_Video, BroxOpticalFlow, ALL_DEVICES);

/////////////////////////////////////////////////////////////////////////////////////////////////
// GoodFeaturesToTrack

IMPLEMENT_PARAM_CLASS(MinDistance, double)

PARAM_TEST_CASE(GoodFeaturesToTrack, cv::gpu::DeviceInfo, MinDistance)
{
    cv::gpu::DeviceInfo devInfo;
    double minDistance;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        minDistance = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(GoodFeaturesToTrack, Accuracy)
{
    cv::Mat image = readImage("opticalflow/frame0.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    int maxCorners = 1000;
    double qualityLevel = 0.01;

    cv::gpu::GoodFeaturesToTrackDetector_GPU detector(maxCorners, qualityLevel, minDistance);

    if (!supportFeature(devInfo, cv::gpu::GLOBAL_ATOMICS))
    {
        try
        {
            cv::gpu::GpuMat d_pts;
            detector(loadMat(image), d_pts);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsNotImplemented, e.code);
        }
    }
    else
    {
        cv::gpu::GpuMat d_pts;
        detector(loadMat(image), d_pts);

        std::vector<cv::Point2f> pts(d_pts.cols);
        cv::Mat pts_mat(1, d_pts.cols, CV_32FC2, (void*)&pts[0]);
        d_pts.download(pts_mat);

        std::vector<cv::Point2f> pts_gold;
        cv::goodFeaturesToTrack(image, pts_gold, maxCorners, qualityLevel, minDistance);

        ASSERT_EQ(pts_gold.size(), pts.size());

        size_t mistmatch = 0;
        for (size_t i = 0; i < pts.size(); ++i)
        {
            cv::Point2i a = pts_gold[i];
            cv::Point2i b = pts[i];

            bool eq = std::abs(a.x - b.x) < 1 && std::abs(a.y - b.y) < 1;

            if (!eq)
                ++mistmatch;
        }

        double bad_ratio = static_cast<double>(mistmatch) / pts.size();

        ASSERT_LE(bad_ratio, 0.01);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Video, GoodFeaturesToTrack, testing::Combine(
    ALL_DEVICES,
    testing::Values(MinDistance(0.0), MinDistance(3.0))));

/////////////////////////////////////////////////////////////////////////////////////////////////
// PyrLKOpticalFlow

IMPLEMENT_PARAM_CLASS(UseGray, bool)

PARAM_TEST_CASE(PyrLKOpticalFlow, cv::gpu::DeviceInfo, UseGray)
{
    cv::gpu::DeviceInfo devInfo;
    bool useGray;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        useGray = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(PyrLKOpticalFlow, Sparse)
{
    cv::Mat frame0 = readImage("opticalflow/frame0.png", useGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage("opticalflow/frame1.png", useGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    ASSERT_FALSE(frame1.empty());

    cv::Mat gray_frame;
    if (useGray)
        gray_frame = frame0;
    else
        cv::cvtColor(frame0, gray_frame, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> pts;
    cv::goodFeaturesToTrack(gray_frame, pts, 1000, 0.01, 0.0);

    cv::gpu::GpuMat d_pts;
    cv::Mat pts_mat(1, (int)pts.size(), CV_32FC2, (void*)&pts[0]);
    d_pts.upload(pts_mat);

    cv::gpu::PyrLKOpticalFlow pyrLK;

    cv::gpu::GpuMat d_nextPts;
    cv::gpu::GpuMat d_status;
    cv::gpu::GpuMat d_err;
    pyrLK.sparse(loadMat(frame0), loadMat(frame1), d_pts, d_nextPts, d_status, &d_err);

    std::vector<cv::Point2f> nextPts(d_nextPts.cols);
    cv::Mat nextPts_mat(1, d_nextPts.cols, CV_32FC2, (void*)&nextPts[0]);
    d_nextPts.download(nextPts_mat);

    std::vector<unsigned char> status(d_status.cols);
    cv::Mat status_mat(1, d_status.cols, CV_8UC1, (void*)&status[0]);
    d_status.download(status_mat);

    std::vector<float> err(d_err.cols);
    cv::Mat err_mat(1, d_err.cols, CV_32FC1, (void*)&err[0]);
    d_err.download(err_mat);

    std::vector<cv::Point2f> nextPts_gold;
    std::vector<unsigned char> status_gold;
    std::vector<float> err_gold;
    cv::calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts_gold, status_gold, err_gold);

    ASSERT_EQ(nextPts_gold.size(), nextPts.size());
    ASSERT_EQ(status_gold.size(), status.size());
    ASSERT_EQ(err_gold.size(), err.size());

    size_t mistmatch = 0;
    for (size_t i = 0; i < nextPts.size(); ++i)
    {
        if (status[i] != status_gold[i])
        {
            ++mistmatch;
            continue;
        }

        if (status[i])
        {
            cv::Point2i a = nextPts[i];
            cv::Point2i b = nextPts_gold[i];

            bool eq = std::abs(a.x - b.x) < 1 && std::abs(a.y - b.y) < 1;
            float errdiff = std::abs(err[i] - err_gold[i]);

            if (!eq || errdiff > 1e-1)
                ++mistmatch;
        }
    }

    double bad_ratio = static_cast<double>(mistmatch) / nextPts.size();

    ASSERT_LE(bad_ratio, 0.01);
}

INSTANTIATE_TEST_CASE_P(GPU_Video, PyrLKOpticalFlow, testing::Combine(
    ALL_DEVICES,
    testing::Values(UseGray(true), UseGray(false))));

/////////////////////////////////////////////////////////////////////////////////////////////////
// FarnebackOpticalFlow

IMPLEMENT_PARAM_CLASS(PyrScale, double)
IMPLEMENT_PARAM_CLASS(PolyN, int)
CV_FLAGS(FarnebackOptFlowFlags, 0, cv::OPTFLOW_FARNEBACK_GAUSSIAN)
IMPLEMENT_PARAM_CLASS(UseInitFlow, bool)

PARAM_TEST_CASE(FarnebackOpticalFlow, cv::gpu::DeviceInfo, PyrScale, PolyN, FarnebackOptFlowFlags, UseInitFlow)
{
    cv::gpu::DeviceInfo devInfo;
    double pyrScale;
    int polyN;
    int flags;
    bool useInitFlow;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        pyrScale = GET_PARAM(1);
        polyN = GET_PARAM(2);
        flags = GET_PARAM(3);
        useInitFlow = GET_PARAM(4);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(FarnebackOpticalFlow, Accuracy)
{
    cv::Mat frame0 = readImage("opticalflow/rubberwhale1.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage("opticalflow/rubberwhale2.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    double polySigma = polyN <= 5 ? 1.1 : 1.5;

    cv::gpu::FarnebackOpticalFlow calc;
    calc.pyrScale = pyrScale;
    calc.polyN = polyN;
    calc.polySigma = polySigma;
    calc.flags = flags;

    cv::gpu::GpuMat d_flowx, d_flowy;
    calc(loadMat(frame0), loadMat(frame1), d_flowx, d_flowy);

    cv::Mat flow;
    if (useInitFlow)
    {
        cv::Mat flowxy[] = {cv::Mat(d_flowx), cv::Mat(d_flowy)};
        cv::merge(flowxy, 2, flow);
    }

    if (useInitFlow)
    {
        calc.flags |= cv::OPTFLOW_USE_INITIAL_FLOW;
        calc(loadMat(frame0), loadMat(frame1), d_flowx, d_flowy);
    }

    cv::calcOpticalFlowFarneback(
        frame0, frame1, flow, calc.pyrScale, calc.numLevels, calc.winSize,
        calc.numIters,  calc.polyN, calc.polySigma, calc.flags);

    std::vector<cv::Mat> flowxy;
    cv::split(flow, flowxy);

    EXPECT_MAT_SIMILAR(flowxy[0], d_flowx, 0.1);
    EXPECT_MAT_SIMILAR(flowxy[1], d_flowy, 0.1);
}

INSTANTIATE_TEST_CASE_P(GPU_Video, FarnebackOpticalFlow, testing::Combine(
    ALL_DEVICES,
    testing::Values(PyrScale(0.3), PyrScale(0.5), PyrScale(0.8)),
    testing::Values(PolyN(5), PolyN(7)),
    testing::Values(FarnebackOptFlowFlags(0), FarnebackOptFlowFlags(cv::OPTFLOW_FARNEBACK_GAUSSIAN)),
    testing::Values(UseInitFlow(false), UseInitFlow(true))));

/////////////////////////////////////////////////////////////////////////////////////////////////
// VideoWriter

#ifdef WIN32

PARAM_TEST_CASE(VideoWriter, cv::gpu::DeviceInfo, std::string)
{
    cv::gpu::DeviceInfo devInfo;
    std::string inputFile;

    std::string outputFile;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        inputFile = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());

        inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "video/" + inputFile;
        outputFile = inputFile.substr(0, inputFile.find('.')) + "_test.avi";
    }
};

TEST_P(VideoWriter, Regression)
{
    const double FPS = 25.0;

    cv::VideoCapture reader(inputFile);
    ASSERT_TRUE( reader.isOpened() );

    cv::gpu::VideoWriter_GPU d_writer;

    cv::Mat frame;
    std::vector<cv::Mat> frames;
    cv::gpu::GpuMat d_frame;

    for (int i = 1; i < 10; ++i)
    {
        reader >> frame;

        if (frame.empty())
            break;

        frames.push_back(frame.clone());
        d_frame.upload(frame);

        if (!d_writer.isOpened())
            d_writer.open(outputFile, frame.size(), FPS);

        d_writer.write(d_frame);
    }

    reader.release();
    d_writer.close();

    reader.open(outputFile);
    ASSERT_TRUE( reader.isOpened() );

    for (int i = 0; i < 5; ++i)
    {
        reader >> frame;
        ASSERT_FALSE( frame.empty() );
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Video, VideoWriter, testing::Combine(
    ALL_DEVICES,
    testing::Values(std::string("VID00003-20100701-2204.mpg"), std::string("big_buck_bunny.mpg"))));

#endif // WIN32

/////////////////////////////////////////////////////////////////////////////////////////////////
// VideoReader

PARAM_TEST_CASE(VideoReader, cv::gpu::DeviceInfo, std::string)
{
    cv::gpu::DeviceInfo devInfo;
    std::string inputFile;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        inputFile = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());

        inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "video/" + inputFile;
    }
};

TEST_P(VideoReader, Regression)
{
    cv::gpu::VideoReader_GPU reader(inputFile);
    ASSERT_TRUE( reader.isOpened() );

    cv::gpu::GpuMat frame;

    for (int i = 0; i < 5; ++i)
    {
        ASSERT_TRUE( reader.read(frame) );
        ASSERT_FALSE( frame.empty() );
    }

    reader.close();
    ASSERT_FALSE( reader.isOpened() );
}

INSTANTIATE_TEST_CASE_P(GPU_Video, VideoReader, testing::Combine(
    ALL_DEVICES,
    testing::Values(std::string("VID00003-20100701-2204.mpg"))));

} // namespace
