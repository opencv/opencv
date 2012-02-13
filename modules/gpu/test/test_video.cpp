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

//#define DUMP

#define OPTICAL_FLOW_DUMP_FILE            "opticalflow/opticalflow_gold.bin"
#define OPTICAL_FLOW_DUMP_FILE_CC20       "opticalflow/opticalflow_gold_cc20.bin"
#define INTERPOLATE_FRAMES_DUMP_FILE      "opticalflow/interpolate_frames_gold.bin"
#define INTERPOLATE_FRAMES_DUMP_FILE_CC20 "opticalflow/interpolate_frames_gold_cc20.bin"

/////////////////////////////////////////////////////////////////////////////////////////////////
// BroxOpticalFlow

struct BroxOpticalFlow : TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;
    
    cv::Mat frame0;
    cv::Mat frame1;

    cv::Mat u_gold;
    cv::Mat v_gold;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());
        
        frame0 = readImage("opticalflow/frame0.png", cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(frame0.empty());
        frame0.convertTo(frame0, CV_32F, 1.0 / 255.0);
        
        frame1 = readImage("opticalflow/frame1.png", cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(frame1.empty());
        frame1.convertTo(frame1, CV_32F, 1.0 / 255.0);

#ifndef DUMP

        std::string fname(cvtest::TS::ptr()->get_data_path());
        if (devInfo.majorVersion() >= 2)
            fname += OPTICAL_FLOW_DUMP_FILE_CC20;
        else
            fname += OPTICAL_FLOW_DUMP_FILE;

        std::ifstream f(fname.c_str(), std::ios_base::binary);

        int rows, cols;

        f.read((char*)&rows, sizeof(rows));
        f.read((char*)&cols, sizeof(cols));

        u_gold.create(rows, cols, CV_32FC1);

        for (int i = 0; i < u_gold.rows; ++i)
            f.read((char*)u_gold.ptr(i), u_gold.cols * sizeof(float));

        v_gold.create(rows, cols, CV_32FC1);

        for (int i = 0; i < v_gold.rows; ++i)
            f.read((char*)v_gold.ptr(i), v_gold.cols * sizeof(float));

#endif
    }
};

TEST_P(BroxOpticalFlow, Regression)
{
    cv::Mat u;
    cv::Mat v;

    cv::gpu::BroxOpticalFlow d_flow(0.197f /*alpha*/, 50.0f /*gamma*/, 0.8f /*scale_factor*/, 
                                    10 /*inner_iterations*/, 77 /*outer_iterations*/, 10 /*solver_iterations*/);

    cv::gpu::GpuMat d_u; 
    cv::gpu::GpuMat d_v;

    d_flow(cv::gpu::GpuMat(frame0), cv::gpu::GpuMat(frame1), d_u, d_v);

    d_u.download(u);
    d_v.download(v);

#ifndef DUMP

    EXPECT_MAT_NEAR(u_gold, u, 0);
    EXPECT_MAT_NEAR(v_gold, v, 0);

#else

    std::string fname(cvtest::TS::ptr()->get_data_path());
    if (devInfo.majorVersion() >= 2)
        fname += OPTICAL_FLOW_DUMP_FILE_CC20;
    else
        fname += OPTICAL_FLOW_DUMP_FILE;

    std::ofstream f(fname.c_str(), std::ios_base::binary);

    f.write((char*)&u.rows, sizeof(u.rows));
    f.write((char*)&u.cols, sizeof(u.cols));

    for (int i = 0; i < u.rows; ++i)
        f.write((char*)u.ptr(i), u.cols * sizeof(float));

    for (int i = 0; i < v.rows; ++i)
        f.write((char*)v.ptr(i), v.cols * sizeof(float));

#endif
}

INSTANTIATE_TEST_CASE_P(Video, BroxOpticalFlow, ALL_DEVICES);

/////////////////////////////////////////////////////////////////////////////////////////////////
// InterpolateFrames

struct InterpolateFrames : TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;
    
    cv::Mat frame0;
    cv::Mat frame1;

    cv::Mat newFrame_gold;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());
        
        frame0 = readImage("opticalflow/frame0.png", cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(frame0.empty());
        frame0.convertTo(frame0, CV_32F, 1.0 / 255.0);
        
        frame1 = readImage("opticalflow/frame1.png", cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(frame1.empty());
        frame1.convertTo(frame1, CV_32F, 1.0 / 255.0);

#ifndef DUMP

        std::string fname(cvtest::TS::ptr()->get_data_path());
        if (devInfo.majorVersion() >= 2)
            fname += INTERPOLATE_FRAMES_DUMP_FILE_CC20;
        else
            fname += INTERPOLATE_FRAMES_DUMP_FILE;

        std::ifstream f(fname.c_str(), std::ios_base::binary);

        int rows, cols;

        f.read((char*)&rows, sizeof(rows));
        f.read((char*)&cols, sizeof(cols));

        newFrame_gold.create(rows, cols, CV_32FC1);

        for (int i = 0; i < newFrame_gold.rows; ++i)
            f.read((char*)newFrame_gold.ptr(i), newFrame_gold.cols * sizeof(float));

#endif
    }
};

TEST_P(InterpolateFrames, Regression)
{
    cv::Mat newFrame;

    cv::gpu::BroxOpticalFlow d_flow(0.197f /*alpha*/, 50.0f /*gamma*/, 0.8f /*scale_factor*/, 
                                    10 /*inner_iterations*/, 77 /*outer_iterations*/, 10 /*solver_iterations*/);

    cv::gpu::GpuMat d_frame0(frame0);
    cv::gpu::GpuMat d_frame1(frame1);

    cv::gpu::GpuMat d_fu; 
    cv::gpu::GpuMat d_fv;
    cv::gpu::GpuMat d_bu; 
    cv::gpu::GpuMat d_bv;

    d_flow(d_frame0, d_frame1, d_fu, d_fv);
    d_flow(d_frame1, d_frame0, d_bu, d_bv);

    cv::gpu::GpuMat d_newFrame;
    cv::gpu::GpuMat d_buf;

    cv::gpu::interpolateFrames(d_frame0, d_frame1, d_fu, d_fv, d_bu, d_bv, 0.5f, d_newFrame, d_buf);

    d_newFrame.download(newFrame);

#ifndef DUMP

    EXPECT_MAT_NEAR(newFrame_gold, newFrame, 1e-3);

#else

    std::string fname(cvtest::TS::ptr()->get_data_path());
    if (devInfo.majorVersion() >= 2)
        fname += INTERPOLATE_FRAMES_DUMP_FILE_CC20;
    else
        fname += INTERPOLATE_FRAMES_DUMP_FILE;

    std::ofstream f(fname.c_str(), std::ios_base::binary);

    f.write((char*)&newFrame.rows, sizeof(newFrame.rows));
    f.write((char*)&newFrame.cols, sizeof(newFrame.cols));

    for (int i = 0; i < newFrame.rows; ++i)
        f.write((char*)newFrame.ptr(i), newFrame.cols * sizeof(float));

#endif
}

INSTANTIATE_TEST_CASE_P(Video, InterpolateFrames, ALL_DEVICES);

/////////////////////////////////////////////////////////////////////////////////////////////////
// GoodFeaturesToTrack

PARAM_TEST_CASE(GoodFeaturesToTrack, cv::gpu::DeviceInfo, double)
{
    cv::gpu::DeviceInfo devInfo;
    
    cv::Mat image;

    int maxCorners;
    double qualityLevel;
    double minDistance;

    std::vector<cv::Point2f> pts_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        minDistance = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());
        
        image = readImage("opticalflow/frame0.png", cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(image.empty());

        maxCorners = 1000;
        qualityLevel= 0.01;

        cv::goodFeaturesToTrack(image, pts_gold, maxCorners, qualityLevel, minDistance);
    }
};

TEST_P(GoodFeaturesToTrack, Accuracy)
{
    cv::gpu::GoodFeaturesToTrackDetector_GPU detector(maxCorners, qualityLevel, minDistance);

    cv::gpu::GpuMat d_pts;

    detector(loadMat(image), d_pts);

    std::vector<cv::Point2f> pts(d_pts.cols);
    cv::Mat pts_mat(1, d_pts.cols, CV_32FC2, (void*)&pts[0]);
    d_pts.download(pts_mat);

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

INSTANTIATE_TEST_CASE_P(Video, GoodFeaturesToTrack, Combine(ALL_DEVICES, Values(0.0, 3.0)));

/////////////////////////////////////////////////////////////////////////////////////////////////
// PyrLKOpticalFlow

PARAM_TEST_CASE(PyrLKOpticalFlowSparse, cv::gpu::DeviceInfo, bool)
{
    cv::gpu::DeviceInfo devInfo;
    
    cv::Mat frame0;
    cv::Mat frame1;

    std::vector<cv::Point2f> pts;

    std::vector<cv::Point2f> nextPts_gold;
    std::vector<unsigned char> status_gold;
    std::vector<float> err_gold;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        bool useGray = GET_PARAM(1);

        cv::gpu::setDevice(devInfo.deviceID());
        
        frame0 = readImage("opticalflow/frame0.png", useGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
        ASSERT_FALSE(frame0.empty());
        
        frame1 = readImage("opticalflow/frame1.png", useGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
        ASSERT_FALSE(frame1.empty());

        cv::Mat gray_frame;
        if (useGray)
            gray_frame = frame0;
        else
            cv::cvtColor(frame0, gray_frame, cv::COLOR_BGR2GRAY);

        cv::goodFeaturesToTrack(gray_frame, pts, 1000, 0.01, 0.0);

        cv::calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts_gold, status_gold, err_gold, cv::Size(21, 21), 3, 
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), 0.5, CV_LKFLOW_GET_MIN_EIGENVALS);
    }
};

TEST_P(PyrLKOpticalFlowSparse, Accuracy)
{
    cv::gpu::PyrLKOpticalFlow d_pyrLK;

    cv::gpu::GpuMat d_pts;
    cv::Mat pts_mat(1, pts.size(), CV_32FC2, (void*)&pts[0]);
    d_pts.upload(pts_mat);

    cv::gpu::GpuMat d_nextPts;
    cv::gpu::GpuMat d_status;
    cv::gpu::GpuMat d_err;

    d_pyrLK.sparse(loadMat(frame0), loadMat(frame1), d_pts, d_nextPts, d_status, &d_err);

    std::vector<cv::Point2f> nextPts(d_nextPts.cols);
    cv::Mat nextPts_mat(1, d_nextPts.cols, CV_32FC2, (void*)&nextPts[0]);
    d_nextPts.download(nextPts_mat);

    std::vector<unsigned char> status(d_status.cols);
    cv::Mat status_mat(1, d_status.cols, CV_8UC1, (void*)&status[0]);
    d_status.download(status_mat);

    std::vector<float> err(d_err.cols);
    cv::Mat err_mat(1, d_err.cols, CV_32FC1, (void*)&err[0]);
    d_err.download(err_mat);

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

            if (!eq || errdiff > 1e-4)
                ++mistmatch;
        }
    }

    double bad_ratio = static_cast<double>(mistmatch) / nextPts.size();

    ASSERT_LE(bad_ratio, 0.01);
}

INSTANTIATE_TEST_CASE_P(Video, PyrLKOpticalFlowSparse, Combine(ALL_DEVICES, Bool()));

#endif // HAVE_CUDA
