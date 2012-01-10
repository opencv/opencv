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

    ASSERT_NO_THROW(
        cv::gpu::GpuMat d_u; 
        cv::gpu::GpuMat d_v;
        d_flow(cv::gpu::GpuMat(frame0), cv::gpu::GpuMat(frame1), d_u, d_v);
        d_u.download(u);
        d_v.download(v);
        d_flow.buf.release();
    );

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

    ASSERT_NO_THROW(
        cv::gpu::GpuMat d_frame0(frame0);
        cv::gpu::GpuMat d_frame1(frame1);
        cv::gpu::GpuMat d_fu; 
        cv::gpu::GpuMat d_fv;
        cv::gpu::GpuMat d_bu; 
        cv::gpu::GpuMat d_bv;
        cv::gpu::GpuMat d_newFrame;
        cv::gpu::GpuMat d_buf;
        d_flow(d_frame0, d_frame1, d_fu, d_fv);
        d_flow(d_frame1, d_frame0, d_bu, d_bv);
        cv::gpu::interpolateFrames(d_frame0, d_frame1, d_fu, d_fv, d_bu, d_bv, 0.5f, d_newFrame, d_buf);
        d_newFrame.download(newFrame);
        d_flow.buf.release();
    );

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

#endif
