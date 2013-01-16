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
#include <iomanip>

#ifdef HAVE_OPENCL

using namespace cv;
using namespace cv::ocl;
using namespace cvtest;
using namespace testing;
using namespace std;

//#define DUMP

/////////////////////////////////////////////////////////////////////////////////////////////////
// BroxOpticalFlow
extern string workdir;
#define BROX_OPTICAL_FLOW_DUMP_FILE            "opticalflow/brox_optical_flow.bin"
#define BROX_OPTICAL_FLOW_DUMP_FILE_CC20       "opticalflow/brox_optical_flow_cc20.bin"


/////////////////////////////////////////////////////////////////////////////////////////////////
// PyrLKOpticalFlow

//IMPLEMENT_PARAM_CLASS(UseGray, bool)

PARAM_TEST_CASE(Sparse, bool, bool)
{
    bool useGray;
    bool UseSmart;

    virtual void SetUp()
    {
        UseSmart = GET_PARAM(0);
        useGray = GET_PARAM(0);
    }
};

TEST_P(Sparse, Mat)
{
    cv::Mat frame0 = readImage(workdir + "../gpu/rubberwhale1.png", useGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage(workdir + "../gpu/rubberwhale2.png", useGray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    ASSERT_FALSE(frame1.empty());

    cv::Mat gray_frame;
    if (useGray)
        gray_frame = frame0;
    else
        cv::cvtColor(frame0, gray_frame, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> pts;
    cv::goodFeaturesToTrack(gray_frame, pts, 1000, 0.01, 0.0);

    cv::ocl::oclMat d_pts;
    cv::Mat pts_mat(1, (int)pts.size(), CV_32FC2, (void *)&pts[0]);
    d_pts.upload(pts_mat);

    cv::ocl::PyrLKOpticalFlow pyrLK;

    cv::ocl::oclMat oclFrame0;
    cv::ocl::oclMat oclFrame1;
    cv::ocl::oclMat d_nextPts;
    cv::ocl::oclMat d_status;
    cv::ocl::oclMat d_err;

    oclFrame0 = frame0;
    oclFrame1 = frame1;

    pyrLK.sparse(oclFrame0, oclFrame1, d_pts, d_nextPts, d_status, &d_err);

    std::vector<cv::Point2f> nextPts(d_nextPts.cols);
    cv::Mat nextPts_mat(1, d_nextPts.cols, CV_32FC2, (void *)&nextPts[0]);
    d_nextPts.download(nextPts_mat);

    std::vector<unsigned char> status(d_status.cols);
    cv::Mat status_mat(1, d_status.cols, CV_8UC1, (void *)&status[0]);
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
            //float errdiff = std::abs(err[i] - err_gold[i]);
            float errdiff = 0.0f;

            if (!eq || errdiff > 1e-1)
                ++mistmatch;
        }
    }

    double bad_ratio = static_cast<double>(mistmatch) / (nextPts.size());

    ASSERT_LE(bad_ratio, 0.02f);

}

INSTANTIATE_TEST_CASE_P(Video, Sparse, Combine(
                            Values(false, true),
                            Values(false)));

#endif // HAVE_OPENCL

