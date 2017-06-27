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


#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"


#ifdef HAVE_OPENCL


namespace cvtest {
namespace ocl {

/////////////////////////////////////////////////////////////////////////////////////////////////
// PyrLKOpticalFlow

PARAM_TEST_CASE(PyrLKOpticalFlow, int, int)
{
    Size winSize;
    int maxLevel;
    TermCriteria criteria;
    int flags;
    double minEigThreshold;

    virtual void SetUp()
    {
        winSize = Size(GET_PARAM(0), GET_PARAM(0));
        maxLevel = GET_PARAM(1);
        criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
        flags = 0;
        minEigThreshold = 1e-4f;
    }
};

OCL_TEST_P(PyrLKOpticalFlow, Mat)
{
    static const int npoints = 1000;
    static const float eps = 0.03f;
    static const float erreps = 0.1f;

    cv::Mat frame0 = readImage("optflow/RubberWhale1.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());
    UMat umatFrame0; frame0.copyTo(umatFrame0);

    cv::Mat frame1 = readImage("optflow/RubberWhale2.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());
    UMat umatFrame1; frame1.copyTo(umatFrame1);

    std::vector<cv::Point2f> pts;
    cv::goodFeaturesToTrack(frame0, pts, npoints, 0.01, 0.0);

    std::vector<cv::Point2f> cpuNextPts;
    std::vector<unsigned char> cpuStatusCPU;
    std::vector<float> cpuErr;
    OCL_OFF(cv::calcOpticalFlowPyrLK(frame0, frame1, pts, cpuNextPts, cpuStatusCPU, cpuErr, winSize, maxLevel, criteria, flags, minEigThreshold));

    UMat umatNextPts, umatStatus, umatErr;
    OCL_ON(cv::calcOpticalFlowPyrLK(umatFrame0, umatFrame1, pts, umatNextPts, umatStatus, umatErr, winSize, maxLevel, criteria, flags, minEigThreshold));
    std::vector<cv::Point2f> nextPts; umatNextPts.reshape(2, 1).copyTo(nextPts);
    std::vector<unsigned char> status; umatStatus.reshape(1, 1).copyTo(status);
    std::vector<float> err; umatErr.reshape(1, 1).copyTo(err);

    ASSERT_EQ(cpuNextPts.size(), nextPts.size());
    ASSERT_EQ(cpuStatusCPU.size(), status.size());

    size_t mistmatch = 0;
    size_t errmatch = 0;

    for (size_t i = 0; i < nextPts.size(); ++i)
    {
        if (status[i] != cpuStatusCPU[i])
        {
            ++mistmatch;
            continue;
        }

        if (status[i])
        {
            cv::Point2i a = nextPts[i];
            cv::Point2i b = cpuNextPts[i];

            bool eq = std::abs(a.x - b.x) < 1 && std::abs(a.y - b.y) < 1;
            float errdiff = 0.0f;

            if (!eq || errdiff > 1e-1)
            {
                ++mistmatch;
                continue;
            }

            eq = std::abs(cpuErr[i] - err[i]) < 0.01;
            if(!eq)
                ++errmatch;
        }
    }

    double bad_ratio = static_cast<double>(mistmatch) / (nextPts.size());
    double err_ratio = static_cast<double>(errmatch) / (nextPts.size());

    ASSERT_LE(bad_ratio, eps);
    ASSERT_LE(err_ratio, erreps);
}

OCL_INSTANTIATE_TEST_CASE_P(Video, PyrLKOpticalFlow,
                            Combine(
                                Values(11, 15, 21, 25),
                                Values(3, 5)
                                )
                           );

} } // namespace cvtest::ocl


#endif // HAVE_OPENCL