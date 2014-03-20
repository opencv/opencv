///////////////////////////////////////////////////////////////////////////////////////
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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
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
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

//////////////////////////// GoodFeaturesToTrack //////////////////////////


PARAM_TEST_CASE(GoodFeaturesToTrack, double, bool)
{
    double minDistance;
    bool useRoi;

    static const int maxCorners;
    static const double qualityLevel;

    TEST_DECLARE_INPUT_PARAMETER(src);
    UMat points, upoints;

    virtual void SetUp()
    {
        minDistance = GET_PARAM(0);
        useRoi = GET_PARAM(1);
    }

    void generateTestData()
    {
        Mat frame = readImage("../gpu/opticalflow/rubberwhale1.png", IMREAD_GRAYSCALE);
        ASSERT_FALSE(frame.empty()) << "could not load gpu/opticalflow/rubberwhale1.png";

        Size roiSize = frame.size();
        Border srcBorder = randomBorder(0, useRoi ? 2 : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, frame.type(), 5, 256);
        src_roi.copyTo(frame);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
    }

    void UMatToVector(const UMat & um, std::vector<Point2f> & v) const
    {
        v.resize(um.size().area());
        um.copyTo(Mat(um.size(), CV_32FC2, &v[0]));
    }
};

const int GoodFeaturesToTrack::maxCorners = 1000;
const double GoodFeaturesToTrack::qualityLevel = 0.01;

OCL_TEST_P(GoodFeaturesToTrack, Accuracy)
{
    for (int j = 0; j < test_loop_times; ++j)
    {
        generateTestData();

        std::vector<Point2f> upts, pts;

        OCL_OFF(cv::goodFeaturesToTrack(src_roi, points, maxCorners, qualityLevel, minDistance, noArray()));
        ASSERT_FALSE(points.empty());
        UMatToVector(points, pts);

        OCL_ON(cv::goodFeaturesToTrack(usrc_roi, upoints, maxCorners, qualityLevel, minDistance));
        ASSERT_FALSE(upoints.empty());
        UMatToVector(upoints, upts);

        ASSERT_EQ(upts.size(), pts.size());

        int mistmatch = 0;
        for (size_t i = 0; i < pts.size(); ++i)
        {
            Point2i a = upts[i], b = pts[i];

            bool eq = std::abs(a.x - b.x) < 1 && std::abs(a.y - b.y) < 1;

            if (!eq)
                ++mistmatch;
        }

        double bad_ratio = static_cast<double>(mistmatch) / pts.size();
        ASSERT_GE(1e-2, bad_ratio);
    }
}

OCL_TEST_P(GoodFeaturesToTrack, EmptyCorners)
{
    generateTestData();
    usrc_roi.setTo(Scalar::all(0));

    OCL_ON(cv::goodFeaturesToTrack(usrc_roi, upoints, maxCorners, qualityLevel, minDistance));

    ASSERT_TRUE(upoints.empty());
}

OCL_INSTANTIATE_TEST_CASE_P(Imgproc, GoodFeaturesToTrack,
                            ::testing::Combine(testing::Values(0.0, 3.0), Bool()));

} } // namespace cvtest::ocl

#endif
