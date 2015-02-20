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

#if defined HAVE_CUDA && defined HAVE_OPENCV_CALIB3D

#include "opencv2/calib3d.hpp"

using namespace cvtest;

///////////////////////////////////////////////////////////////////////////////////////////////////////
// transformPoints

struct TransformPoints : testing::TestWithParam<cv::cuda::DeviceInfo>
{
    cv::cuda::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(TransformPoints, Accuracy)
{
    cv::Mat src = randomMat(cv::Size(1000, 1), CV_32FC3, 0, 10);
    cv::Mat rvec = randomMat(cv::Size(3, 1), CV_32F, 0, 1);
    cv::Mat tvec = randomMat(cv::Size(3, 1), CV_32F, 0, 1);

    cv::cuda::GpuMat dst;
    cv::cuda::transformPoints(loadMat(src), rvec, tvec, dst);

    ASSERT_EQ(src.size(), dst.size());
    ASSERT_EQ(src.type(), dst.type());

    cv::Mat h_dst(dst);

    cv::Mat rot;
    cv::Rodrigues(rvec, rot);

    for (int i = 0; i < h_dst.cols; ++i)
    {
        cv::Point3f res = h_dst.at<cv::Point3f>(0, i);

        cv::Point3f p = src.at<cv::Point3f>(0, i);
        cv::Point3f res_gold(
                rot.at<float>(0, 0) * p.x + rot.at<float>(0, 1) * p.y + rot.at<float>(0, 2) * p.z + tvec.at<float>(0, 0),
                rot.at<float>(1, 0) * p.x + rot.at<float>(1, 1) * p.y + rot.at<float>(1, 2) * p.z + tvec.at<float>(0, 1),
                rot.at<float>(2, 0) * p.x + rot.at<float>(2, 1) * p.y + rot.at<float>(2, 2) * p.z + tvec.at<float>(0, 2));

        ASSERT_POINT3_NEAR(res_gold, res, 1e-5);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Calib3D, TransformPoints, ALL_DEVICES);

///////////////////////////////////////////////////////////////////////////////////////////////////////
// ProjectPoints

struct ProjectPoints : testing::TestWithParam<cv::cuda::DeviceInfo>
{
    cv::cuda::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(ProjectPoints, Accuracy)
{
    cv::Mat src = randomMat(cv::Size(1000, 1), CV_32FC3, 0, 10);
    cv::Mat rvec = randomMat(cv::Size(3, 1), CV_32F, 0, 1);
    cv::Mat tvec = randomMat(cv::Size(3, 1), CV_32F, 0, 1);
    cv::Mat camera_mat = randomMat(cv::Size(3, 3), CV_32F, 0.5, 1);
    camera_mat.at<float>(0, 1) = 0.f;
    camera_mat.at<float>(1, 0) = 0.f;
    camera_mat.at<float>(2, 0) = 0.f;
    camera_mat.at<float>(2, 1) = 0.f;

    cv::cuda::GpuMat dst;
    cv::cuda::projectPoints(loadMat(src), rvec, tvec, camera_mat, cv::Mat(), dst);

    ASSERT_EQ(1, dst.rows);
    ASSERT_EQ(MatType(CV_32FC2), MatType(dst.type()));

    std::vector<cv::Point2f> dst_gold;
    cv::projectPoints(src, rvec, tvec, camera_mat, cv::Mat(1, 8, CV_32F, cv::Scalar::all(0)), dst_gold);

    ASSERT_EQ(dst_gold.size(), static_cast<size_t>(dst.cols));

    cv::Mat h_dst(dst);

    for (size_t i = 0; i < dst_gold.size(); ++i)
    {
        cv::Point2f res = h_dst.at<cv::Point2f>(0, (int)i);
        cv::Point2f res_gold = dst_gold[i];

        ASSERT_LE(cv::norm(res_gold - res) / cv::norm(res_gold), 1e-3f);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Calib3D, ProjectPoints, ALL_DEVICES);

///////////////////////////////////////////////////////////////////////////////////////////////////////
// SolvePnPRansac

struct SolvePnPRansac : testing::TestWithParam<cv::cuda::DeviceInfo>
{
    cv::cuda::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(SolvePnPRansac, Accuracy)
{
    cv::Mat object = randomMat(cv::Size(5000, 1), CV_32FC3, 0, 100);
    cv::Mat camera_mat = randomMat(cv::Size(3, 3), CV_32F, 0.5, 1);
    camera_mat.at<float>(0, 1) = 0.f;
    camera_mat.at<float>(1, 0) = 0.f;
    camera_mat.at<float>(2, 0) = 0.f;
    camera_mat.at<float>(2, 1) = 0.f;

    std::vector<cv::Point2f> image_vec;
    cv::Mat rvec_gold;
    cv::Mat tvec_gold;
    rvec_gold = randomMat(cv::Size(3, 1), CV_32F, 0, 1);
    tvec_gold = randomMat(cv::Size(3, 1), CV_32F, 0, 1);
    cv::projectPoints(object, rvec_gold, tvec_gold, camera_mat, cv::Mat(1, 8, CV_32F, cv::Scalar::all(0)), image_vec);

    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    cv::cuda::solvePnPRansac(object, cv::Mat(1, (int)image_vec.size(), CV_32FC2, &image_vec[0]),
                            camera_mat, cv::Mat(1, 8, CV_32F, cv::Scalar::all(0)),
                            rvec, tvec, false, 200, 2.f, 100, &inliers);

    ASSERT_LE(cv::norm(rvec - rvec_gold), 1e-3);
    ASSERT_LE(cv::norm(tvec - tvec_gold), 1e-3);
}

INSTANTIATE_TEST_CASE_P(CUDA_Calib3D, SolvePnPRansac, ALL_DEVICES);

#endif // HAVE_CUDA
