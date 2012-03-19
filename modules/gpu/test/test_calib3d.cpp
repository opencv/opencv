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

#ifdef HAVE_CUDA

using namespace cvtest;
using namespace testing;

//////////////////////////////////////////////////////////////////////////
// BlockMatching

struct StereoBlockMatching : TestWithParam<cv::gpu::DeviceInfo>
{
    cv::Mat img_l;
    cv::Mat img_r;
    cv::Mat img_template;

    cv::gpu::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        img_l = readImage("stereobm/aloe-L.png", CV_LOAD_IMAGE_GRAYSCALE);
        img_r = readImage("stereobm/aloe-R.png", CV_LOAD_IMAGE_GRAYSCALE);
        img_template = readImage("stereobm/aloe-disp.png", CV_LOAD_IMAGE_GRAYSCALE);

        ASSERT_FALSE(img_l.empty());
        ASSERT_FALSE(img_r.empty());
        ASSERT_FALSE(img_template.empty());
    }
};

TEST_P(StereoBlockMatching, Regression)
{
    cv::Mat disp;

    cv::gpu::GpuMat dev_disp;
    cv::gpu::StereoBM_GPU bm(0, 128, 19);

    bm(cv::gpu::GpuMat(img_l), cv::gpu::GpuMat(img_r), dev_disp);

    dev_disp.download(disp);

    disp.convertTo(disp, img_template.type());

    EXPECT_MAT_NEAR(img_template, disp, 0.0);
}

INSTANTIATE_TEST_CASE_P(Calib3D, StereoBlockMatching, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////////
// BeliefPropagation

struct StereoBeliefPropagation : TestWithParam<cv::gpu::DeviceInfo>
{
    cv::Mat img_l;
    cv::Mat img_r;
    cv::Mat img_template;

    cv::gpu::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        img_l = readImage("stereobp/aloe-L.png");
        img_r = readImage("stereobp/aloe-R.png");
        img_template = readImage("stereobp/aloe-disp.png", CV_LOAD_IMAGE_GRAYSCALE);

        ASSERT_FALSE(img_l.empty());
        ASSERT_FALSE(img_r.empty());
        ASSERT_FALSE(img_template.empty());
    }
};

TEST_P(StereoBeliefPropagation, Regression)
{
    cv::Mat disp;

    cv::gpu::GpuMat dev_disp;
    cv::gpu::StereoBeliefPropagation bpm(64, 8, 2, 25, 0.1f, 15, 1, CV_16S);

    bpm(cv::gpu::GpuMat(img_l), cv::gpu::GpuMat(img_r), dev_disp);

    dev_disp.download(disp);

    disp.convertTo(disp, img_template.type());

    EXPECT_MAT_NEAR(img_template, disp, 0.0);
}

INSTANTIATE_TEST_CASE_P(Calib3D, StereoBeliefPropagation, ALL_DEVICES);

//////////////////////////////////////////////////////////////////////////
// ConstantSpaceBP

struct StereoConstantSpaceBP : TestWithParam<cv::gpu::DeviceInfo>
{
    cv::Mat img_l;
    cv::Mat img_r;
    cv::Mat img_template;

    cv::gpu::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        img_l = readImage("csstereobp/aloe-L.png");
        img_r = readImage("csstereobp/aloe-R.png");

        if (supportFeature(devInfo, cv::gpu::FEATURE_SET_COMPUTE_20))
            img_template = readImage("csstereobp/aloe-disp.png", CV_LOAD_IMAGE_GRAYSCALE);
        else
            img_template = readImage("csstereobp/aloe-disp_CC1X.png", CV_LOAD_IMAGE_GRAYSCALE);

        ASSERT_FALSE(img_l.empty());
        ASSERT_FALSE(img_r.empty());
        ASSERT_FALSE(img_template.empty());
    }
};

TEST_P(StereoConstantSpaceBP, Regression)
{
    cv::Mat disp;

    cv::gpu::GpuMat dev_disp;
    cv::gpu::StereoConstantSpaceBP bpm(128, 16, 4, 4);

    bpm(cv::gpu::GpuMat(img_l), cv::gpu::GpuMat(img_r), dev_disp);

    dev_disp.download(disp);

    disp.convertTo(disp, img_template.type());

    EXPECT_MAT_NEAR(img_template, disp, 1.0);
}

INSTANTIATE_TEST_CASE_P(Calib3D, StereoConstantSpaceBP, ALL_DEVICES);

///////////////////////////////////////////////////////////////////////////////////////////////////////
// projectPoints

struct ProjectPoints : TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Mat src;
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat camera_mat;

    std::vector<cv::Point2f> dst_gold;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        src = cvtest::randomMat(rng, cv::Size(1000, 1), CV_32FC3, 0, 10, false);
        rvec = cvtest::randomMat(rng, cv::Size(3, 1), CV_32F, 0, 1, false);
        tvec = cvtest::randomMat(rng, cv::Size(3, 1), CV_32F, 0, 1, false);
        camera_mat = cvtest::randomMat(rng, cv::Size(3, 3), CV_32F, 0, 1, false);
        camera_mat.at<float>(0, 1) = 0.f;
        camera_mat.at<float>(1, 0) = 0.f;
        camera_mat.at<float>(2, 0) = 0.f;
        camera_mat.at<float>(2, 1) = 0.f;

        cv::projectPoints(src, rvec, tvec, camera_mat, cv::Mat(1, 8, CV_32F, cv::Scalar::all(0)), dst_gold);
    }
};

TEST_P(ProjectPoints, Accuracy)
{
    cv::Mat dst;

    cv::gpu::GpuMat d_dst;

    cv::gpu::projectPoints(cv::gpu::GpuMat(src), rvec, tvec, camera_mat, cv::Mat(), d_dst);

    d_dst.download(dst);

    ASSERT_EQ(dst_gold.size(), static_cast<size_t>(dst.cols));
    ASSERT_EQ(1, dst.rows);
    ASSERT_EQ(CV_32FC2, dst.type());

    for (size_t i = 0; i < dst_gold.size(); ++i)
    {
        cv::Point2f res_gold = dst_gold[i];
        cv::Point2f res_actual = dst.at<cv::Point2f>(0, i);
        cv::Point2f err = res_actual - res_gold;

        ASSERT_LE(err.dot(err) / res_gold.dot(res_gold), 1e-3f);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, ProjectPoints, ALL_DEVICES);

///////////////////////////////////////////////////////////////////////////////////////////////////////
// transformPoints

struct TransformPoints : TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    cv::Mat src;
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat rot;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        src = cvtest::randomMat(rng, cv::Size(1000, 1), CV_32FC3, 0, 10, false);
        rvec = cvtest::randomMat(rng, cv::Size(3, 1), CV_32F, 0, 1, false);
        tvec = cvtest::randomMat(rng, cv::Size(3, 1), CV_32F, 0, 1, false);

        cv::Rodrigues(rvec, rot);
    }
};

TEST_P(TransformPoints, Accuracy)
{
    cv::Mat dst;

    cv::gpu::GpuMat d_dst;

    cv::gpu::transformPoints(cv::gpu::GpuMat(src), rvec, tvec, d_dst);

    d_dst.download(dst);

    ASSERT_EQ(src.size(), dst.size());
    ASSERT_EQ(src.type(), dst.type());

    for (int i = 0; i < dst.cols; ++i)
    {
        cv::Point3f p = src.at<cv::Point3f>(0, i);
        cv::Point3f res_gold(
                rot.at<float>(0, 0) * p.x + rot.at<float>(0, 1) * p.y + rot.at<float>(0, 2) * p.z + tvec.at<float>(0, 0),
                rot.at<float>(1, 0) * p.x + rot.at<float>(1, 1) * p.y + rot.at<float>(1, 2) * p.z + tvec.at<float>(0, 1),
                rot.at<float>(2, 0) * p.x + rot.at<float>(2, 1) * p.y + rot.at<float>(2, 2) * p.z + tvec.at<float>(0, 2));
        cv::Point3f res_actual = dst.at<cv::Point3f>(0, i);
        cv::Point3f err = res_actual - res_gold;

        ASSERT_LE(err.dot(err) / res_gold.dot(res_gold), 1e-3f);
    }
}

INSTANTIATE_TEST_CASE_P(Calib3D, TransformPoints, ALL_DEVICES);

///////////////////////////////////////////////////////////////////////////////////////////////////////
// solvePnPRansac

struct SolvePnPRansac : TestWithParam<cv::gpu::DeviceInfo>
{
    static const int num_points = 5000;

    cv::gpu::DeviceInfo devInfo;

    cv::Mat object;
    cv::Mat camera_mat;
    std::vector<cv::Point2f> image_vec;

    cv::Mat rvec_gold;
    cv::Mat tvec_gold;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        object = cvtest::randomMat(rng, cv::Size(num_points, 1), CV_32FC3, 0, 100, false);
        camera_mat = cvtest::randomMat(rng, cv::Size(3, 3), CV_32F, 0.5, 1, false);
        camera_mat.at<float>(0, 1) = 0.f;
        camera_mat.at<float>(1, 0) = 0.f;
        camera_mat.at<float>(2, 0) = 0.f;
        camera_mat.at<float>(2, 1) = 0.f;

        rvec_gold = cvtest::randomMat(rng, cv::Size(3, 1), CV_32F, 0, 1, false);
        tvec_gold = cvtest::randomMat(rng, cv::Size(3, 1), CV_32F, 0, 1, false);

        cv::projectPoints(object, rvec_gold, tvec_gold, camera_mat, cv::Mat(1, 8, CV_32F, cv::Scalar::all(0)), image_vec);
    }
};

TEST_P(SolvePnPRansac, Accuracy)
{
    cv::Mat rvec, tvec;
    std::vector<int> inliers;

    cv::gpu::solvePnPRansac(object, cv::Mat(1, image_vec.size(), CV_32FC2, &image_vec[0]), camera_mat,
                            cv::Mat(1, 8, CV_32F, cv::Scalar::all(0)), rvec, tvec, false, 200, 2.f, 100, &inliers);

    ASSERT_LE(cv::norm(rvec - rvec_gold), 1e-3f);
    ASSERT_LE(cv::norm(tvec - tvec_gold), 1e-3f);
}

INSTANTIATE_TEST_CASE_P(Calib3D, SolvePnPRansac, ALL_DEVICES);

#endif // HAVE_CUDA
