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

using namespace cv;
using namespace cv::gpu;
using namespace cvtest;

TEST(projectPoints, accuracy)
{
    RNG& rng = TS::ptr()->get_rng();
    Mat src = randomMat(rng, Size(1000, 1), CV_32FC3, 0, 10, false);
    Mat rvec = randomMat(rng, Size(3, 1), CV_32F, 0, 1, false);
    Mat tvec = randomMat(rng, Size(3, 1), CV_32F, 0, 1, false);
    Mat camera_mat = randomMat(rng, Size(3, 3), CV_32F, 0, 1, false);
    camera_mat.at<float>(0, 1) = 0.f;
    camera_mat.at<float>(1, 0) = 0.f;
    camera_mat.at<float>(2, 0) = 0.f;
    camera_mat.at<float>(2, 1) = 0.f;

    vector<Point2f> dst;
    projectPoints(src, rvec, tvec, camera_mat, Mat(), dst);

    GpuMat d_dst;
    projectPoints(GpuMat(src), rvec, tvec, camera_mat, Mat(), d_dst);

    ASSERT_EQ(dst.size(), (size_t)d_dst.cols);
    ASSERT_EQ(1, d_dst.rows);
    ASSERT_EQ(CV_32FC2, d_dst.type());

    Mat h_dst(d_dst);
    for (size_t i = 0; i < dst.size(); ++i)
    {
        Point2f res_gold = dst[i];
        Point2f res_actual = h_dst.at<Point2f>(0, i);
        Point2f err = res_actual - res_gold;
        ASSERT_LT(err.dot(err) / res_gold.dot(res_gold), 1e-3f);
    }
}


TEST(transformPoints, accuracy)
{
    RNG& rng = TS::ptr()->get_rng();
    Mat src = randomMat(rng, Size(1000, 1), CV_32FC3, 0, 10, false);
    Mat rvec = randomMat(rng, Size(3, 1), CV_32F, 0, 1, false);
    Mat tvec = randomMat(rng, Size(3, 1), CV_32F, 0, 1, false);

    GpuMat d_dst;
    transformPoints(GpuMat(src), rvec, tvec, d_dst);
    ASSERT_TRUE(src.size() == d_dst.size());
    ASSERT_EQ(src.type(), d_dst.type());

    Mat h_dst(d_dst);
    Mat rot;
    Rodrigues(rvec, rot);
    for (int i = 0; i < h_dst.cols; ++i)
    {
        Point3f p = src.at<Point3f>(0, i);
        Point3f res_gold(
                rot.at<float>(0, 0) * p.x + rot.at<float>(0, 1) * p.y + rot.at<float>(0, 2) * p.z + tvec.at<float>(0, 0),
                rot.at<float>(1, 0) * p.x + rot.at<float>(1, 1) * p.y + rot.at<float>(1, 2) * p.z + tvec.at<float>(0, 1),
                rot.at<float>(2, 0) * p.x + rot.at<float>(2, 1) * p.y + rot.at<float>(2, 2) * p.z + tvec.at<float>(0, 2));
        Point3f res_actual = h_dst.at<Point3f>(0, i);
        Point3f err = res_actual - res_gold;
        ASSERT_LT(err.dot(err) / res_gold.dot(res_gold), 1e-3f);
    }
}


TEST(solvePnPRansac, accuracy)
{
    RNG& rng = TS::ptr()->get_rng();

    const int num_points = 5000;
    Mat object = randomMat(rng, Size(num_points, 1), CV_32FC3, 0, 100, false);
    Mat camera_mat = randomMat(rng, Size(3, 3), CV_32F, 0.5, 1, false);
    camera_mat.at<float>(0, 1) = 0.f;
    camera_mat.at<float>(1, 0) = 0.f;
    camera_mat.at<float>(2, 0) = 0.f;
    camera_mat.at<float>(2, 1) = 0.f;

    Mat rvec_gold = randomMat(rng, Size(3, 1), CV_32F, 0, 1, false);
    Mat tvec_gold = randomMat(rng, Size(3, 1), CV_32F, 0, 1, false);

    vector<Point2f> image_vec;
    projectPoints(object, rvec_gold, tvec_gold, camera_mat, Mat(), image_vec);
    Mat image(1, image_vec.size(), CV_32FC2, &image_vec[0]);

    Mat rvec, tvec;
    vector<int> inliers;
    gpu::solvePnPRansac(object, image, camera_mat, Mat(), rvec, tvec, false, 200, 2.f, 100, &inliers);

    ASSERT_LE(norm(rvec - rvec_gold), 1e-3f);
    ASSERT_LE(norm(tvec - tvec_gold), 1e-3f);
}
