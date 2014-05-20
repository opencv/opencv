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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
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
// This software is provided by the copyright holders and contributors as is and
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

#include "perf_precomp.hpp"

#ifdef HAVE_OPENCV_OCL

using namespace cv;
using namespace cv::ocl;
using namespace std;

typedef perf::TestBaseWithParam<std::string> OCL_SURF;

#define SURF_IMAGES \
    "cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png",\
    "stitching/a3.png"

#define OCL_TEST_CYCLE() for( ; startTimer(), next(); cv::ocl::finish(), stopTimer())

PERF_TEST_P(OCL_SURF, DISABLED_with_data_transfer, testing::Values(SURF_IMAGES))
{
    string filename = getDataPath(GetParam());
    Mat src = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(src.empty());

    Mat cpu_kp, cpu_dp;
    declare.time(60);

    if (getSelectedImpl() == "ocl")
    {
        SURF_OCL d_surf;
        oclMat d_keypoints, d_descriptors;

        OCL_TEST_CYCLE()
        {
            oclMat d_src(src);

            d_surf(d_src, oclMat(), d_keypoints, d_descriptors);

            d_keypoints.download(cpu_kp);
            d_descriptors.download(cpu_dp);
        }
    }
    else if (getSelectedImpl() == "plain")
    {
        cv::SURF surf;
        std::vector<cv::KeyPoint> kp;

        TEST_CYCLE() surf(src, Mat(), kp, cpu_dp);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(OCL_SURF, DISABLED_without_data_transfer, testing::Values(SURF_IMAGES))
{
    string filename = getDataPath(GetParam());
    Mat src = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(src.empty());

    Mat cpu_kp, cpu_dp;
    declare.time(60);

    if (getSelectedImpl() == "ocl")
    {
        SURF_OCL d_surf;
        oclMat d_keypoints, d_descriptors, d_src(src);

        OCL_TEST_CYCLE() d_surf(d_src, oclMat(), d_keypoints, d_descriptors);
    }
    else if (getSelectedImpl() == "plain")
    {
        cv::SURF surf;
        std::vector<cv::KeyPoint> kp;

        TEST_CYCLE() surf(src, Mat(), kp, cpu_dp);
    }

    SANITY_CHECK_NOTHING();
}



PERF_TEST_P(OCL_SURF, DISABLED_detect, testing::Values(SURF_IMAGES))
{
    String filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame.empty()) << "Unable to load source image " << filename;

    declare.in(frame);

    Mat mask;
    vector<KeyPoint> points;
    Ptr<Feature2D> detector;

    if (getSelectedImpl() == "plain")
    {
        detector = new SURF;
        TEST_CYCLE() detector->operator()(frame, mask, points, noArray());
    }
    else if (getSelectedImpl() == "ocl")
    {
        detector = new ocl::SURF_OCL;
        OCL_TEST_CYCLE() detector->operator()(frame, mask, points, noArray());
    }
    else CV_TEST_FAIL_NO_IMPL();

    SANITY_CHECK_KEYPOINTS(points, 1e-3);
}

PERF_TEST_P(OCL_SURF, DISABLED_extract, testing::Values(SURF_IMAGES))
{
    String filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame.empty()) << "Unable to load source image " << filename;

    declare.in(frame);

    Mat mask;
    Ptr<Feature2D> detector;
    vector<KeyPoint> points;
    vector<float> descriptors;

    if (getSelectedImpl() == "plain")
    {
        detector = new SURF;
        detector->operator()(frame, mask, points, noArray());
        TEST_CYCLE() detector->operator()(frame, mask, points, descriptors, true);
    }
    else if (getSelectedImpl() == "ocl")
    {
        detector = new ocl::SURF_OCL;
        detector->operator()(frame, mask, points, noArray());
        OCL_TEST_CYCLE() detector->operator()(frame, mask, points, descriptors, true);
    }
    else CV_TEST_FAIL_NO_IMPL();

    SANITY_CHECK(descriptors, 1e-4);
}

PERF_TEST_P(OCL_SURF, DISABLED_full, testing::Values(SURF_IMAGES))
{
    String filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame.empty()) << "Unable to load source image " << filename;

    declare.in(frame).time(90);

    Mat mask;
    Ptr<Feature2D> detector;
    vector<KeyPoint> points;
    vector<float> descriptors;

    if (getSelectedImpl() == "plain")
    {
        detector = new SURF;
        TEST_CYCLE() detector->operator()(frame, mask, points, descriptors, false);
    }
    else if (getSelectedImpl() == "ocl")
    {
        detector = new ocl::SURF_OCL;
        detector->operator()(frame, mask, points, noArray());
        OCL_TEST_CYCLE() detector->operator()(frame, mask, points, descriptors, false);
    }
    else CV_TEST_FAIL_NO_IMPL();

    SANITY_CHECK_KEYPOINTS(points, 1e-3);
    SANITY_CHECK(descriptors, 1e-4);
}

#endif // HAVE_OPENCV_OCL
