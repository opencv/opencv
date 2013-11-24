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

PERF_TEST_P(OCL_SURF, DISABLED_with_data_transfer, testing::Values(SURF_IMAGES))
{
    string filename = getDataPath(GetParam());
    Mat img = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    SURF_OCL d_surf;
    oclMat d_keypoints;
    oclMat d_descriptors;
    Mat cpu_kp;
    Mat cpu_dp;

    declare.time(60);

    TEST_CYCLE()
    {
        oclMat d_src(img);

        d_surf(d_src, oclMat(), d_keypoints, d_descriptors);

        d_keypoints.download(cpu_kp);
        d_descriptors.download(cpu_dp);
    }

    SANITY_CHECK(cpu_kp, 1);
    SANITY_CHECK(cpu_dp, 1);
}

PERF_TEST_P(OCL_SURF, DISABLED_without_data_transfer, testing::Values(SURF_IMAGES))
{
    string filename = getDataPath(GetParam());
    Mat img = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    SURF_OCL d_surf;
    oclMat d_keypoints;
    oclMat d_descriptors;
    oclMat d_src(img);

    declare.time(60);

    TEST_CYCLE() d_surf(d_src, oclMat(), d_keypoints, d_descriptors);

    Mat cpu_kp;
    Mat cpu_dp;
    d_keypoints.download(cpu_kp);
    d_descriptors.download(cpu_dp);
    SANITY_CHECK(cpu_kp, 1);
    SANITY_CHECK(cpu_dp, 1);
}

#endif // HAVE_OPENCV_OCL
