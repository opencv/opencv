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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// Authors:
//  * Peter Andreas Entschev, peter@entschev.com
//
//M*/

#include "perf_precomp.hpp"

using namespace perf;

/////////////////// ORB ///////////////////

typedef std::tr1::tuple<std::string, int> Image_NFeatures_t;
typedef perf::TestBaseWithParam<Image_NFeatures_t> Image_NFeatures;

PERF_TEST_P(Image_NFeatures, ORB,
            testing::Combine(testing::Values<string>("gpu/perf/aloe.png"),
                             testing::Values(4000)))
{
    declare.time(300.0);

    const Image_NFeatures_t params = GetParam();
    const std::string imgFile = std::tr1::get<0>(params);
    const int nFeatures = std::tr1::get<1>(params);

    const cv::Mat img = imread(getDataPath(imgFile), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    if (RUN_OCL_IMPL)
    {
        cv::ocl::ORB_OCL d_orb(nFeatures);

        const cv::ocl::oclMat d_img(img);
        cv::ocl::oclMat d_keypoints, d_descriptors;

        TEST_CYCLE() d_orb(d_img, cv::ocl::oclMat(), d_keypoints, d_descriptors);

        std::vector<cv::KeyPoint> ocl_keypoints;
        d_orb.downloadKeyPoints(d_keypoints, ocl_keypoints);

        cv::Mat ocl_descriptors(d_descriptors);

        ocl_keypoints.resize(10);
        ocl_descriptors = ocl_descriptors.rowRange(0, 10);

        sortKeyPoints(ocl_keypoints, ocl_descriptors);

        SANITY_CHECK_KEYPOINTS(ocl_keypoints, 1e-4);
        SANITY_CHECK(ocl_descriptors);
    }
    else if (RUN_PLAIN_IMPL)
    {
        cv::ORB orb(nFeatures);

        std::vector<cv::KeyPoint> cpu_keypoints;
        cv::Mat cpu_descriptors;

        TEST_CYCLE() orb(img, cv::noArray(), cpu_keypoints, cpu_descriptors);

        SANITY_CHECK_KEYPOINTS(cpu_keypoints);
        SANITY_CHECK(cpu_descriptors);
    }
    else
        OCL_PERF_ELSE;
}
