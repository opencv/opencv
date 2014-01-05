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

#include "test_precomp.hpp"

#ifdef HAVE_OPENCL

////////////////////////////////////////////////////////
// ORB

namespace
{
    IMPLEMENT_PARAM_CLASS(ORB_FeaturesCount, int)
    IMPLEMENT_PARAM_CLASS(ORB_ScaleFactor, float)
    IMPLEMENT_PARAM_CLASS(ORB_LevelsCount, int)
    IMPLEMENT_PARAM_CLASS(ORB_EdgeThreshold, int)
    IMPLEMENT_PARAM_CLASS(ORB_firstLevel, int)
    IMPLEMENT_PARAM_CLASS(ORB_WTA_K, int)
    IMPLEMENT_PARAM_CLASS(ORB_PatchSize, int)
    IMPLEMENT_PARAM_CLASS(ORB_BlurForDescriptor, bool)
}

CV_ENUM(ORB_ScoreType, ORB::HARRIS_SCORE, ORB::FAST_SCORE)

PARAM_TEST_CASE(ORB, ORB_FeaturesCount, ORB_ScaleFactor, ORB_LevelsCount, ORB_EdgeThreshold,
                ORB_firstLevel, ORB_WTA_K, ORB_ScoreType, ORB_PatchSize, ORB_BlurForDescriptor)
{
    int nFeatures;
    float scaleFactor;
    int nLevels;
    int edgeThreshold;
    int firstLevel;
    int WTA_K;
    int scoreType;
    int patchSize;
    bool blurForDescriptor;

    virtual void SetUp()
    {
        nFeatures = GET_PARAM(0);
        scaleFactor = GET_PARAM(1);
        nLevels = GET_PARAM(2);
        edgeThreshold = GET_PARAM(3);
        firstLevel = GET_PARAM(4);
        WTA_K = GET_PARAM(5);
        scoreType = GET_PARAM(6);
        patchSize = GET_PARAM(7);
        blurForDescriptor = GET_PARAM(8);
    }
};

OCL_TEST_P(ORB, Accuracy)
{
    cv::Mat image = readImage("gpu/perf/aloe.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar::all(1));
    mask(cv::Range(0, image.rows / 2), cv::Range(0, image.cols / 2)).setTo(cv::Scalar::all(0));

    cv::ocl::oclMat ocl_image = cv::ocl::oclMat(image);
    cv::ocl::oclMat ocl_mask = cv::ocl::oclMat(mask);

    cv::ocl::ORB_OCL orb(nFeatures, scaleFactor, nLevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize);
    orb.blurForDescriptor = blurForDescriptor;

    std::vector<cv::KeyPoint> keypoints;
    cv::ocl::oclMat descriptors;
    orb(ocl_image, ocl_mask, keypoints, descriptors);

    cv::ORB orb_gold(nFeatures, scaleFactor, nLevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize);

    std::vector<cv::KeyPoint> keypoints_gold;
    cv::Mat descriptors_gold;
    orb_gold(image, mask, keypoints_gold, descriptors_gold);

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors_gold, cv::Mat(descriptors), matches);

    int matchedCount = getMatchedPointsCount(keypoints_gold, keypoints, matches);
    double matchedRatio = static_cast<double>(matchedCount) / keypoints.size();

    EXPECT_GT(matchedRatio, 0.35);
}

INSTANTIATE_TEST_CASE_P(OCL_Features2D, ORB,  testing::Combine(
                        testing::Values(ORB_FeaturesCount(1000)),
                        testing::Values(ORB_ScaleFactor(1.2f)),
                        testing::Values(ORB_LevelsCount(4), ORB_LevelsCount(8)),
                        testing::Values(ORB_EdgeThreshold(31)),
                        testing::Values(ORB_firstLevel(0), ORB_firstLevel(2)),
                        testing::Values(ORB_WTA_K(2), ORB_WTA_K(3), ORB_WTA_K(4)),
                        testing::Values(ORB_ScoreType(cv::ORB::HARRIS_SCORE)),
                        testing::Values(ORB_PatchSize(31), ORB_PatchSize(29)),
                        testing::Values(ORB_BlurForDescriptor(false), ORB_BlurForDescriptor(true))));

#endif
