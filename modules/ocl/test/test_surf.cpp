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
//     and/or other oclMaterials provided with the distribution.
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


#include "precomp.hpp"
#ifdef HAVE_OPENCL

extern std::string workdir;

using namespace std;

static bool keyPointsEquals(const cv::KeyPoint& p1, const cv::KeyPoint& p2)
{
    const double maxPtDif = 1.0;
    const double maxSizeDif = 1.0;
    const double maxAngleDif = 2.0;
    const double maxResponseDif = 0.1;

    double dist = cv::norm(p1.pt - p2.pt);

    if (dist < maxPtDif &&
        fabs(p1.size - p2.size) < maxSizeDif &&
        abs(p1.angle - p2.angle) < maxAngleDif &&
        abs(p1.response - p2.response) < maxResponseDif &&
        p1.octave == p2.octave &&
        p1.class_id == p2.class_id)
    {
        return true;
    }

    return false;
}


struct KeyPointLess : std::binary_function<cv::KeyPoint, cv::KeyPoint, bool>
{
    bool operator()(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2) const
    {
        return kp1.pt.y < kp2.pt.y || (kp1.pt.y == kp2.pt.y && kp1.pt.x < kp2.pt.x);
    }
};


#define ASSERT_KEYPOINTS_EQ(gold, actual) EXPECT_PRED_FORMAT2(assertKeyPointsEquals, gold, actual);

static int getMatchedPointsCount(std::vector<cv::KeyPoint>& gold, std::vector<cv::KeyPoint>& actual)
{
    std::sort(actual.begin(), actual.end(), KeyPointLess());
    std::sort(gold.begin(), gold.end(), KeyPointLess());

    int validCount = 0;

    for (size_t i = 0; i < gold.size(); ++i)
    {
        const cv::KeyPoint& p1 = gold[i];
        const cv::KeyPoint& p2 = actual[i];

        if (keyPointsEquals(p1, p2))
            ++validCount;
    }

    return validCount;
}

static int getMatchedPointsCount(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2, const std::vector<cv::DMatch>& matches)
{
    int validCount = 0;

    for (size_t i = 0; i < matches.size(); ++i)
    {
        const cv::DMatch& m = matches[i];

        const cv::KeyPoint& p1 = keypoints1[m.queryIdx];
        const cv::KeyPoint& p2 = keypoints2[m.trainIdx];

        if (keyPointsEquals(p1, p2))
            ++validCount;
    }

    return validCount;
}

IMPLEMENT_PARAM_CLASS(SURF_HessianThreshold, double)
IMPLEMENT_PARAM_CLASS(SURF_Octaves, int)
IMPLEMENT_PARAM_CLASS(SURF_OctaveLayers, int)
IMPLEMENT_PARAM_CLASS(SURF_Extended, bool)
IMPLEMENT_PARAM_CLASS(SURF_Upright, bool)

PARAM_TEST_CASE(SURF, SURF_HessianThreshold, SURF_Octaves, SURF_OctaveLayers, SURF_Extended, SURF_Upright)
{
    double hessianThreshold;
    int nOctaves;
    int nOctaveLayers;
    bool extended;
    bool upright;

    virtual void SetUp()
    {
        hessianThreshold = GET_PARAM(0);
        nOctaves = GET_PARAM(1);
        nOctaveLayers = GET_PARAM(2);
        extended = GET_PARAM(3);
        upright = GET_PARAM(4);
    }
};
TEST_P(SURF, Detector)
{
    cv::Mat image = readImage(workdir + "fruits.jpg", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    cv::ocl::SURF_OCL surf;
    surf.hessianThreshold = static_cast<float>(hessianThreshold);
    surf.nOctaves = nOctaves;
    surf.nOctaveLayers = nOctaveLayers;
    surf.extended = extended;
    surf.upright = upright;
    surf.keypointsRatio = 0.05f;

    std::vector<cv::KeyPoint> keypoints;
    surf(cv::ocl::oclMat(image), cv::ocl::oclMat(), keypoints);

    cv::SURF surf_gold;
    surf_gold.hessianThreshold = hessianThreshold;
    surf_gold.nOctaves = nOctaves;
    surf_gold.nOctaveLayers = nOctaveLayers;
    surf_gold.extended = extended;
    surf_gold.upright = upright;

    std::vector<cv::KeyPoint> keypoints_gold;
    surf_gold(image, cv::noArray(), keypoints_gold);

    ASSERT_EQ(keypoints_gold.size(), keypoints.size());
    int matchedCount = getMatchedPointsCount(keypoints_gold, keypoints);
    double matchedRatio = static_cast<double>(matchedCount) / keypoints_gold.size();

    EXPECT_GT(matchedRatio, 0.95);
}

TEST_P(SURF, Descriptor)
{
    cv::Mat image = readImage(workdir + "fruits.jpg", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    cv::ocl::SURF_OCL surf;
    surf.hessianThreshold = static_cast<float>(hessianThreshold);
    surf.nOctaves = nOctaves;
    surf.nOctaveLayers = nOctaveLayers;
    surf.extended = extended;
    surf.upright = upright;
    surf.keypointsRatio = 0.05f;

    cv::SURF surf_gold;
    surf_gold.hessianThreshold = hessianThreshold;
    surf_gold.nOctaves = nOctaves;
    surf_gold.nOctaveLayers = nOctaveLayers;
    surf_gold.extended = extended;
    surf_gold.upright = upright;

    std::vector<cv::KeyPoint> keypoints;
    surf_gold(image, cv::noArray(), keypoints);

    cv::ocl::oclMat descriptors;
    surf(cv::ocl::oclMat(image), cv::ocl::oclMat(), keypoints, descriptors, true);

    cv::Mat descriptors_gold;
    surf_gold(image, cv::noArray(), keypoints, descriptors_gold, true);

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors_gold, cv::Mat(descriptors), matches);

    int matchedCount = getMatchedPointsCount(keypoints, keypoints, matches);
    double matchedRatio = static_cast<double>(matchedCount) / keypoints.size();

    EXPECT_GT(matchedRatio, 0.35);
}

INSTANTIATE_TEST_CASE_P(OCL_Features2D, SURF, testing::Combine(
    testing::Values(/*SURF_HessianThreshold(100.0), */SURF_HessianThreshold(500.0), SURF_HessianThreshold(1000.0)),
    testing::Values(SURF_Octaves(3), SURF_Octaves(4)),
    testing::Values(SURF_OctaveLayers(2), SURF_OctaveLayers(3)),
    testing::Values(SURF_Extended(false), SURF_Extended(true)),
    testing::Values(SURF_Upright(false), SURF_Upright(true))));

#endif
