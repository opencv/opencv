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

namespace {

bool keyPointsEquals(const cv::KeyPoint& p1, const cv::KeyPoint& p2)
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

testing::AssertionResult assertKeyPointsEquals(const char* gold_expr, const char* actual_expr, std::vector<cv::KeyPoint>& gold, std::vector<cv::KeyPoint>& actual)
{
    if (gold.size() != actual.size())
    {
        return testing::AssertionFailure() << "KeyPoints size mistmach\n"
                                           << "\"" << gold_expr << "\" : " << gold.size() << "\n"
                                           << "\"" << actual_expr << "\" : " << actual.size();
    }

    std::sort(actual.begin(), actual.end(), KeyPointLess());
    std::sort(gold.begin(), gold.end(), KeyPointLess());

    for (size_t i = 0; i < gold.size(); ++i)
    {
        const cv::KeyPoint& p1 = gold[i];
        const cv::KeyPoint& p2 = actual[i];

        if (!keyPointsEquals(p1, p2))
        {
            return testing::AssertionFailure() << "KeyPoints differ at " << i << "\n"
                                               << "\"" << gold_expr << "\" vs \"" << actual_expr << "\" : \n"
                                               << "pt : " << testing::PrintToString(p1.pt) << " vs " << testing::PrintToString(p2.pt) << "\n"
                                               << "size : " << p1.size << " vs " << p2.size << "\n"
                                               << "angle : " << p1.angle << " vs " << p2.angle << "\n"
                                               << "response : " << p1.response << " vs " << p2.response << "\n"
                                               << "octave : " << p1.octave << " vs " << p2.octave << "\n"
                                               << "class_id : " << p1.class_id << " vs " << p2.class_id;
        }
    }

    return ::testing::AssertionSuccess();
}

#define ASSERT_KEYPOINTS_EQ(gold, actual) EXPECT_PRED_FORMAT2(assertKeyPointsEquals, gold, actual);

int getMatchedPointsCount(std::vector<cv::KeyPoint>& gold, std::vector<cv::KeyPoint>& actual)
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

int getMatchedPointsCount(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2, const std::vector<cv::DMatch>& matches)
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

/////////////////////////////////////////////////////////////////////////////////////////////////
// SURF

IMPLEMENT_PARAM_CLASS(SURF_HessianThreshold, double)
IMPLEMENT_PARAM_CLASS(SURF_Octaves, int)
IMPLEMENT_PARAM_CLASS(SURF_OctaveLayers, int)
IMPLEMENT_PARAM_CLASS(SURF_Extended, bool)
IMPLEMENT_PARAM_CLASS(SURF_Upright, bool)

PARAM_TEST_CASE(SURF, cv::gpu::DeviceInfo, SURF_HessianThreshold, SURF_Octaves, SURF_OctaveLayers, SURF_Extended, SURF_Upright)
{
    cv::gpu::DeviceInfo devInfo;
    double hessianThreshold;
    int nOctaves;
    int nOctaveLayers;
    bool extended;
    bool upright;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        hessianThreshold = GET_PARAM(1);
        nOctaves = GET_PARAM(2);
        nOctaveLayers = GET_PARAM(3);
        extended = GET_PARAM(4);
        upright = GET_PARAM(5);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(SURF, Detector)
{
    cv::Mat image = readImage("features2d/aloe.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    cv::gpu::SURF_GPU surf;
    surf.hessianThreshold = hessianThreshold;
    surf.nOctaves = nOctaves;
    surf.nOctaveLayers = nOctaveLayers;
    surf.extended = extended;
    surf.upright = upright;
    surf.keypointsRatio = 0.05f;

    if (!supportFeature(devInfo, cv::gpu::GLOBAL_ATOMICS))
    {
        try
        {
            std::vector<cv::KeyPoint> keypoints;
            surf(loadMat(image), cv::gpu::GpuMat(), keypoints);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsNotImplemented, e.code);
        }
    }
    else
    {
        std::vector<cv::KeyPoint> keypoints;
        surf(loadMat(image), cv::gpu::GpuMat(), keypoints);

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
}

TEST_P(SURF, Detector_Masked)
{
    cv::Mat image = readImage("features2d/aloe.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar::all(1));
    mask(cv::Range(0, image.rows / 2), cv::Range(0, image.cols / 2)).setTo(cv::Scalar::all(0));

    cv::gpu::SURF_GPU surf;
    surf.hessianThreshold = hessianThreshold;
    surf.nOctaves = nOctaves;
    surf.nOctaveLayers = nOctaveLayers;
    surf.extended = extended;
    surf.upright = upright;
    surf.keypointsRatio = 0.05f;

    if (!supportFeature(devInfo, cv::gpu::GLOBAL_ATOMICS))
    {
        try
        {
            std::vector<cv::KeyPoint> keypoints;
            surf(loadMat(image), loadMat(mask), keypoints);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsNotImplemented, e.code);
        }
    }
    else
    {
        std::vector<cv::KeyPoint> keypoints;
        surf(loadMat(image), loadMat(mask), keypoints);

        cv::SURF surf_gold;
        surf_gold.hessianThreshold = hessianThreshold;
        surf_gold.nOctaves = nOctaves;
        surf_gold.nOctaveLayers = nOctaveLayers;
        surf_gold.extended = extended;
        surf_gold.upright = upright;

        std::vector<cv::KeyPoint> keypoints_gold;
        surf_gold(image, mask, keypoints_gold);

        ASSERT_EQ(keypoints_gold.size(), keypoints.size());
        int matchedCount = getMatchedPointsCount(keypoints_gold, keypoints);
        double matchedRatio = static_cast<double>(matchedCount) / keypoints_gold.size();

        EXPECT_GT(matchedRatio, 0.95);
    }
}

TEST_P(SURF, Descriptor)
{
    cv::Mat image = readImage("features2d/aloe.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    cv::gpu::SURF_GPU surf;
    surf.hessianThreshold = hessianThreshold;
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

    if (!supportFeature(devInfo, cv::gpu::GLOBAL_ATOMICS))
    {
        try
        {
            std::vector<cv::KeyPoint> keypoints;
            cv::gpu::GpuMat descriptors;
            surf(loadMat(image), cv::gpu::GpuMat(), keypoints, descriptors);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsNotImplemented, e.code);
        }
    }
    else
    {
        std::vector<cv::KeyPoint> keypoints;
        surf_gold(image, cv::noArray(), keypoints);

        cv::gpu::GpuMat descriptors;
        surf(loadMat(image), cv::gpu::GpuMat(), keypoints, descriptors, true);

        cv::Mat descriptors_gold;
        surf_gold(image, cv::noArray(), keypoints, descriptors_gold, true);

        cv::BFMatcher matcher(cv::NORM_L2);
        std::vector<cv::DMatch> matches;
        matcher.match(descriptors_gold, cv::Mat(descriptors), matches);

        int matchedCount = getMatchedPointsCount(keypoints, keypoints, matches);
        double matchedRatio = static_cast<double>(matchedCount) / keypoints.size();

        EXPECT_GT(matchedRatio, 0.35);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Features2D, SURF, testing::Combine(
    ALL_DEVICES,
    testing::Values(SURF_HessianThreshold(100.0), SURF_HessianThreshold(500.0), SURF_HessianThreshold(1000.0)),
    testing::Values(SURF_Octaves(3), SURF_Octaves(4)),
    testing::Values(SURF_OctaveLayers(2), SURF_OctaveLayers(3)),
    testing::Values(SURF_Extended(false), SURF_Extended(true)),
    testing::Values(SURF_Upright(false), SURF_Upright(true))));

/////////////////////////////////////////////////////////////////////////////////////////////////
// FAST

IMPLEMENT_PARAM_CLASS(FAST_Threshold, int)
IMPLEMENT_PARAM_CLASS(FAST_NonmaxSupression, bool)

PARAM_TEST_CASE(FAST, cv::gpu::DeviceInfo, FAST_Threshold, FAST_NonmaxSupression)
{
    cv::gpu::DeviceInfo devInfo;
    int threshold;
    bool nonmaxSupression;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        threshold = GET_PARAM(1);
        nonmaxSupression = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(FAST, Accuracy)
{
    cv::Mat image = readImage("features2d/aloe.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    cv::gpu::FAST_GPU fast(threshold);
    fast.nonmaxSupression = nonmaxSupression;

    if (!supportFeature(devInfo, cv::gpu::GLOBAL_ATOMICS))
    {
        try
        {
            std::vector<cv::KeyPoint> keypoints;
            fast(loadMat(image), cv::gpu::GpuMat(), keypoints);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsNotImplemented, e.code);
        }
    }
    else
    {
        std::vector<cv::KeyPoint> keypoints;
        fast(loadMat(image), cv::gpu::GpuMat(), keypoints);

        std::vector<cv::KeyPoint> keypoints_gold;
        cv::FAST(image, keypoints_gold, threshold, nonmaxSupression);

        ASSERT_KEYPOINTS_EQ(keypoints_gold, keypoints);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Features2D, FAST, testing::Combine(
    ALL_DEVICES,
    testing::Values(FAST_Threshold(25), FAST_Threshold(50)),
    testing::Values(FAST_NonmaxSupression(false), FAST_NonmaxSupression(true))));

/////////////////////////////////////////////////////////////////////////////////////////////////
// ORB

IMPLEMENT_PARAM_CLASS(ORB_FeaturesCount, int)
IMPLEMENT_PARAM_CLASS(ORB_ScaleFactor, float)
IMPLEMENT_PARAM_CLASS(ORB_LevelsCount, int)
IMPLEMENT_PARAM_CLASS(ORB_EdgeThreshold, int)
IMPLEMENT_PARAM_CLASS(ORB_firstLevel, int)
IMPLEMENT_PARAM_CLASS(ORB_WTA_K, int)
IMPLEMENT_PARAM_CLASS(ORB_PatchSize, int)
IMPLEMENT_PARAM_CLASS(ORB_BlurForDescriptor, bool)

CV_ENUM(ORB_ScoreType, cv::ORB::HARRIS_SCORE, cv::ORB::FAST_SCORE)

PARAM_TEST_CASE(ORB, cv::gpu::DeviceInfo, ORB_FeaturesCount, ORB_ScaleFactor, ORB_LevelsCount, ORB_EdgeThreshold, ORB_firstLevel, ORB_WTA_K, ORB_ScoreType, ORB_PatchSize, ORB_BlurForDescriptor)
{
    cv::gpu::DeviceInfo devInfo;
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
        devInfo = GET_PARAM(0);
        nFeatures = GET_PARAM(1);
        scaleFactor = GET_PARAM(2);
        nLevels = GET_PARAM(3);
        edgeThreshold = GET_PARAM(4);
        firstLevel = GET_PARAM(5);
        WTA_K = GET_PARAM(6);
        scoreType = GET_PARAM(7);
        patchSize = GET_PARAM(8);
        blurForDescriptor = GET_PARAM(9);

        cv::gpu::setDevice(devInfo.deviceID());
    }
};

TEST_P(ORB, Accuracy)
{
    cv::Mat image = readImage("features2d/aloe.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty());

    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar::all(1));
    mask(cv::Range(0, image.rows / 2), cv::Range(0, image.cols / 2)).setTo(cv::Scalar::all(0));

    cv::gpu::ORB_GPU orb(nFeatures, scaleFactor, nLevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize);
    orb.blurForDescriptor = blurForDescriptor;

    if (!supportFeature(devInfo, cv::gpu::GLOBAL_ATOMICS))
    {
        try
        {
            std::vector<cv::KeyPoint> keypoints;
            cv::gpu::GpuMat descriptors;
            orb(loadMat(image), loadMat(mask), keypoints, descriptors);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsNotImplemented, e.code);
        }
    }
    else
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::gpu::GpuMat descriptors;
        orb(loadMat(image), loadMat(mask), keypoints, descriptors);

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
}

INSTANTIATE_TEST_CASE_P(GPU_Features2D, ORB,  testing::Combine(
    ALL_DEVICES,
    testing::Values(ORB_FeaturesCount(1000)),
    testing::Values(ORB_ScaleFactor(1.2f)),
    testing::Values(ORB_LevelsCount(4), ORB_LevelsCount(8)),
    testing::Values(ORB_EdgeThreshold(31)),
    testing::Values(ORB_firstLevel(0), ORB_firstLevel(2)),
    testing::Values(ORB_WTA_K(2), ORB_WTA_K(3), ORB_WTA_K(4)),
    testing::Values(ORB_ScoreType(cv::ORB::HARRIS_SCORE)),
    testing::Values(ORB_PatchSize(31), ORB_PatchSize(29)),
    testing::Values(ORB_BlurForDescriptor(false), ORB_BlurForDescriptor(true))));

/////////////////////////////////////////////////////////////////////////////////////////////////
// BruteForceMatcher

CV_ENUM(DistType, cv::gpu::BruteForceMatcher_GPU_base::L1Dist, cv::gpu::BruteForceMatcher_GPU_base::L2Dist, cv::gpu::BruteForceMatcher_GPU_base::HammingDist)
IMPLEMENT_PARAM_CLASS(DescriptorSize, int)

PARAM_TEST_CASE(BruteForceMatcher, cv::gpu::DeviceInfo, DistType, DescriptorSize)
{
    cv::gpu::DeviceInfo devInfo;
    cv::gpu::BruteForceMatcher_GPU_base::DistType distType;
    int dim;

    int queryDescCount;
    int countFactor;

    cv::Mat query, train;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        distType = (cv::gpu::BruteForceMatcher_GPU_base::DistType)(int)GET_PARAM(1);
        dim = GET_PARAM(2);

        cv::gpu::setDevice(devInfo.deviceID());

        queryDescCount = 300; // must be even number because we split train data in some cases in two
        countFactor = 4; // do not change it

        cv::RNG& rng = cvtest::TS::ptr()->get_rng();

        cv::Mat queryBuf, trainBuf;

        // Generate query descriptors randomly.
        // Descriptor vector elements are integer values.
        queryBuf.create(queryDescCount, dim, CV_32SC1);
        rng.fill(queryBuf, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(3));
        queryBuf.convertTo(queryBuf, CV_32FC1);

        // Generate train decriptors as follows:
        // copy each query descriptor to train set countFactor times
        // and perturb some one element of the copied descriptors in
        // in ascending order. General boundaries of the perturbation
        // are (0.f, 1.f).
        trainBuf.create(queryDescCount * countFactor, dim, CV_32FC1);
        float step = 1.f / countFactor;
        for (int qIdx = 0; qIdx < queryDescCount; qIdx++)
        {
            cv::Mat queryDescriptor = queryBuf.row(qIdx);
            for (int c = 0; c < countFactor; c++)
            {
                int tIdx = qIdx * countFactor + c;
                cv::Mat trainDescriptor = trainBuf.row(tIdx);
                queryDescriptor.copyTo(trainDescriptor);
                int elem = rng(dim);
                float diff = rng.uniform(step * c, step * (c + 1));
                trainDescriptor.at<float>(0, elem) += diff;
            }
        }

        queryBuf.convertTo(query, CV_32F);
        trainBuf.convertTo(train, CV_32F);
    }
};

TEST_P(BruteForceMatcher, Match)
{
    cv::gpu::BruteForceMatcher_GPU_base matcher(distType);

    std::vector<cv::DMatch> matches;
    matcher.match(loadMat(query), loadMat(train), matches);

    ASSERT_EQ(static_cast<size_t>(queryDescCount), matches.size());

    int badCount = 0;
    for (size_t i = 0; i < matches.size(); i++)
    {
        cv::DMatch match = matches[i];
        if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor) || (match.imgIdx != 0))
            badCount++;
    }

    ASSERT_EQ(0, badCount);
}


TEST_P(BruteForceMatcher, MatchAdd)
{
    cv::gpu::BruteForceMatcher_GPU_base matcher(distType);

    cv::gpu::GpuMat d_train(train);

    // make add() twice to test such case
    matcher.add(std::vector<cv::gpu::GpuMat>(1, d_train.rowRange(0, train.rows / 2)));
    matcher.add(std::vector<cv::gpu::GpuMat>(1, d_train.rowRange(train.rows / 2, train.rows)));

    // prepare masks (make first nearest match illegal)
    std::vector<cv::gpu::GpuMat> masks(2);
    for (int mi = 0; mi < 2; mi++)
    {
        masks[mi] = cv::gpu::GpuMat(query.rows, train.rows/2, CV_8UC1, cv::Scalar::all(1));
        for (int di = 0; di < queryDescCount/2; di++)
            masks[mi].col(di * countFactor).setTo(cv::Scalar::all(0));
    }

    std::vector<cv::DMatch> matches;
    matcher.match(cv::gpu::GpuMat(query), matches, masks);

    ASSERT_EQ(static_cast<size_t>(queryDescCount), matches.size());

    int badCount = 0;
    int shift = matcher.isMaskSupported() ? 1 : 0;
    for (size_t i = 0; i < matches.size(); i++)
    {
        cv::DMatch match = matches[i];

        if ((int)i < queryDescCount / 2)
        {
            if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor + shift) || (match.imgIdx != 0))
                badCount++;
        }
        else
        {
            if ((match.queryIdx != (int)i) || (match.trainIdx != ((int)i - queryDescCount / 2) * countFactor + shift) || (match.imgIdx != 1))
                badCount++;
        }
    }

    ASSERT_EQ(0, badCount);
}

TEST_P(BruteForceMatcher, KnnMatch2)
{
    const int knn = 2;

    cv::gpu::BruteForceMatcher_GPU_base matcher(distType);

    std::vector< std::vector<cv::DMatch> > matches;
    matcher.knnMatch(loadMat(query), loadMat(train), matches, knn);

    ASSERT_EQ(static_cast<size_t>(queryDescCount), matches.size());

    int badCount = 0;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if ((int)matches[i].size() != knn)
            badCount++;
        else
        {
            int localBadCount = 0;
            for (int k = 0; k < knn; k++)
            {
                cv::DMatch match = matches[i][k];
                if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor + k) || (match.imgIdx != 0))
                    localBadCount++;
            }
            badCount += localBadCount > 0 ? 1 : 0;
        }
    }

    ASSERT_EQ(0, badCount);
}

TEST_P(BruteForceMatcher, KnnMatch3)
{
    cv::gpu::BruteForceMatcher_GPU_base matcher(distType);

    const int knn = 3;

    std::vector< std::vector<cv::DMatch> > matches;
    matcher.knnMatch(loadMat(query), loadMat(train), matches, knn);

    ASSERT_EQ(static_cast<size_t>(queryDescCount), matches.size());

    int badCount = 0;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if ((int)matches[i].size() != knn)
            badCount++;
        else
        {
            int localBadCount = 0;
            for (int k = 0; k < knn; k++)
            {
                cv::DMatch match = matches[i][k];
                if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor + k) || (match.imgIdx != 0))
                    localBadCount++;
            }
            badCount += localBadCount > 0 ? 1 : 0;
        }
    }

    ASSERT_EQ(0, badCount);
}

TEST_P(BruteForceMatcher, KnnMatchAdd2)
{
    const int knn = 2;

    cv::gpu::BruteForceMatcher_GPU_base matcher(distType);

    cv::gpu::GpuMat d_train(train);

    // make add() twice to test such case
    matcher.add(std::vector<cv::gpu::GpuMat>(1, d_train.rowRange(0, train.rows / 2)));
    matcher.add(std::vector<cv::gpu::GpuMat>(1, d_train.rowRange(train.rows / 2, train.rows)));

    // prepare masks (make first nearest match illegal)
    std::vector<cv::gpu::GpuMat> masks(2);
    for (int mi = 0; mi < 2; mi++ )
    {
        masks[mi] = cv::gpu::GpuMat(query.rows, train.rows / 2, CV_8UC1, cv::Scalar::all(1));
        for (int di = 0; di < queryDescCount / 2; di++)
            masks[mi].col(di * countFactor).setTo(cv::Scalar::all(0));
    }

    std::vector< std::vector<cv::DMatch> > matches;

    matcher.knnMatch(cv::gpu::GpuMat(query), matches, knn, masks);

    ASSERT_EQ(static_cast<size_t>(queryDescCount), matches.size());

    int badCount = 0;
    int shift = matcher.isMaskSupported() ? 1 : 0;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if ((int)matches[i].size() != knn)
            badCount++;
        else
        {
            int localBadCount = 0;
            for (int k = 0; k < knn; k++)
            {
                cv::DMatch match = matches[i][k];
                {
                    if ((int)i < queryDescCount / 2)
                    {
                        if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor + k + shift) || (match.imgIdx != 0) )
                            localBadCount++;
                    }
                    else
                    {
                        if ((match.queryIdx != (int)i) || (match.trainIdx != ((int)i - queryDescCount / 2) * countFactor + k + shift) || (match.imgIdx != 1) )
                            localBadCount++;
                    }
                }
            }
            badCount += localBadCount > 0 ? 1 : 0;
        }
    }

    ASSERT_EQ(0, badCount);
}

TEST_P(BruteForceMatcher, KnnMatchAdd3)
{
    const int knn = 3;

    cv::gpu::BruteForceMatcher_GPU_base matcher(distType);

    cv::gpu::GpuMat d_train(train);

    // make add() twice to test such case
    matcher.add(std::vector<cv::gpu::GpuMat>(1, d_train.rowRange(0, train.rows / 2)));
    matcher.add(std::vector<cv::gpu::GpuMat>(1, d_train.rowRange(train.rows / 2, train.rows)));

    // prepare masks (make first nearest match illegal)
    std::vector<cv::gpu::GpuMat> masks(2);
    for (int mi = 0; mi < 2; mi++ )
    {
        masks[mi] = cv::gpu::GpuMat(query.rows, train.rows / 2, CV_8UC1, cv::Scalar::all(1));
        for (int di = 0; di < queryDescCount / 2; di++)
            masks[mi].col(di * countFactor).setTo(cv::Scalar::all(0));
    }

    std::vector< std::vector<cv::DMatch> > matches;
    matcher.knnMatch(cv::gpu::GpuMat(query), matches, knn, masks);

    ASSERT_EQ(static_cast<size_t>(queryDescCount), matches.size());

    int badCount = 0;
    int shift = matcher.isMaskSupported() ? 1 : 0;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if ((int)matches[i].size() != knn)
            badCount++;
        else
        {
            int localBadCount = 0;
            for (int k = 0; k < knn; k++)
            {
                cv::DMatch match = matches[i][k];
                {
                    if ((int)i < queryDescCount / 2)
                    {
                        if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor + k + shift) || (match.imgIdx != 0) )
                            localBadCount++;
                    }
                    else
                    {
                        if ((match.queryIdx != (int)i) || (match.trainIdx != ((int)i - queryDescCount / 2) * countFactor + k + shift) || (match.imgIdx != 1) )
                            localBadCount++;
                    }
                }
            }
            badCount += localBadCount > 0 ? 1 : 0;
        }
    }

    ASSERT_EQ(0, badCount);
}

TEST_P(BruteForceMatcher, RadiusMatch)
{
    const float radius = 1.f / countFactor;

    cv::gpu::BruteForceMatcher_GPU_base matcher(distType);

    if (!supportFeature(devInfo, cv::gpu::GLOBAL_ATOMICS))
    {
        try
        {
            std::vector< std::vector<cv::DMatch> > matches;
            matcher.radiusMatch(loadMat(query), loadMat(train), matches, radius);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsNotImplemented, e.code);
        }
    }
    else
    {
        std::vector< std::vector<cv::DMatch> > matches;
        matcher.radiusMatch(loadMat(query), loadMat(train), matches, radius);

        ASSERT_EQ(static_cast<size_t>(queryDescCount), matches.size());

        int badCount = 0;
        for (size_t i = 0; i < matches.size(); i++)
        {
            if ((int)matches[i].size() != 1)
                badCount++;
            else
            {
                cv::DMatch match = matches[i][0];
                if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor) || (match.imgIdx != 0))
                    badCount++;
            }
        }

        ASSERT_EQ(0, badCount);
    }
}

TEST_P(BruteForceMatcher, RadiusMatchAdd)
{
    const int n = 3;
    const float radius = 1.f / countFactor * n;

    cv::gpu::BruteForceMatcher_GPU_base matcher(distType);

    cv::gpu::GpuMat d_train(train);

    // make add() twice to test such case
    matcher.add(std::vector<cv::gpu::GpuMat>(1, d_train.rowRange(0, train.rows / 2)));
    matcher.add(std::vector<cv::gpu::GpuMat>(1, d_train.rowRange(train.rows / 2, train.rows)));

    // prepare masks (make first nearest match illegal)
    std::vector<cv::gpu::GpuMat> masks(2);
    for (int mi = 0; mi < 2; mi++)
    {
        masks[mi] = cv::gpu::GpuMat(query.rows, train.rows / 2, CV_8UC1, cv::Scalar::all(1));
        for (int di = 0; di < queryDescCount / 2; di++)
            masks[mi].col(di * countFactor).setTo(cv::Scalar::all(0));
    }

    if (!supportFeature(devInfo, cv::gpu::GLOBAL_ATOMICS))
    {
        try
        {
            std::vector< std::vector<cv::DMatch> > matches;
            matcher.radiusMatch(cv::gpu::GpuMat(query), matches, radius, masks);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(CV_StsNotImplemented, e.code);
        }
    }
    else
    {
        std::vector< std::vector<cv::DMatch> > matches;
        matcher.radiusMatch(cv::gpu::GpuMat(query), matches, radius, masks);

        ASSERT_EQ(static_cast<size_t>(queryDescCount), matches.size());

        int badCount = 0;
        int shift = matcher.isMaskSupported() ? 1 : 0;
        int needMatchCount = matcher.isMaskSupported() ? n-1 : n;
        for (size_t i = 0; i < matches.size(); i++)
        {
            if ((int)matches[i].size() != needMatchCount)
                badCount++;
            else
            {
                int localBadCount = 0;
                for (int k = 0; k < needMatchCount; k++)
                {
                    cv::DMatch match = matches[i][k];
                    {
                        if ((int)i < queryDescCount / 2)
                        {
                            if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor + k + shift) || (match.imgIdx != 0) )
                                localBadCount++;
                        }
                        else
                        {
                            if ((match.queryIdx != (int)i) || (match.trainIdx != ((int)i - queryDescCount / 2) * countFactor + k + shift) || (match.imgIdx != 1) )
                                localBadCount++;
                        }
                    }
                }
                badCount += localBadCount > 0 ? 1 : 0;
            }
        }

        ASSERT_EQ(0, badCount);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_Features2D, BruteForceMatcher, testing::Combine(
    ALL_DEVICES,
    testing::Values(DistType(cv::gpu::BruteForceMatcher_GPU_base::L1Dist), DistType(cv::gpu::BruteForceMatcher_GPU_base::L2Dist)),
    testing::Values(DescriptorSize(57), DescriptorSize(64), DescriptorSize(83), DescriptorSize(128), DescriptorSize(179), DescriptorSize(256), DescriptorSize(304))));

} // namespace
