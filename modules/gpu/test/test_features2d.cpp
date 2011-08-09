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

#ifdef HAVE_CUDA

/////////////////////////////////////////////////////////////////////////////////////////////////
// SURF

struct SURF : testing::TestWithParam<cv::gpu::DeviceInfo>
{
    cv::Mat image;
    cv::Mat mask;
    std::vector<cv::KeyPoint> keypoints_gold;
    std::vector<float> descriptors_gold;

    cv::gpu::DeviceInfo devInfo;
    
    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());
        
        image = readImage("features2d/aloe.png", CV_LOAD_IMAGE_GRAYSCALE);
        ASSERT_FALSE(image.empty());        
        
        mask = cv::Mat(image.size(), CV_8UC1, cv::Scalar::all(1));
        mask(cv::Range(0, image.rows / 2), cv::Range(0, image.cols / 2)).setTo(cv::Scalar::all(0));
                
        cv::SURF fdetector_gold; fdetector_gold.extended = false;
        fdetector_gold(image, mask, keypoints_gold, descriptors_gold);        
    }

    bool isSimilarKeypoints(const cv::KeyPoint& p1, const cv::KeyPoint& p2)
    {
        const float maxPtDif = 1.f;
        const float maxSizeDif = 1.f;
        const float maxAngleDif = 2.f;
        const float maxResponseDif = 0.1f;

        float dist = (float)cv::norm(p1.pt - p2.pt);
        return (dist < maxPtDif &&
                fabs(p1.size - p2.size) < maxSizeDif &&
                abs(p1.angle - p2.angle) < maxAngleDif &&
                abs(p1.response - p2.response) < maxResponseDif &&
                p1.octave == p2.octave &&
                p1.class_id == p2.class_id );
    }
};

TEST_P(SURF, EmptyDataTest)
{
    PRINT_PARAM(devInfo);

    cv::gpu::SURF_GPU fdetector;

    cv::gpu::GpuMat image;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<float> descriptors;

    ASSERT_NO_THROW(
        fdetector(image, cv::gpu::GpuMat(), keypoints, descriptors);
    );

    EXPECT_TRUE(keypoints.empty());
    EXPECT_TRUE(descriptors.empty());
}

TEST_P(SURF, Accuracy)
{
    PRINT_PARAM(devInfo);

    // Compute keypoints.
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    ASSERT_NO_THROW(
        cv::gpu::GpuMat dev_descriptors;
        cv::gpu::SURF_GPU fdetector; fdetector.extended = false;

        fdetector(cv::gpu::GpuMat(image), cv::gpu::GpuMat(mask), keypoints, dev_descriptors);

        dev_descriptors.download(descriptors);
    );

    cv::BruteForceMatcher< cv::L2<float> > matcher;
    std::vector<cv::DMatch> matches;

    matcher.match(cv::Mat(static_cast<int>(keypoints_gold.size()), 64, CV_32FC1, &descriptors_gold[0]), descriptors, matches);

    int validCount = 0;
    
    for (size_t i = 0; i < matches.size(); ++i)
    {
        const cv::DMatch& m = matches[i];

        const cv::KeyPoint& p1 = keypoints_gold[m.queryIdx];
        const cv::KeyPoint& p2 = keypoints[m.trainIdx];

        const float maxPtDif = 1.f;
        const float maxSizeDif = 1.f;
        const float maxAngleDif = 2.f;
        const float maxResponseDif = 0.1f;

        float dist = (float)cv::norm(p1.pt - p2.pt);
        if (dist < maxPtDif &&
            fabs(p1.size - p2.size) < maxSizeDif &&
            abs(p1.angle - p2.angle) < maxAngleDif &&
            abs(p1.response - p2.response) < maxResponseDif &&
            p1.octave == p2.octave &&
            p1.class_id == p2.class_id)
        {
            ++validCount;
        }
    }

    double validRatio = (double)validCount / matches.size();

    EXPECT_GT(validRatio, 0.5);
}

INSTANTIATE_TEST_CASE_P(Features2D, SURF, testing::ValuesIn(devices(cv::gpu::GLOBAL_ATOMICS)));

/////////////////////////////////////////////////////////////////////////////////////////////////
// BruteForceMatcher

static const char* dists[] = {"L1Dist", "L2Dist", "HammingDist"};

struct BruteForceMatcher : testing::TestWithParam< std::tr1::tuple<cv::gpu::DeviceInfo, cv::gpu::BruteForceMatcher_GPU_base::DistType, int> >
{
    static const int queryDescCount = 300; // must be even number because we split train data in some cases in two
    static const int countFactor = 4; // do not change it

    cv::gpu::DeviceInfo devInfo;
    cv::gpu::BruteForceMatcher_GPU_base::DistType distType;
    int dim;
    
    cv::Mat query, train;

    virtual void SetUp() 
    {
        devInfo = std::tr1::get<0>(GetParam());
        distType = std::tr1::get<1>(GetParam());
        dim = std::tr1::get<2>(GetParam());

        cv::gpu::setDevice(devInfo.deviceID());

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

const int BruteForceMatcher::queryDescCount;
const int BruteForceMatcher::countFactor;

TEST_P(BruteForceMatcher, Match)
{
    const char* distStr = dists[distType];

    PRINT_PARAM(devInfo);
    PRINT_PARAM(distStr);
    PRINT_PARAM(dim);

    std::vector<cv::DMatch> matches;

    ASSERT_NO_THROW(
        cv::gpu::BruteForceMatcher_GPU_base matcher(distType);

        matcher.match(cv::gpu::GpuMat(query), cv::gpu::GpuMat(train), matches);
    );

    ASSERT_EQ(queryDescCount, matches.size());

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
    const char* distStr = dists[distType];

    PRINT_PARAM(devInfo);
    PRINT_PARAM(distStr);
    PRINT_PARAM(dim);

    std::vector<cv::DMatch> matches;

    bool isMaskSupported;

    ASSERT_NO_THROW(
        cv::gpu::BruteForceMatcher_GPU_base matcher(distType);

        cv::gpu::GpuMat d_train(train);

        // make add() twice to test such case
        matcher.add(std::vector<cv::gpu::GpuMat>(1, d_train.rowRange(0, train.rows/2)));
        matcher.add(std::vector<cv::gpu::GpuMat>(1, d_train.rowRange(train.rows/2, train.rows)));

        // prepare masks (make first nearest match illegal)
        std::vector<cv::gpu::GpuMat> masks(2);
        for (int mi = 0; mi < 2; mi++)
        {
            masks[mi] = cv::gpu::GpuMat(query.rows, train.rows/2, CV_8UC1, cv::Scalar::all(1));
            for (int di = 0; di < queryDescCount/2; di++)
                masks[mi].col(di * countFactor).setTo(cv::Scalar::all(0));
        }

        matcher.match(cv::gpu::GpuMat(query), matches, masks);

        isMaskSupported = matcher.isMaskSupported();
    );

    ASSERT_EQ(queryDescCount, matches.size());

    int badCount = 0;
    for (size_t i = 0; i < matches.size(); i++)
    {
        cv::DMatch match = matches[i];
        int shift = isMaskSupported ? 1 : 0;
        {
            if (i < queryDescCount / 2)
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
    }

    ASSERT_EQ(0, badCount);
}

TEST_P(BruteForceMatcher, KnnMatch)
{
    const char* distStr = dists[distType];

    PRINT_PARAM(devInfo);
    PRINT_PARAM(distStr);
    PRINT_PARAM(dim);

    const int knn = 3;

    std::vector< std::vector<cv::DMatch> > matches;

    ASSERT_NO_THROW(
        cv::gpu::BruteForceMatcher_GPU_base matcher(distType);
        matcher.knnMatch(cv::gpu::GpuMat(query), cv::gpu::GpuMat(train), matches, knn);
    );

    ASSERT_EQ(queryDescCount, matches.size());

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

TEST_P(BruteForceMatcher, KnnMatchAdd)
{
    const char* distStr = dists[distType];

    PRINT_PARAM(devInfo);
    PRINT_PARAM(distStr);
    PRINT_PARAM(dim);

    const int knn = 2;
    std::vector< std::vector<cv::DMatch> > matches;

    bool isMaskSupported;

    ASSERT_NO_THROW(
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

        matcher.knnMatch(cv::gpu::GpuMat(query), matches, knn, masks);

        isMaskSupported = matcher.isMaskSupported();
    );

    ASSERT_EQ(queryDescCount, matches.size());

    int badCount = 0;
    int shift = isMaskSupported ? 1 : 0;
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
                    if (i < queryDescCount / 2)
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
    if (!supportFeature(devInfo, cv::gpu::GLOBAL_ATOMICS))
        return;

    const char* distStr = dists[distType];

    PRINT_PARAM(devInfo);
    PRINT_PARAM(distStr);
    PRINT_PARAM(dim);

    const float radius = 1.f / countFactor;

    std::vector< std::vector<cv::DMatch> > matches;

    ASSERT_NO_THROW(
        cv::gpu::BruteForceMatcher_GPU_base matcher(distType);

        matcher.radiusMatch(cv::gpu::GpuMat(query), cv::gpu::GpuMat(train), matches, radius);
    );

    ASSERT_EQ(queryDescCount, matches.size());

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

TEST_P(BruteForceMatcher, RadiusMatchAdd)
{
    if (!supportFeature(devInfo, cv::gpu::GLOBAL_ATOMICS))
        return;

    const char* distStr = dists[distType];

    PRINT_PARAM(devInfo);
    PRINT_PARAM(distStr);
    PRINT_PARAM(dim);

    int n = 3;
    const float radius = 1.f / countFactor * n;

    std::vector< std::vector<cv::DMatch> > matches;

    bool isMaskSupported;

    ASSERT_NO_THROW(
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

        matcher.radiusMatch(cv::gpu::GpuMat(query), matches, radius, masks);

        isMaskSupported = matcher.isMaskSupported();
    );

    ASSERT_EQ(queryDescCount, matches.size());

    int badCount = 0;
    int shift = isMaskSupported ? 1 : 0;
    int needMatchCount = isMaskSupported ? n-1 : n;
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
                    if (i < queryDescCount / 2)
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

INSTANTIATE_TEST_CASE_P(Features2D, BruteForceMatcher, testing::Combine(
                        testing::ValuesIn(devices()),
                        testing::Values(cv::gpu::BruteForceMatcher_GPU_base::L1Dist, cv::gpu::BruteForceMatcher_GPU_base::L2Dist),
                        testing::Values(57, 64, 83, 128, 179, 256, 304)));

#endif // HAVE_CUDA
