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
    static cv::Mat image;
    static cv::Mat mask;
    static std::vector<cv::KeyPoint> keypoints_gold;
    static std::vector<float> descriptors_gold;

    static void SetUpTestCase() 
    {
        image = readImage("features2d/aloe.png", CV_LOAD_IMAGE_GRAYSCALE);        
        
        mask = cv::Mat(image.size(), CV_8UC1, cv::Scalar::all(1));
        mask(cv::Range(0, image.rows / 2), cv::Range(0, image.cols / 2)).setTo(cv::Scalar::all(0));
                
        cv::SURF fdetector_gold; fdetector_gold.extended = false;
        fdetector_gold(image, mask, keypoints_gold, descriptors_gold);
    }

    static void TearDownTestCase() 
    {
        image.release();
        mask.release();
        keypoints_gold.clear();
        descriptors_gold.clear();
    }

    cv::gpu::DeviceInfo devInfo;
    
    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());
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

cv::Mat SURF::image;
cv::Mat SURF::mask;
std::vector<cv::KeyPoint> SURF::keypoints_gold;
std::vector<float> SURF::descriptors_gold;

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
    ASSERT_TRUE(!image.empty());

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

    matcher.match(cv::Mat(keypoints_gold.size(), 64, CV_32FC1, &descriptors_gold[0]), descriptors, matches);

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
                p1.class_id == p2.class_id )
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

















//struct CV_GpuBFMTest : CV_GpuTestBase
//{
//    void run_gpu_test();
//       
//    void generateData(GpuMat& query, GpuMat& train, int dim, int depth);
//
//    virtual void test(const GpuMat& query, const GpuMat& train, BruteForceMatcher_GPU_base& matcher) = 0;
//
//    static const int queryDescCount = 300; // must be even number because we split train data in some cases in two
//    static const int countFactor = 4; // do not change it
//};
//
//void CV_GpuBFMTest::run_gpu_test()
//{
//    BruteForceMatcher_GPU_base::DistType dists[] = {BruteForceMatcher_GPU_base::L1Dist, BruteForceMatcher_GPU_base::L2Dist, BruteForceMatcher_GPU_base::HammingDist};
//    const char* dists_str[] = {"L1Dist", "L2Dist", "HammingDist"};
//    int dists_count = sizeof(dists) / sizeof(dists[0]);
//
//    RNG rng = ts->get_rng();
//
//    int dims[] = {rng.uniform(30, 60), 64, rng.uniform(70, 110), 128, rng.uniform(130, 250), 256, rng.uniform(260, 350)};
//    int dims_count = sizeof(dims) / sizeof(dims[0]);
//
//    for (int dist = 0; dist < dists_count; ++dist)
//    {
//        int depth_end = dists[dist] == BruteForceMatcher_GPU_base::HammingDist ? CV_32S : CV_32F;
//
//        for (int depth = CV_8U; depth <= depth_end; ++depth)
//        {
//            for (int dim = 0; dim < dims_count; ++dim)
//            {
//                PRINT_ARGS("dist=%s depth=%s dim=%d", dists_str[dist], getTypeName(depth), dims[dim]);
//                
//                BruteForceMatcher_GPU_base matcher(dists[dist]);
//
//                GpuMat query, train;
//                generateData(query, train, dim, depth);
//
//                test(query, train, matcher);
//            }
//        }
//    }
//}
//
//void CV_GpuBFMTest::generateData(GpuMat& queryGPU, GpuMat& trainGPU, int dim, int depth)
//{
//    RNG& rng = ts->get_rng();
//
//    Mat queryBuf, trainBuf;
//
//    // Generate query descriptors randomly.
//    // Descriptor vector elements are integer values.
//    queryBuf.create(queryDescCount, dim, CV_32SC1);
//    rng.fill(queryBuf, RNG::UNIFORM, Scalar::all(0), Scalar(3));
//    queryBuf.convertTo(queryBuf, CV_32FC1);
//
//    // Generate train decriptors as follows:
//    // copy each query descriptor to train set countFactor times
//    // and perturb some one element of the copied descriptors in
//    // in ascending order. General boundaries of the perturbation
//    // are (0.f, 1.f).
//    trainBuf.create(queryDescCount * countFactor, dim, CV_32FC1);
//    float step = 1.f / countFactor;
//    for (int qIdx = 0; qIdx < queryDescCount; qIdx++)
//    {
//        Mat queryDescriptor = queryBuf.row(qIdx);
//        for (int c = 0; c < countFactor; c++)
//        {
//            int tIdx = qIdx * countFactor + c;
//            Mat trainDescriptor = trainBuf.row(tIdx);
//            queryDescriptor.copyTo(trainDescriptor);
//            int elem = rng(dim);
//            float diff = rng.uniform(step * c, step * (c + 1));
//            trainDescriptor.at<float>(0, elem) += diff;
//        }
//    }
//
//    Mat query, train;
//    queryBuf.convertTo(query, depth);
//    trainBuf.convertTo(train, depth);
//
//    queryGPU.upload(query);
//    trainGPU.upload(train);
//}
//
//#define GPU_BFM_TEST(test_name) 
//    struct CV_GpuBFM_ ##test_name ## _Test : CV_GpuBFMTest 
//    { 
//        void test(const GpuMat& query, const GpuMat& train, BruteForceMatcher_GPU_base& matcher); 
//    }; 
//    TEST(BruteForceMatcher, test_name) { CV_GpuBFM_ ##test_name ## _Test test; test.safe_run(); } 
//    void CV_GpuBFM_ ##test_name ## _Test::test(const GpuMat& query, const GpuMat& train, BruteForceMatcher_GPU_base& matcher)
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//// match
//
//GPU_BFM_TEST(match)
//{
//    vector<DMatch> matches;
//
//    matcher.match(query, train, matches);
//
//    CHECK((int)matches.size() == queryDescCount, TS::FAIL_INVALID_OUTPUT);
//
//    int badCount = 0;
//    for (size_t i = 0; i < matches.size(); i++)
//    {
//        DMatch match = matches[i];
//        if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor) || (match.imgIdx != 0))
//            badCount++;
//    }
//
//    CHECK(badCount == 0, TS::FAIL_INVALID_OUTPUT);
//}
//
//GPU_BFM_TEST(match_add)
//{
//    vector<DMatch> matches;
//
//    // make add() twice to test such case
//    matcher.add(vector<GpuMat>(1, train.rowRange(0, train.rows/2)));
//    matcher.add(vector<GpuMat>(1, train.rowRange(train.rows/2, train.rows)));
//
//    // prepare masks (make first nearest match illegal)
//    vector<GpuMat> masks(2);
//    for (int mi = 0; mi < 2; mi++)
//    {
//        masks[mi] = GpuMat(query.rows, train.rows/2, CV_8UC1, Scalar::all(1));
//        for (int di = 0; di < queryDescCount/2; di++)
//            masks[mi].col(di * countFactor).setTo(Scalar::all(0));
//    }
//
//    matcher.match(query, matches, masks);
//
//    CHECK((int)matches.size() == queryDescCount, TS::FAIL_INVALID_OUTPUT);
//
//    int badCount = 0;
//    for (size_t i = 0; i < matches.size(); i++)
//    {
//        DMatch match = matches[i];
//        int shift = matcher.isMaskSupported() ? 1 : 0;
//        {
//            if (i < queryDescCount / 2)
//            {
//                if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor + shift) || (match.imgIdx != 0))
//                    badCount++;
//            }
//            else
//            {
//                if ((match.queryIdx != (int)i) || (match.trainIdx != ((int)i - queryDescCount / 2) * countFactor + shift) || (match.imgIdx != 1))
//                    badCount++;
//            }
//        }
//    }
//
//    CHECK(badCount == 0, TS::FAIL_INVALID_OUTPUT);
//}
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//// knnMatch
//
//GPU_BFM_TEST(knnMatch)
//{
//    const int knn = 3;
//
//    vector< vector<DMatch> > matches;
//
//    matcher.knnMatch(query, train, matches, knn);
//
//    CHECK((int)matches.size() == queryDescCount, TS::FAIL_INVALID_OUTPUT);
//
//    int badCount = 0;
//    for (size_t i = 0; i < matches.size(); i++)
//    {
//        if ((int)matches[i].size() != knn)
//            badCount++;
//        else
//        {
//            int localBadCount = 0;
//            for (int k = 0; k < knn; k++)
//            {
//                DMatch match = matches[i][k];
//                if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor + k) || (match.imgIdx != 0))
//                    localBadCount++;
//            }
//            badCount += localBadCount > 0 ? 1 : 0;
//        }
//    }
//
//    CHECK(badCount == 0, TS::FAIL_INVALID_OUTPUT);
//}
//
//GPU_BFM_TEST(knnMatch_add)
//{
//    const int knn = 2;
//    vector<vector<DMatch> > matches;
//
//    // make add() twice to test such case
//    matcher.add(vector<GpuMat>(1,train.rowRange(0, train.rows / 2)));
//    matcher.add(vector<GpuMat>(1,train.rowRange(train.rows / 2, train.rows)));
//
//    // prepare masks (make first nearest match illegal)
//    vector<GpuMat> masks(2);
//    for (int mi = 0; mi < 2; mi++ )
//    {
//        masks[mi] = GpuMat(query.rows, train.rows / 2, CV_8UC1, Scalar::all(1));
//        for (int di = 0; di < queryDescCount / 2; di++)
//            masks[mi].col(di * countFactor).setTo(Scalar::all(0));
//    }
//
//    matcher.knnMatch(query, matches, knn, masks);
//
//    CHECK((int)matches.size() == queryDescCount, TS::FAIL_INVALID_OUTPUT);
//
//    int badCount = 0;
//    int shift = matcher.isMaskSupported() ? 1 : 0;
//    for (size_t i = 0; i < matches.size(); i++)
//    {
//        if ((int)matches[i].size() != knn)
//            badCount++;
//        else
//        {
//            int localBadCount = 0;
//            for (int k = 0; k < knn; k++)
//            {
//                DMatch match = matches[i][k];
//                {
//                    if (i < queryDescCount / 2)
//                    {
//                        if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor + k + shift) || (match.imgIdx != 0) )
//                            localBadCount++;
//                    }
//                    else
//                    {
//                        if ((match.queryIdx != (int)i) || (match.trainIdx != ((int)i - queryDescCount / 2) * countFactor + k + shift) || (match.imgIdx != 1) )
//                            localBadCount++;
//                    }
//                }
//            }
//            badCount += localBadCount > 0 ? 1 : 0;
//        }
//    }
//
//    CHECK(badCount == 0, TS::FAIL_INVALID_OUTPUT);
//}
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//// radiusMatch
//
//GPU_BFM_TEST(radiusMatch)
//{
//    CHECK_RETURN(support(GLOBAL_ATOMICS), TS::SKIPPED);
//
//    const float radius = 1.f / countFactor;
//
//    vector< vector<DMatch> > matches;
//
//    matcher.radiusMatch(query, train, matches, radius);
//
//    CHECK((int)matches.size() == queryDescCount, TS::FAIL_INVALID_OUTPUT);
//
//    int badCount = 0;
//    for (size_t i = 0; i < matches.size(); i++)
//    {
//        if ((int)matches[i].size() != 1)
//            badCount++;
//        else
//        {
//            DMatch match = matches[i][0];
//            if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor) || (match.imgIdx != 0))
//                badCount++;
//        }
//    }
//
//    CHECK(badCount == 0, TS::FAIL_INVALID_OUTPUT);
//}
//
//GPU_BFM_TEST(radiusMatch_add)
//{
//    CHECK_RETURN(support(GLOBAL_ATOMICS), TS::SKIPPED);
//
//    int n = 3;
//    const float radius = 1.f / countFactor * n;
//    vector< vector<DMatch> > matches;
//
//    // make add() twice to test such case
//    matcher.add(vector<GpuMat>(1,train.rowRange(0, train.rows / 2)));
//    matcher.add(vector<GpuMat>(1,train.rowRange(train.rows / 2, train.rows)));
//
//    // prepare masks (make first nearest match illegal)
//    vector<GpuMat> masks(2);
//    for (int mi = 0; mi < 2; mi++)
//    {
//        masks[mi] = GpuMat(query.rows, train.rows / 2, CV_8UC1, Scalar::all(1));
//        for (int di = 0; di < queryDescCount / 2; di++)
//            masks[mi].col(di * countFactor).setTo(Scalar::all(0));
//    }
//
//    matcher.radiusMatch(query, matches, radius, masks);
//
//    CHECK((int)matches.size() == queryDescCount, TS::FAIL_INVALID_OUTPUT);
//
//    int badCount = 0;
//    int shift = matcher.isMaskSupported() ? 1 : 0;
//    int needMatchCount = matcher.isMaskSupported() ? n-1 : n;
//    for (size_t i = 0; i < matches.size(); i++)
//    {
//        if ((int)matches[i].size() != needMatchCount)
//            badCount++;
//        else
//        {
//            int localBadCount = 0;
//            for (int k = 0; k < needMatchCount; k++)
//            {
//                DMatch match = matches[i][k];
//                {
//                    if (i < queryDescCount / 2)
//                    {
//                        if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor + k + shift) || (match.imgIdx != 0) )
//                            localBadCount++;
//                    }
//                    else
//                    {
//                        if ((match.queryIdx != (int)i) || (match.trainIdx != ((int)i - queryDescCount / 2) * countFactor + k + shift) || (match.imgIdx != 1) )
//                            localBadCount++;
//                    }
//                }
//            }
//            badCount += localBadCount > 0 ? 1 : 0;
//        }
//    }
//
//    CHECK(badCount == 0, TS::FAIL_INVALID_OUTPUT);
//}
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
////struct CV_GpuBruteForceMatcherTest : CV_GpuTestBase
////{
////    void run_gpu_test();
////    
////    void emptyDataTest();
////    void dataTest(int dim);
////    
////    void generateData(GpuMat& query, GpuMat& train, int dim);
////
////    void matchTest(const GpuMat& query, const GpuMat& train);
////    void knnMatchTest(const GpuMat& query, const GpuMat& train);
////    void radiusMatchTest(const GpuMat& query, const GpuMat& train);
////
////    BruteForceMatcher_GPU< L2<float> > dmatcher;
////
////    static const int queryDescCount = 300; // must be even number because we split train data in some cases in two
////    static const int countFactor = 4; // do not change it
////};
////
////void CV_GpuBruteForceMatcherTest::emptyDataTest()
////{
////    GpuMat queryDescriptors, trainDescriptors, mask;
////    vector<GpuMat> trainDescriptorCollection, masks;
////    vector<DMatch> matches;
////    vector< vector<DMatch> > vmatches;
////
////    try
////    {
////        dmatcher.match(queryDescriptors, trainDescriptors, matches, mask);
////    }
////    catch(...)
////    {
////        PRINTLN("match() on empty descriptors must not generate exception (1)");
////        ts->set_failed_test_info(TS::FAIL_EXCEPTION);
////    }
////
////    try
////    {
////        dmatcher.knnMatch(queryDescriptors, trainDescriptors, vmatches, 2, mask);
////    }
////    catch(...)
////    {
////        PRINTLN("knnMatch() on empty descriptors must not generate exception (1)");
////        ts->set_failed_test_info(TS::FAIL_EXCEPTION);
////    }
////
////    try
////    {
////        dmatcher.radiusMatch(queryDescriptors, trainDescriptors, vmatches, 10.f, mask);
////    }
////    catch(...)
////    {
////        PRINTLN("radiusMatch() on empty descriptors must not generate exception (1)");
////        ts->set_failed_test_info(TS::FAIL_EXCEPTION);
////    }
////
////    try
////    {
////        dmatcher.add(trainDescriptorCollection);
////    }
////    catch(...)
////    {
////        PRINTLN("add() on empty descriptors must not generate exception");
////        ts->set_failed_test_info(TS::FAIL_EXCEPTION);
////    }
////
////    try
////    {
////        dmatcher.match(queryDescriptors, matches, masks);
////    }
////    catch(...)
////    {
////        PRINTLN("match() on empty descriptors must not generate exception (2)");
////        ts->set_failed_test_info(TS::FAIL_EXCEPTION);
////    }
////
////    try
////    {
////        dmatcher.knnMatch(queryDescriptors, vmatches, 2, masks);
////    }
////    catch(...)
////    {
////        PRINTLN("knnMatch() on empty descriptors must not generate exception (2)");
////        ts->set_failed_test_info(TS::FAIL_EXCEPTION);
////    }
////
////    try
////    {
////        dmatcher.radiusMatch( queryDescriptors, vmatches, 10.f, masks );
////    }
////    catch(...)
////    {
////        PRINTLN("radiusMatch() on empty descriptors must not generate exception (2)");
////        ts->set_failed_test_info(TS::FAIL_EXCEPTION);
////    }
////
////}
////
////void CV_GpuBruteForceMatcherTest::generateData(GpuMat& queryGPU, GpuMat& trainGPU, int dim)
////{
////    Mat query, train;
////    RNG& rng = ts->get_rng();
////
////    // Generate query descriptors randomly.
////    // Descriptor vector elements are integer values.
////    Mat buf(queryDescCount, dim, CV_32SC1);
////    rng.fill(buf, RNG::UNIFORM, Scalar::all(0), Scalar(3));
////    buf.convertTo(query, CV_32FC1);
////
////    // Generate train decriptors as follows:
////    // copy each query descriptor to train set countFactor times
////    // and perturb some one element of the copied descriptors in
////    // in ascending order. General boundaries of the perturbation
////    // are (0.f, 1.f).
////    train.create( query.rows*countFactor, query.cols, CV_32FC1 );
////    float step = 1.f / countFactor;
////    for (int qIdx = 0; qIdx < query.rows; qIdx++)
////    {
////        Mat queryDescriptor = query.row(qIdx);
////        for (int c = 0; c < countFactor; c++)
////        {
////            int tIdx = qIdx * countFactor + c;
////            Mat trainDescriptor = train.row(tIdx);
////            queryDescriptor.copyTo(trainDescriptor);
////            int elem = rng(dim);
////            float diff = rng.uniform(step * c, step * (c + 1));
////            trainDescriptor.at<float>(0, elem) += diff;
////        }
////    }
////
////    queryGPU.upload(query);
////    trainGPU.upload(train);
////}
////
////void CV_GpuBruteForceMatcherTest::matchTest(const GpuMat& query, const GpuMat& train)
////{
////    dmatcher.clear();
////
////    // test const version of match()
////    {
////        vector<DMatch> matches;
////        dmatcher.match(query, train, matches);
////
////        CHECK((int)matches.size() == queryDescCount, TS::FAIL_INVALID_OUTPUT);
////
////        int badCount = 0;
////        for (size_t i = 0; i < matches.size(); i++)
////        {
////            DMatch match = matches[i];
////            if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor) || (match.imgIdx != 0))
////                badCount++;
////        }
////
////        CHECK(badCount == 0, TS::FAIL_INVALID_OUTPUT);
////    }
////
////    // test version of match() with add()
////    {
////        vector<DMatch> matches;
////
////        // make add() twice to test such case
////        dmatcher.add(vector<GpuMat>(1, train.rowRange(0, train.rows/2)));
////        dmatcher.add(vector<GpuMat>(1, train.rowRange(train.rows/2, train.rows)));
////
////        // prepare masks (make first nearest match illegal)
////        vector<GpuMat> masks(2);
////        for (int mi = 0; mi < 2; mi++)
////        {
////            masks[mi] = GpuMat(query.rows, train.rows/2, CV_8UC1, Scalar::all(1));
////            for (int di = 0; di < queryDescCount/2; di++)
////                masks[mi].col(di * countFactor).setTo(Scalar::all(0));
////        }
////
////        dmatcher.match(query, matches, masks);
////
////        CHECK((int)matches.size() == queryDescCount, TS::FAIL_INVALID_OUTPUT);
////
////        int badCount = 0;
////        for (size_t i = 0; i < matches.size(); i++)
////        {
////            DMatch match = matches[i];
////            int shift = dmatcher.isMaskSupported() ? 1 : 0;
////            {
////                if (i < queryDescCount / 2)
////                {
////                    if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor + shift) || (match.imgIdx != 0))
////                        badCount++;
////                }
////                else
////                {
////                    if ((match.queryIdx != (int)i) || (match.trainIdx != ((int)i - queryDescCount / 2) * countFactor + shift) || (match.imgIdx != 1))
////                        badCount++;
////                }
////            }
////        }
////
////        CHECK(badCount == 0, TS::FAIL_INVALID_OUTPUT);
////    }
////}
////
////void CV_GpuBruteForceMatcherTest::knnMatchTest(const GpuMat& query, const GpuMat& train)
////{
////    dmatcher.clear();
////
////    // test const version of knnMatch()
////    {
////        const int knn = 3;
////
////        vector< vector<DMatch> > matches;
////        dmatcher.knnMatch(query, train, matches, knn);
////
////        CHECK((int)matches.size() == queryDescCount, TS::FAIL_INVALID_OUTPUT);
////
////        int badCount = 0;
////        for (size_t i = 0; i < matches.size(); i++)
////        {
////            if ((int)matches[i].size() != knn)
////                badCount++;
////            else
////            {
////                int localBadCount = 0;
////                for (int k = 0; k < knn; k++)
////                {
////                    DMatch match = matches[i][k];
////                    if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor + k) || (match.imgIdx != 0))
////                        localBadCount++;
////                }
////                badCount += localBadCount > 0 ? 1 : 0;
////            }
////        }
////
////        CHECK(badCount == 0, TS::FAIL_INVALID_OUTPUT);
////    }
////
////    // test version of knnMatch() with add()
////    {
////        const int knn = 2;
////        vector<vector<DMatch> > matches;
////
////        // make add() twice to test such case
////        dmatcher.add(vector<GpuMat>(1,train.rowRange(0, train.rows / 2)));
////        dmatcher.add(vector<GpuMat>(1,train.rowRange(train.rows / 2, train.rows)));
////
////        // prepare masks (make first nearest match illegal)
////        vector<GpuMat> masks(2);
////        for (int mi = 0; mi < 2; mi++ )
////        {
////            masks[mi] = GpuMat(query.rows, train.rows / 2, CV_8UC1, Scalar::all(1));
////            for (int di = 0; di < queryDescCount / 2; di++)
////                masks[mi].col(di * countFactor).setTo(Scalar::all(0));
////        }
////
////        dmatcher.knnMatch(query, matches, knn, masks);
////
////        CHECK((int)matches.size() == queryDescCount, TS::FAIL_INVALID_OUTPUT);
////
////        int badCount = 0;
////        int shift = dmatcher.isMaskSupported() ? 1 : 0;
////        for (size_t i = 0; i < matches.size(); i++)
////        {
////            if ((int)matches[i].size() != knn)
////                badCount++;
////            else
////            {
////                int localBadCount = 0;
////                for (int k = 0; k < knn; k++)
////                {
////                    DMatch match = matches[i][k];
////                    {
////                        if (i < queryDescCount / 2)
////                        {
////                            if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor + k + shift) || (match.imgIdx != 0) )
////                                localBadCount++;
////                        }
////                        else
////                        {
////                            if ((match.queryIdx != (int)i) || (match.trainIdx != ((int)i - queryDescCount / 2) * countFactor + k + shift) || (match.imgIdx != 1) )
////                                localBadCount++;
////                        }
////                    }
////                }
////                badCount += localBadCount > 0 ? 1 : 0;
////            }
////        }
////
////        CHECK(badCount == 0, TS::FAIL_INVALID_OUTPUT);
////    }
////}
////
////void CV_GpuBruteForceMatcherTest::radiusMatchTest(const GpuMat& query, const GpuMat& train)
////{
////    CHECK_RETURN(support(GLOBAL_ATOMICS), TS::SKIPPED);
////
////    dmatcher.clear();
////
////    // test const version of match()
////    {
////        const float radius = 1.f / countFactor;
////
////        vector< vector<DMatch> > matches;
////        dmatcher.radiusMatch(query, train, matches, radius);
////
////        CHECK((int)matches.size() == queryDescCount, TS::FAIL_INVALID_OUTPUT);
////
////        int badCount = 0;
////        for (size_t i = 0; i < matches.size(); i++)
////        {
////            if ((int)matches[i].size() != 1)
////                badCount++;
////            else
////            {
////                DMatch match = matches[i][0];
////                if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i*countFactor) || (match.imgIdx != 0))
////                    badCount++;
////            }
////        }
////
////        CHECK(badCount == 0, TS::FAIL_INVALID_OUTPUT);
////    }
////
////    // test version of match() with add()
////    {
////        int n = 3;
////        const float radius = 1.f / countFactor * n;
////        vector< vector<DMatch> > matches;
////
////        // make add() twice to test such case
////        dmatcher.add(vector<GpuMat>(1,train.rowRange(0, train.rows / 2)));
////        dmatcher.add(vector<GpuMat>(1,train.rowRange(train.rows / 2, train.rows)));
////
////        // prepare masks (make first nearest match illegal)
////        vector<GpuMat> masks(2);
////        for (int mi = 0; mi < 2; mi++)
////        {
////            masks[mi] = GpuMat(query.rows, train.rows / 2, CV_8UC1, Scalar::all(1));
////            for (int di = 0; di < queryDescCount / 2; di++)
////                masks[mi].col(di * countFactor).setTo(Scalar::all(0));
////        }
////
////        dmatcher.radiusMatch(query, matches, radius, masks);
////
////        CHECK((int)matches.size() == queryDescCount, TS::FAIL_INVALID_OUTPUT);
////
////        int badCount = 0;
////        int shift = dmatcher.isMaskSupported() ? 1 : 0;
////        int needMatchCount = dmatcher.isMaskSupported() ? n-1 : n;
////        for (size_t i = 0; i < matches.size(); i++)
////        {
////            if ((int)matches[i].size() != needMatchCount)
////                badCount++;
////            else
////            {
////                int localBadCount = 0;
////                for (int k = 0; k < needMatchCount; k++)
////                {
////                    DMatch match = matches[i][k];
////                    {
////                        if (i < queryDescCount / 2)
////                        {
////                            if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor + k + shift) || (match.imgIdx != 0) )
////                                localBadCount++;
////                        }
////                        else
////                        {
////                            if ((match.queryIdx != (int)i) || (match.trainIdx != ((int)i - queryDescCount / 2) * countFactor + k + shift) || (match.imgIdx != 1) )
////                                localBadCount++;
////                        }
////                    }
////                }
////                badCount += localBadCount > 0 ? 1 : 0;
////            }
////        }
////
////        CHECK(badCount == 0, TS::FAIL_INVALID_OUTPUT);
////    }
////}
////
////void CV_GpuBruteForceMatcherTest::dataTest(int dim)
////{
////    GpuMat query, train;
////    generateData(query, train, dim);
////
////    matchTest(query, train);
////    knnMatchTest(query, train);
////    radiusMatchTest(query, train);
////
////    dmatcher.clear();
////}
////
////void CV_GpuBruteForceMatcherTest::run_gpu_test()
////{
////    emptyDataTest();
////
////    dataTest(50);
////    dataTest(64);
////    dataTest(100);
////    dataTest(128);
////    dataTest(200);
////    dataTest(256);
////    dataTest(300);
////}
////
////TEST(BruteForceMatcher, accuracy) { CV_GpuBruteForceMatcherTest test; test.safe_run(); }
