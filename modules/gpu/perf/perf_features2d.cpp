#include "perf_precomp.hpp"

using namespace std;
using namespace testing;

namespace {

//////////////////////////////////////////////////////////////////////
// SURF

DEF_PARAM_TEST_1(Image, string);

PERF_TEST_P(Image, Features2D_SURF, Values<string>("gpu/perf/aloe.jpg"))
{
    declare.time(50.0);

    cv::Mat img = readImage(GetParam(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    if (runOnGpu)
    {
        cv::gpu::SURF_GPU d_surf;

        cv::gpu::GpuMat d_img(img);
        cv::gpu::GpuMat d_keypoints, d_descriptors;

        d_surf(d_img, cv::gpu::GpuMat(), d_keypoints, d_descriptors);

        TEST_CYCLE()
        {
            d_surf(d_img, cv::gpu::GpuMat(), d_keypoints, d_descriptors);
        }
    }
    else
    {
        cv::SURF surf;

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        surf(img, cv::noArray(), keypoints, descriptors);

        TEST_CYCLE()
        {
            keypoints.clear();
            surf(img, cv::noArray(), keypoints, descriptors);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// FAST

PERF_TEST_P(Image, Features2D_FAST, Values<string>("gpu/perf/aloe.jpg"))
{
    cv::Mat img = readImage(GetParam(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    if (runOnGpu)
    {
        cv::gpu::FAST_GPU d_fast(20);

        cv::gpu::GpuMat d_img(img);
        cv::gpu::GpuMat d_keypoints;

        d_fast(d_img, cv::gpu::GpuMat(), d_keypoints);

        TEST_CYCLE()
        {
            d_fast(d_img, cv::gpu::GpuMat(), d_keypoints);
        }
    }
    else
    {
        std::vector<cv::KeyPoint> keypoints;

        cv::FAST(img, keypoints, 20);

        TEST_CYCLE()
        {
            keypoints.clear();
            cv::FAST(img, keypoints, 20);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// ORB

PERF_TEST_P(Image, Features2D_ORB, Values<string>("gpu/perf/aloe.jpg"))
{
    cv::Mat img = readImage(GetParam(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    if (runOnGpu)
    {
        cv::gpu::ORB_GPU d_orb(4000);

        cv::gpu::GpuMat d_img(img);
        cv::gpu::GpuMat d_keypoints, d_descriptors;

        d_orb(d_img, cv::gpu::GpuMat(), d_keypoints, d_descriptors);

        TEST_CYCLE()
        {
            d_orb(d_img, cv::gpu::GpuMat(), d_keypoints, d_descriptors);
        }
    }
    else
    {
        cv::ORB orb(4000);

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        orb(img, cv::noArray(), keypoints, descriptors);

        TEST_CYCLE()
        {
            keypoints.clear();
            orb(img, cv::noArray(), keypoints, descriptors);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// BFMatch

DEF_PARAM_TEST(DescSize_Norm, int, NormType);

PERF_TEST_P(DescSize_Norm, Features2D_BFMatch, Combine(Values(64, 128, 256), Values(NormType(cv::NORM_L1), NormType(cv::NORM_L2), NormType(cv::NORM_HAMMING))))
{
    declare.time(20.0);

    int desc_size = GET_PARAM(0);
    int normType = GET_PARAM(1);

    int type = normType == cv::NORM_HAMMING ? CV_8U : CV_32F;

    cv::Mat query(3000, desc_size, type);
    fillRandom(query);

    cv::Mat train(3000, desc_size, type);
    fillRandom(train);

    if (runOnGpu)
    {
        cv::gpu::BFMatcher_GPU d_matcher(normType);

        cv::gpu::GpuMat d_query(query);
        cv::gpu::GpuMat d_train(train);
        cv::gpu::GpuMat d_trainIdx, d_distance;

        d_matcher.matchSingle(d_query, d_train, d_trainIdx, d_distance);

        TEST_CYCLE()
        {
            d_matcher.matchSingle(d_query, d_train, d_trainIdx, d_distance);
        }
    }
    else
    {
        cv::BFMatcher matcher(normType);

        std::vector<cv::DMatch> matches;

        matcher.match(query, train, matches);

        TEST_CYCLE()
        {
            matcher.match(query, train, matches);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// BFKnnMatch

DEF_PARAM_TEST(DescSize_K_Norm, int, int, NormType);

PERF_TEST_P(DescSize_K_Norm, Features2D_BFKnnMatch, Combine(
    Values(64, 128, 256),
    Values(2, 3),
    Values(NormType(cv::NORM_L1), NormType(cv::NORM_L2), NormType(cv::NORM_HAMMING))))
{
    declare.time(30.0);

    int desc_size = GET_PARAM(0);
    int k = GET_PARAM(1);
    int normType = GET_PARAM(2);

    int type = normType == cv::NORM_HAMMING ? CV_8U : CV_32F;

    cv::Mat query(3000, desc_size, type);
    fillRandom(query);

    cv::Mat train(3000, desc_size, type);
    fillRandom(train);

    if (runOnGpu)
    {
        cv::gpu::BFMatcher_GPU d_matcher(normType);

        cv::gpu::GpuMat d_query(query);
        cv::gpu::GpuMat d_train(train);
        cv::gpu::GpuMat d_trainIdx, d_distance, d_allDist;

        d_matcher.knnMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_allDist, k);

        TEST_CYCLE()
        {
            d_matcher.knnMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_allDist, k);
        }
    }
    else
    {
        cv::BFMatcher matcher(normType);

        std::vector< std::vector<cv::DMatch> > matches;

        matcher.knnMatch(query, train, matches, k);

        TEST_CYCLE()
        {
            matcher.knnMatch(query, train, matches, k);
        }
    }
}

//////////////////////////////////////////////////////////////////////
// BFRadiusMatch

PERF_TEST_P(DescSize_Norm, Features2D_BFRadiusMatch, Combine(Values(64, 128, 256), Values(NormType(cv::NORM_L1), NormType(cv::NORM_L2), NormType(cv::NORM_HAMMING))))
{
    declare.time(30.0);

    int desc_size = GET_PARAM(0);
    int normType = GET_PARAM(1);

    int type = normType == cv::NORM_HAMMING ? CV_8U : CV_32F;

    cv::Mat query(3000, desc_size, type);
    fillRandom(query, 0.0, 1.0);

    cv::Mat train(3000, desc_size, type);
    fillRandom(train, 0.0, 1.0);

    if (runOnGpu)
    {
        cv::gpu::BFMatcher_GPU d_matcher(normType);

        cv::gpu::GpuMat d_query(query);
        cv::gpu::GpuMat d_train(train);
        cv::gpu::GpuMat d_trainIdx, d_nMatches, d_distance;

        d_matcher.radiusMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_nMatches, 2.0);

        TEST_CYCLE()
        {
            d_matcher.radiusMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_nMatches, 2.0);
        }
    }
    else
    {
        cv::BFMatcher matcher(normType);

        std::vector< std::vector<cv::DMatch> > matches;

        matcher.radiusMatch(query, train, matches, 2.0);

        TEST_CYCLE()
        {
            matcher.radiusMatch(query, train, matches, 2.0);
        }
    }
}

} // namespace
