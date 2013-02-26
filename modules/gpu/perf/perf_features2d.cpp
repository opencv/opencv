#include "perf_precomp.hpp"

using namespace std;
using namespace testing;

struct KeypointIdxCompare
{
    std::vector<cv::KeyPoint>* keypoints;

    explicit KeypointIdxCompare(std::vector<cv::KeyPoint>* _keypoints) : keypoints(_keypoints) {}

    bool operator ()(size_t i1, size_t i2) const
    {
        cv::KeyPoint kp1 = (*keypoints)[i1];
        cv::KeyPoint kp2 = (*keypoints)[i2];
        if (kp1.pt.x != kp2.pt.x)
            return kp1.pt.x < kp2.pt.x;
        if (kp1.pt.y != kp2.pt.y)
            return kp1.pt.y < kp2.pt.y;
        if (kp1.response != kp2.response)
            return kp1.response < kp2.response;
        return kp1.octave < kp2.octave;
    }
};

static void sortKeyPoints(std::vector<cv::KeyPoint>& keypoints, cv::InputOutputArray _descriptors = cv::noArray())
{
    std::vector<size_t> indexies(keypoints.size());
    for (size_t i = 0; i < indexies.size(); ++i)
        indexies[i] = i;

    std::sort(indexies.begin(), indexies.end(), KeypointIdxCompare(&keypoints));

    std::vector<cv::KeyPoint> new_keypoints;
    cv::Mat new_descriptors;

    new_keypoints.resize(keypoints.size());

    cv::Mat descriptors;
    if (_descriptors.needed())
    {
        descriptors = _descriptors.getMat();
        new_descriptors.create(descriptors.size(), descriptors.type());
    }

    for (size_t i = 0; i < indexies.size(); ++i)
    {
        size_t new_idx = indexies[i];
        new_keypoints[i] = keypoints[new_idx];
        if (!new_descriptors.empty())
            descriptors.row((int) new_idx).copyTo(new_descriptors.row((int) i));
    }

    keypoints.swap(new_keypoints);
    if (_descriptors.needed())
        new_descriptors.copyTo(_descriptors);
}

//////////////////////////////////////////////////////////////////////
// SURF

DEF_PARAM_TEST_1(Image, string);

PERF_TEST_P(Image, Features2D_SURF,
            Values<string>("gpu/perf/aloe.png"))
{
    declare.time(50.0);

    const cv::Mat img = readImage(GetParam(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    if (PERF_RUN_GPU())
    {
        cv::gpu::SURF_GPU d_surf;

        const cv::gpu::GpuMat d_img(img);
        cv::gpu::GpuMat d_keypoints, d_descriptors;

        TEST_CYCLE() d_surf(d_img, cv::gpu::GpuMat(), d_keypoints, d_descriptors);

        std::vector<cv::KeyPoint> gpu_keypoints;
        d_surf.downloadKeypoints(d_keypoints, gpu_keypoints);

        cv::Mat gpu_descriptors(d_descriptors);

        sortKeyPoints(gpu_keypoints, gpu_descriptors);

        SANITY_CHECK_KEYPOINTS(gpu_keypoints);
        SANITY_CHECK(gpu_descriptors);
    }
    else
    {
        cv::SURF surf;

        std::vector<cv::KeyPoint> cpu_keypoints;
        cv::Mat cpu_descriptors;

        TEST_CYCLE() surf(img, cv::noArray(), cpu_keypoints, cpu_descriptors);

        SANITY_CHECK_KEYPOINTS(cpu_keypoints);
        SANITY_CHECK(cpu_descriptors);
    }
}

//////////////////////////////////////////////////////////////////////
// FAST

DEF_PARAM_TEST(Image_Threshold_NonMaxSupression, string, int, bool);

PERF_TEST_P(Image_Threshold_NonMaxSupression, Features2D_FAST,
            Combine(Values<string>("gpu/perf/aloe.png"),
                    Values(20),
                    Bool()))
{
    const cv::Mat img = readImage(GET_PARAM(0), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    const int threshold = GET_PARAM(1);
    const bool nonMaxSuppersion = GET_PARAM(2);

    if (PERF_RUN_GPU())
    {
        cv::gpu::FAST_GPU d_fast(threshold, nonMaxSuppersion, 0.5);

        const cv::gpu::GpuMat d_img(img);
        cv::gpu::GpuMat d_keypoints;

        TEST_CYCLE() d_fast(d_img, cv::gpu::GpuMat(), d_keypoints);

        std::vector<cv::KeyPoint> gpu_keypoints;
        d_fast.downloadKeypoints(d_keypoints, gpu_keypoints);

        sortKeyPoints(gpu_keypoints);

        SANITY_CHECK_KEYPOINTS(gpu_keypoints);
    }
    else
    {
        std::vector<cv::KeyPoint> cpu_keypoints;

        TEST_CYCLE() cv::FAST(img, cpu_keypoints, threshold, nonMaxSuppersion);

        SANITY_CHECK_KEYPOINTS(cpu_keypoints);
    }
}

//////////////////////////////////////////////////////////////////////
// ORB

DEF_PARAM_TEST(Image_NFeatures, string, int);

PERF_TEST_P(Image_NFeatures, Features2D_ORB,
            Combine(Values<string>("gpu/perf/aloe.png"),
                    Values(4000)))
{
    const cv::Mat img = readImage(GET_PARAM(0), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    const int nFeatures = GET_PARAM(1);

    if (PERF_RUN_GPU())
    {
        cv::gpu::ORB_GPU d_orb(nFeatures);

        const cv::gpu::GpuMat d_img(img);
        cv::gpu::GpuMat d_keypoints, d_descriptors;

        TEST_CYCLE() d_orb(d_img, cv::gpu::GpuMat(), d_keypoints, d_descriptors);

        std::vector<cv::KeyPoint> gpu_keypoints;
        d_orb.downloadKeyPoints(d_keypoints, gpu_keypoints);

        cv::Mat gpu_descriptors(d_descriptors);

        gpu_keypoints.resize(10);
        gpu_descriptors = gpu_descriptors.rowRange(0, 10);

        sortKeyPoints(gpu_keypoints, gpu_descriptors);

        SANITY_CHECK_KEYPOINTS(gpu_keypoints);
        SANITY_CHECK(gpu_descriptors);
    }
    else
    {
        cv::ORB orb(nFeatures);

        std::vector<cv::KeyPoint> cpu_keypoints;
        cv::Mat cpu_descriptors;

        TEST_CYCLE() orb(img, cv::noArray(), cpu_keypoints, cpu_descriptors);

        SANITY_CHECK_KEYPOINTS(cpu_keypoints);
        SANITY_CHECK(cpu_descriptors);
    }
}

//////////////////////////////////////////////////////////////////////
// BFMatch

DEF_PARAM_TEST(DescSize_Norm, int, NormType);

PERF_TEST_P(DescSize_Norm, Features2D_BFMatch,
            Combine(Values(64, 128, 256),
                    Values(NormType(cv::NORM_L1), NormType(cv::NORM_L2), NormType(cv::NORM_HAMMING))))
{
    declare.time(20.0);

    const int desc_size = GET_PARAM(0);
    const int normType = GET_PARAM(1);

    const int type = normType == cv::NORM_HAMMING ? CV_8U : CV_32F;

    cv::Mat query(3000, desc_size, type);
    declare.in(query, WARMUP_RNG);

    cv::Mat train(3000, desc_size, type);
    declare.in(train, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        cv::gpu::BFMatcher_GPU d_matcher(normType);

        const cv::gpu::GpuMat d_query(query);
        const cv::gpu::GpuMat d_train(train);
        cv::gpu::GpuMat d_trainIdx, d_distance;

        TEST_CYCLE() d_matcher.matchSingle(d_query, d_train, d_trainIdx, d_distance);

        std::vector<cv::DMatch> gpu_matches;
        d_matcher.matchDownload(d_trainIdx, d_distance, gpu_matches);

        SANITY_CHECK_MATCHES(gpu_matches);
    }
    else
    {
        cv::BFMatcher matcher(normType);

        std::vector<cv::DMatch> cpu_matches;

        TEST_CYCLE() matcher.match(query, train, cpu_matches);

        SANITY_CHECK_MATCHES(cpu_matches);
    }
}

//////////////////////////////////////////////////////////////////////
// BFKnnMatch

static void toOneRowMatches(const std::vector< std::vector<cv::DMatch> >& src, std::vector<cv::DMatch>& dst)
{
    dst.clear();
    for (size_t i = 0; i < src.size(); ++i)
        for (size_t j = 0; j < src[i].size(); ++j)
            dst.push_back(src[i][j]);
}

DEF_PARAM_TEST(DescSize_K_Norm, int, int, NormType);

PERF_TEST_P(DescSize_K_Norm, Features2D_BFKnnMatch,
            Combine(Values(64, 128, 256),
                    Values(2, 3),
                    Values(NormType(cv::NORM_L1), NormType(cv::NORM_L2))))
{
    declare.time(30.0);

    const int desc_size = GET_PARAM(0);
    const int k = GET_PARAM(1);
    const int normType = GET_PARAM(2);

    const int type = normType == cv::NORM_HAMMING ? CV_8U : CV_32F;

    cv::Mat query(3000, desc_size, type);
    declare.in(query, WARMUP_RNG);

    cv::Mat train(3000, desc_size, type);
    declare.in(train, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        cv::gpu::BFMatcher_GPU d_matcher(normType);

        const cv::gpu::GpuMat d_query(query);
        const cv::gpu::GpuMat d_train(train);
        cv::gpu::GpuMat d_trainIdx, d_distance, d_allDist;

        TEST_CYCLE() d_matcher.knnMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_allDist, k);

        std::vector< std::vector<cv::DMatch> > matchesTbl;
        d_matcher.knnMatchDownload(d_trainIdx, d_distance, matchesTbl);

        std::vector<cv::DMatch> gpu_matches;
        toOneRowMatches(matchesTbl, gpu_matches);

        SANITY_CHECK_MATCHES(gpu_matches);
    }
    else
    {
        cv::BFMatcher matcher(normType);

        std::vector< std::vector<cv::DMatch> > matchesTbl;

        TEST_CYCLE() matcher.knnMatch(query, train, matchesTbl, k);

        std::vector<cv::DMatch> cpu_matches;
        toOneRowMatches(matchesTbl, cpu_matches);

        SANITY_CHECK_MATCHES(cpu_matches);
    }
}

//////////////////////////////////////////////////////////////////////
// BFRadiusMatch

PERF_TEST_P(DescSize_Norm, Features2D_BFRadiusMatch,
            Combine(Values(64, 128, 256),
                    Values(NormType(cv::NORM_L1), NormType(cv::NORM_L2))))
{
    declare.time(30.0);

    const int desc_size = GET_PARAM(0);
    const int normType = GET_PARAM(1);

    const int type = normType == cv::NORM_HAMMING ? CV_8U : CV_32F;
    const float maxDistance = 10000;

    cv::Mat query(3000, desc_size, type);
    declare.in(query, WARMUP_RNG);

    cv::Mat train(3000, desc_size, type);
    declare.in(train, WARMUP_RNG);

    if (PERF_RUN_GPU())
    {
        cv::gpu::BFMatcher_GPU d_matcher(normType);

        const cv::gpu::GpuMat d_query(query);
        const cv::gpu::GpuMat d_train(train);
        cv::gpu::GpuMat d_trainIdx, d_nMatches, d_distance;

        TEST_CYCLE() d_matcher.radiusMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_nMatches, maxDistance);

        std::vector< std::vector<cv::DMatch> > matchesTbl;
        d_matcher.radiusMatchDownload(d_trainIdx, d_distance, d_nMatches, matchesTbl);

        std::vector<cv::DMatch> gpu_matches;
        toOneRowMatches(matchesTbl, gpu_matches);

        SANITY_CHECK_MATCHES(gpu_matches);
    }
    else
    {
        cv::BFMatcher matcher(normType);

        std::vector< std::vector<cv::DMatch> > matchesTbl;

        TEST_CYCLE() matcher.radiusMatch(query, train, matchesTbl, maxDistance);

        std::vector<cv::DMatch> cpu_matches;
        toOneRowMatches(matchesTbl, cpu_matches);

        SANITY_CHECK_MATCHES(cpu_matches);
    }
}
