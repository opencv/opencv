// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

CV_ENUM(CCLAlgorithm, CCL_DEFAULT, CCL_WU, CCL_GRANA, CCL_BOLELLI, CCL_SAUF, CCL_BBDT, CCL_SPAGHETTI)

typedef tuple<Size, int, CCLAlgorithm> CCParams;
typedef perf::TestBaseWithParam<CCParams> ConnectedComponents_Test;

PERF_TEST_P(ConnectedComponents_Test, connectedComponents,
            testing::Combine(
                testing::Values(cv::Size(512, 512), cv::Size(1024, 1024),
                                cv::Size(2048, 2048), cv::Size(4096, 4096)),
                testing::Values(4, 8),
                testing::Values(CCL_DEFAULT, CCL_WU, CCL_BOLELLI)
            )
)
{
    Size srcSize = get<0>(GetParam());
    int connectivity = get<1>(GetParam());
    int algorithm = get<2>(GetParam());

    Mat src(srcSize, CV_8U);
    Mat labels(srcSize, CV_32S);

    // Generate a binary image with random blobs to create many connected components.
    // Use threshold on random noise to produce medium-density regions.
    declare.in(src, WARMUP_RNG);
    cv::threshold(src, src, 200, 255, cv::THRESH_BINARY);

    declare.out(labels).time(60);

    TEST_CYCLE() cv::connectedComponents(src, labels, connectivity, CV_32S, algorithm);

    SANITY_CHECK_NOTHING();
}

typedef tuple<Size, int, CCLAlgorithm> CCStatsParams;
typedef perf::TestBaseWithParam<CCStatsParams> ConnectedComponentsWithStats_Test;

PERF_TEST_P(ConnectedComponentsWithStats_Test, connectedComponentsWithStats,
            testing::Combine(
                testing::Values(cv::Size(512, 512), cv::Size(1024, 1024),
                                cv::Size(2048, 2048), cv::Size(4096, 4096)),
                testing::Values(4, 8),
                testing::Values(CCL_DEFAULT, CCL_WU, CCL_BOLELLI)
            )
)
{
    Size srcSize = get<0>(GetParam());
    int connectivity = get<1>(GetParam());
    int algorithm = get<2>(GetParam());

    Mat src(srcSize, CV_8U);
    Mat labels(srcSize, CV_32S);
    Mat stats, centroids;

    declare.in(src, WARMUP_RNG);
    cv::threshold(src, src, 200, 255, cv::THRESH_BINARY);

    declare.out(labels).time(60);

    TEST_CYCLE() cv::connectedComponentsWithStats(src, labels, stats, centroids,
                                                  connectivity, CV_32S, algorithm);

    SANITY_CHECK_NOTHING();
}

}} // namespace
