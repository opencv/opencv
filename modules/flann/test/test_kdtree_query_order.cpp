// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

#include <thread>
#include <utility>

namespace opencv_test { namespace {

typedef cv::flann::GenericIndex<cvflann::L2<float> > KDTreeIndexF;

static cv::Mat makeDescriptors(cv::RNG& rng, int rows)
{
    cv::Mat bytes(rows, 128, CV_8U), values;
    rng.fill(bytes, cv::RNG::UNIFORM, 0, 256);
    bytes.convertTo(values, CV_32F);
    return values;
}

static std::pair<cv::Vec2i, cv::Vec2f> knn2(KDTreeIndexF& index, const cv::Mat& query)
{
    cv::Mat ids(1, 2, CV_32S), distances(1, 2, CV_32F);
    index.knnSearch(query, ids, distances, 2, cvflann::SearchParams(32, 0, true));
    return std::make_pair(cv::Vec2i(ids.at<int>(0, 0), ids.at<int>(0, 1)),
                          cv::Vec2f(distances.at<float>(0, 0), distances.at<float>(0, 1)));
}

// Querying a larger KDTree index must not change the result of a query
// against a smaller index on the same thread (issue #31161). The search
// heap is kept in thread-local storage and its capacity survives across
// queries, so a query against a small index after a large one could push
// more candidate branches than a fresh heap would allow and explore
// branches it would otherwise skip.
//
// The whole case runs on a fresh worker thread so the thread-local heap
// starts empty: if earlier tests on the main thread had already grown it,
// the "before" query would already see the grown capacity and the test
// would pass even without the fix.
TEST(Flann_KDTree, query_result_independent_of_previous_larger_index)
{
    cv::setNumThreads(1);

    std::pair<cv::Vec2i, cv::Vec2f> before, repeat, after;
    std::thread worker([&]()
    {
        const uint64_t seed = 0x5052534645415455ULL;
        cv::RNG rng(seed + 1000003ULL * 14 + 7919ULL * 12);
        const cv::Mat smallData = makeDescriptors(rng, 14);
        const cv::Mat queries = makeDescriptors(rng, 200);
        const cv::Mat largeData = makeDescriptors(rng, 4096);
        const cv::Mat query = queries.row(112);

        cv::theRNG().state = seed;
        KDTreeIndexF small(smallData, cvflann::KDTreeIndexParams(4));
        cv::theRNG().state = seed;
        KDTreeIndexF large(largeData, cvflann::KDTreeIndexParams(4));

        before = knn2(small, query);
        repeat = knn2(small, query);
        knn2(large, smallData.row(0));
        after = knn2(small, query);
    });
    worker.join();

    EXPECT_EQ(repeat.first, before.first);
    EXPECT_EQ(repeat.second, before.second);
    EXPECT_EQ(after.first, before.first) << "small-index query changed after a larger index was queried";
    EXPECT_EQ(after.second, before.second) << "small-index query changed after a larger index was queried";
}

}} // namespace
