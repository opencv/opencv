// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Features2D_KeypointUtils, retainBest_issue_12594)
{
    const size_t N = 9;

    // Construct 4-way tie for 3rd highest - correct answer for "3 best" is 6
    const float no_problem[] = { 5.0f, 4.0f, 1.0f, 2.0f, 0.0f, 3.0f, 3.0f, 3.0f, 3.0f };

    // Same set, different order that exposes partial sort property of std::nth_element
    // Note: the problem case may depend on your particular implementation of STL
    const float problem[] = { 3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 5.0f, 0.0f, 1.0f, 2.0f };

    const size_t NBEST  = 3u;
    const size_t ANSWER = 6u;

    std::vector<cv::KeyPoint> sorted_cv(N);
    std::vector<cv::KeyPoint> unsorted_cv(N);

    for (size_t i = 0; i < N; ++i)
    {
        sorted_cv[i].response   = no_problem[i];
        unsorted_cv[i].response = problem[i];
    }

    cv::KeyPointsFilter::retainBest(sorted_cv, NBEST);
    cv::KeyPointsFilter::retainBest(unsorted_cv, NBEST);

    EXPECT_EQ(ANSWER, sorted_cv.size());
    EXPECT_EQ(ANSWER, unsorted_cv.size());
}

}} // namespace
