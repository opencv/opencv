// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {
TEST(Video_Farneback, gaussianTranslatedRoi)
{
    RNG rng(30388);
    for (int width : {63, 64, 65, 127, 128, 129})
    for (int window : {5, 15, 25})
    {
        Mat storage(100, width + 12, CV_8U);
        rng.fill(storage, RNG::UNIFORM, 0, 256);
        GaussianBlur(storage, storage, Size(5, 5), 1.0);
        Mat first = storage(Rect(3, 2, width, 96));
        Mat second;
        Mat transform = Mat_<double>({2, 3}, {1., 0., 1., 0., 1., -1.});
        warpAffine(first, second, transform, first.size(), INTER_LINEAR, BORDER_REFLECT_101);
        Mat before = first.clone(), flow;
        calcOpticalFlowFarneback(first, second, flow, .5, 3, window, 5, 5, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN);
        ASSERT_TRUE(checkRange(flow));
        Scalar average = mean(flow(Rect(20, 20, width - 40, 56)));
        EXPECT_NEAR(1., average[0], .12);
        EXPECT_NEAR(-1., average[1], .12);
        EXPECT_EQ(0., cv::norm(before, first, NORM_INF));
        calcOpticalFlowFarneback(first, first, flow, .5, 3, window, 3, 5, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN);
        // Border polynomial fits can be ill-conditioned even for identical frames.
        EXPECT_LE(cv::norm(flow(Rect(20, 20, width - 40, 56)), NORM_INF), .01);
    }
}
}} // namespace
