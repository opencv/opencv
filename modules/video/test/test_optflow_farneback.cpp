// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

static void makeFramePair(Size sz, Mat& frame1, Mat& frame2)
{
    RNG& rng = theRNG();
    rng.state = 0x12345678;

    Mat tmp(Size(sz.width >> 3, sz.height >> 3), CV_8U);
    randu(tmp, 0, 255);
    resize(tmp, frame1, sz, 0, 0, INTER_LINEAR);

    Mat disp(Size(sz.width >> 4, sz.height >> 4), CV_32FC2);
    randn(disp, 0.0, 3.0);
    resize(disp, disp, sz, 0, 0, INTER_CUBIC);
    for (int i = 0; i < disp.rows; i++)
        for (int j = 0; j < disp.cols; j++)
            disp.at<Vec2f>(i, j) += Vec2f((float)j, (float)i);

    remap(frame1, frame2, disp, Mat(), INTER_LINEAR, BORDER_REPLICATE);
}

typedef testing::TestWithParam<tuple<int, int> > DenseOpticalFlow_Farneback_Threads;

TEST_P(DenseOpticalFlow_Farneback_Threads, single_vs_multi_thread)
{
    const int polyN = get<0>(GetParam());
    const int flags = get<1>(GetParam());
    const double polySigma = polyN == 5 ? 1.1 : 1.5;

    Mat frame1, frame2;
    makeFramePair(Size(640, 480), frame1, frame2);

    const int nthreads = getNumThreads();

    Mat flow_st, flow_mt;
    setNumThreads(1);
    calcOpticalFlowFarneback(frame1, frame2, flow_st, 0.5, 3, 15, 3, polyN, polySigma, flags);
    setNumThreads(nthreads);
    calcOpticalFlowFarneback(frame1, frame2, flow_mt, 0.5, 3, 15, 3, polyN, polySigma, flags);

    ASSERT_EQ(CV_32FC2, flow_st.type());
    EXPECT_EQ(0, cvtest::norm(flow_st, flow_mt, NORM_INF));
}

INSTANTIATE_TEST_CASE_P(/**/, DenseOpticalFlow_Farneback_Threads,
    testing::Combine(
        testing::Values(5, 7),
        testing::Values(0, (int)OPTFLOW_FARNEBACK_GAUSSIAN)
    ));

}} // namespace
