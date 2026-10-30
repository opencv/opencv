// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

static void makeFramePair(Size sz, Mat& frame1, Mat& frame2)
{
    // Synthetic frame pair: random low-frequency texture + small displacement field.
    int src_scale = 3;
    Mat tmp(Size(sz.width >> src_scale, sz.height >> src_scale), CV_8U);
    randu(tmp, 0, 255);
    resize(tmp, frame1, sz, 0, 0, INTER_LINEAR);

    // Build a smooth displacement field and remap to create frame2.
    Mat disp(Size(sz.width >> 4, sz.height >> 4), CV_32FC2);
    randn(disp, 0.0, 3.0);
    resize(disp, disp, sz, 0, 0, INTER_CUBIC);
    for (int i = 0; i < disp.rows; i++)
        for (int j = 0; j < disp.cols; j++)
            disp.at<Vec2f>(i, j) += Vec2f((float)j, (float)i);

    Mat frame1f;
    frame1.convertTo(frame1f, CV_32F);
    Mat frame2f;
    remap(frame1f, frame2f, disp, Mat(), INTER_LINEAR, BORDER_REPLICATE);
    frame2f.convertTo(frame2, CV_8U);
}

typedef tuple<Size, int> FarnebackParams;
typedef TestBaseWithParam<FarnebackParams> DenseOpticalFlow_Farneback;

PERF_TEST_P(DenseOpticalFlow_Farneback, calcOpticalFlowFarneback,
            Combine(
                Values(szVGA, sz720p),
                Values(5, 7)          // polyN
            ))
{
    Size sz    = get<0>(GetParam());
    int polyN  = get<1>(GetParam());

    double polySigma = polyN == 5 ? 1.1 : 1.5;

    Mat frame1(sz, CV_8U);
    Mat frame2(sz, CV_8U);
    makeFramePair(sz, frame1, frame2);

    Mat flow;

    TEST_CYCLE_N(10)
    {
        calcOpticalFlowFarneback(frame1, frame2, flow,
                                 0.5,  // pyr_scale
                                 3,    // levels
                                 15,   // winsize
                                 3,    // iterations
                                 polyN, polySigma, 0);
    }

    SANITY_CHECK_NOTHING();
}

}} // namespace
