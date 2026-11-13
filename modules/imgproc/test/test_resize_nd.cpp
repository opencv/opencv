// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

// pointer to plane p of an n-dimensional array, as a 2D Mat header
static Mat plane(const Mat& m, int p)
{
    const uchar* ptr = m.data;
    for (int i = m.dims - 3; i >= 0; i--)
    {
        ptr += (size_t)(p % m.size[i])*m.step[i];
        p /= m.size[i];
    }
    return Mat(m.size[m.dims - 2], m.size[m.dims - 1], m.type(), (void*)ptr, m.step[m.dims - 2]);
}

typedef testing::TestWithParam<tuple<perf::MatType, int> > Imgproc_ResizeND;

// every plane of the n-dimensional result equals the 2D resize of the plane
TEST_P(Imgproc_ResizeND, matches_2d)
{
    const int type = get<0>(GetParam()), interp = get<1>(GetParam());
    RNG& rng = theRNG();
    const std::vector<std::vector<int> > shapes = { {3, 17, 23}, {2, 3, 40, 31}, {2, 2, 2, 9, 12} };
    for (const auto& shape : shapes)
        for (int roi = 0; roi < 2; roi++)
        {
            std::vector<int> big = shape;
            std::vector<Range> ranges;
            for (size_t i = 0; i < shape.size(); i++)
            {
                big[i] += roi;
                ranges.push_back(Range(roi, roi + shape[i]));
            }
            Mat whole(big, type);
            rng.fill(whole, RNG::UNIFORM, 0, 200);
            Mat src = roi ? whole(ranges) : whole;
            const int H = shape[shape.size() - 2], W = shape.back();
            for (Size dsize : {Size(W*2 + 1, H*3), Size(W/2, H/2 + 1), Size(W, H)})
            {
                Mat dst;
                cv::resize(src, dst, dsize, 0, 0, interp);
                ASSERT_EQ(src.dims, dst.dims);
                int nplanes = (int)(src.total()/(H*W));
                for (int p = 0; p < nplanes; p++)
                {
                    Mat ref;
                    cv::resize(plane(src, p), ref, dsize, 0, 0, interp);
                    ASSERT_EQ(0, cvtest::norm(ref, plane(dst, p), NORM_INF))
                        << MatShape(shape).str() << " roi=" << roi << " dsize=" << dsize << " plane " << p;
                }
            }
            // scale factors instead of a size
            Mat d1, r1;
            cv::resize(src, d1, Size(), 1.5, 0.75, interp);
            cv::resize(plane(src, 0), r1, Size(), 1.5, 0.75, interp);
            ASSERT_EQ(0, cvtest::norm(r1, plane(d1, 0), NORM_INF));
        }
}

INSTANTIATE_TEST_CASE_P(/**/, Imgproc_ResizeND, testing::Combine(
    testing::Values(perf::MatType(CV_8UC1), CV_8UC3, CV_16UC2, CV_32FC1),
    testing::Values((int)INTER_NEAREST, (int)INTER_LINEAR, (int)INTER_CUBIC, (int)INTER_AREA,
                    (int)INTER_LANCZOS4, (int)INTER_NEAREST_EXACT, (int)INTER_LINEAR_EXACT)
));

// enough planes for the parallel path
TEST(Imgproc_ResizeNDPlanes, parallel)
{
    RNG& rng = theRNG();
    Mat src({1, 64, 30, 30}, CV_32F), dst;
    rng.fill(src, RNG::UNIFORM, -1, 1);
    cv::resize(src, dst, Size(60, 60), 0, 0, INTER_LINEAR);
    for (int p = 0; p < 64; p++)
    {
        Mat ref;
        cv::resize(plane(src, p), ref, Size(60, 60), 0, 0, INTER_LINEAR);
        ASSERT_EQ(0, cvtest::norm(ref, plane(dst, p), NORM_INF)) << p;
    }
}

}} // namespace
