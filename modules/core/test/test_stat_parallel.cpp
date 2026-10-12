// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// The parallel versions of sum, mean, meanStdDev and minMaxIdx must agree with the serial ones.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

static double relDiff(const Scalar& a, const Scalar& b)
{
    double d = 0;
    for (int k = 0; k < 4; k++)
        d = std::max(d, std::abs(a[k] - b[k])/(std::abs(b[k]) + 1));
    return d;
}

typedef testing::TestWithParam<tuple<perf::MatType, int> > Core_StatParallel;

TEST_P(Core_StatParallel, matches_serial)
{
    const int type = get<0>(GetParam()), variant = get<1>(GetParam());
    const int depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    RNG& rng = theRNG();

    // variant 0: continuous 2D, 1: ROI, 2: continuous 3D, 3: 2D with a mask
    Mat big(1100, 1300, type);
    rng.fill(big, RNG::UNIFORM, depth == CV_8U ? 0 : -100, 100);
    Mat src = variant == 1 ? big(Rect(7, 5, 1200, 1000)) : variant == 2 ? big.reshape(cn, std::vector<int>{11, 100, 1300}) : big;
    Mat mask;
    if (variant == 3)
    {
        mask.create(src.size(), CV_8U);
        rng.fill(mask, RNG::UNIFORM, 0, 2);
        mask.rowRange(0, mask.rows*3/4).setTo(0);   // most of the pieces see no elements
    }

    Scalar s1, m1, mean1, sd1;
    double mn1 = 0, mx1 = 0;
    int imn1[3] = {-1, -1, -1}, imx1[3] = {-1, -1, -1};
    const bool withIdx = cn == 1;
    Mat src1 = cn == 1 ? src : src.reshape(1);
    s1 = cv::sum(src);
    m1 = cv::mean(src, mask);
    cv::meanStdDev(src, mean1, sd1, mask);
    cv::minMaxIdx(src1, &mn1, &mx1, withIdx ? imn1 : 0, withIdx ? imx1 : 0, cn == 1 ? mask : noArray());

    // Reference from per-row calls: every row is far below the size at which the work is split,
    // so this runs the original serial code, and the rows are combined here in double.
    Mat src2 = src.dims > 2 ? Mat(src.size[0]*src.size[1], src.size[2], src.type(), src.data) : src;
    Mat src2s = src1.dims > 2 ? Mat(src1.size[0]*src1.size[1], src1.size[2], src1.type(), src1.data) : src1;
    Scalar s0, sx, sxx;
    double n0 = 0, mn0 = 0, mx0 = 0;
    bool anyRow = false;
    for (int r = 0; r < src2.rows; r++)
    {
        Mat row = src2.row(r), mrow = mask.empty() ? Mat() : mask.row(r), row64;
        s0 += cv::sum(row);
        row.convertTo(row64, CV_64F);
        double cnt = mrow.empty() ? row.cols : cv::countNonZero(mrow);
        if (cnt > 0)
        {
            sx += cv::mean(row64, mrow)*cnt;
            sxx += cv::mean(row64.mul(row64), mrow)*cnt;
            n0 += cnt;
        }
        double a, b;
        int ia[2] = {-1, -1}, ib[2] = {-1, -1};
        cv::minMaxIdx(src2s.row(r), &a, &b, withIdx ? ia : 0, withIdx ? ib : 0, cn == 1 ? mrow : Mat());
        if (withIdx && ia[1] < 0)
            continue;                                   // the mask selects nothing in this row
        if (!anyRow || a < mn0) mn0 = a;
        if (!anyRow || b > mx0) mx0 = b;
        anyRow = true;
    }
    Scalar m0 = n0 > 0 ? sx*(1./n0) : Scalar(), mean0 = m0, sd0;
    for (int k = 0; k < 4; k++)
        sd0[k] = n0 > 0 ? std::sqrt(std::max(sxx[k]/n0 - m0[k]*m0[k], 0.)) : 0;

    const double tol = depth == CV_32F ? 1e-6 : 1e-10;
    EXPECT_LE(relDiff(s1, s0), tol);
    EXPECT_LE(relDiff(m1, m0), tol);
    EXPECT_LE(relDiff(mean1, mean0), tol);
    EXPECT_LE(relDiff(sd1, sd0), 1e-6);
    EXPECT_EQ(mn0, mn1);
    EXPECT_EQ(mx0, mx1);
    if (withIdx)
    {
        // Which of several equal extrema a piece reports is up to its (vectorized, possibly HAL)
        // serial code, so check that the position holds the extremum rather than that it matches
        // the reference; the ordering across pieces is covered by first_occurrence below.
        auto valAt = [&](const int* idx)
        {
            Mat e(1, 1, src1.type(), src1.ptr(idx)), e64;
            e.convertTo(e64, CV_64F);
            return e64.at<double>(0);
        };
        ASSERT_GE(imn1[0], 0);
        ASSERT_GE(imx1[0], 0);
        EXPECT_EQ(mn1, valAt(imn1));
        EXPECT_EQ(mx1, valAt(imx1));
        if (!mask.empty())
        {
            EXPECT_NE(0, mask.at<uchar>(imn1[0], imn1[1]));
            EXPECT_NE(0, mask.at<uchar>(imx1[0], imx1[1]));
        }
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Core_StatParallel, testing::Combine(
    testing::Values(perf::MatType(CV_8UC1), CV_8UC3, CV_16SC1, CV_32SC1, CV_32FC1, CV_32FC4, CV_64FC1),
    testing::Values(0, 1, 2, 3)
));

// the first occurrence of the extremum wins, also across pieces
TEST(Core_StatParallelMinMax, first_occurrence)
{
    Mat a(1000, 1000, CV_32F, Scalar(0));
    a.at<float>(300, 5) = -1; a.at<float>(900, 7) = -1;
    a.at<float>(100, 9) = 2;  a.at<float>(700, 1) = 2;
    double mn, mx;
    Point pmn, pmx;
    cv::minMaxLoc(a, &mn, &mx, &pmn, &pmx);
    EXPECT_EQ(Point(5, 300), pmn);
    EXPECT_EQ(Point(9, 100), pmx);

    Mat mask(a.size(), CV_8U, Scalar(0));                // nothing selected
    cv::minMaxLoc(a, &mn, &mx, &pmn, &pmx, mask);
    EXPECT_EQ(Point(-1, -1), pmn);
    EXPECT_EQ(0., mn);
    EXPECT_EQ(Scalar::all(0), cv::mean(a, mask));
}

}} // namespace
