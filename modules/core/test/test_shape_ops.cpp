// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

static std::vector<int> shapeOf(const Mat& m) { return std::vector<int>(m.size.p, m.size.p + m.dims); }

TEST(Core_MatShape, shape_ops)
{
    MatShape s({2, 1, 3, 1});
    EXPECT_EQ(MatShape({2, 3}), s.squeeze());
    EXPECT_EQ(MatShape({2, 3, 1}), s.squeeze({1}));
    EXPECT_EQ(MatShape({2, 1, 3}), s.squeeze({-1}));
    EXPECT_ANY_THROW(s.squeeze({0}));
    EXPECT_ANY_THROW(s.squeeze({1, -3}));
    EXPECT_EQ(MatShape::scalar(), MatShape({1, 1}).squeeze());

    MatShape t({2, 3});
    EXPECT_EQ(MatShape({1, 2, 3}), t.unsqueeze({0}));
    EXPECT_EQ(MatShape({2, 3, 1}), t.unsqueeze({-1}));
    EXPECT_EQ(MatShape({1, 2, 1, 3}), t.unsqueeze({0, 2}));
    EXPECT_ANY_THROW(t.unsqueeze({0, 0}));
    EXPECT_EQ(MatShape({1}), MatShape::scalar().unsqueeze({0}));

    MatShape u({2, 3, 4, 5});
    EXPECT_EQ(MatShape({120}), u.flatten());
    EXPECT_EQ(MatShape({2, 60}), u.flatten(1));
    EXPECT_EQ(MatShape({2, 12, 5}), u.flatten(1, 2));
    EXPECT_EQ(MatShape({2, 3, 4, 5}), u.flatten(2, 2));
    EXPECT_ANY_THROW(u.flatten(2, 1));

    EXPECT_EQ(MatShape({6, 20}), u.reshape(MatShape({6, -1})));
    EXPECT_EQ(MatShape({2, 3, 20}), u.reshape(MatShape({0, 0, -1})));
    EXPECT_ANY_THROW(MatShape({0, 4}).reshape(MatShape({0, -1})));   // -1 next to a zero-size axis is ambiguous
    EXPECT_EQ(MatShape({0, 7}), MatShape({0, 4}).reshape(MatShape({0, 7}), true));
    EXPECT_ANY_THROW(u.reshape(MatShape({7, -1})));
    EXPECT_ANY_THROW(u.reshape(MatShape({-1, -1})));
    EXPECT_ANY_THROW(u.reshape(MatShape({121})));
}

TEST(Core_ShapeOps, views_and_copies)
{
    RNG& rng = theRNG();
    Mat a({2, 1, 3, 4}, CV_32FC2);
    rng.fill(a, RNG::UNIFORM, -1, 1);

    Mat s;
    cv::squeeze(a, s);
    EXPECT_EQ(std::vector<int>({2, 3, 4}), shapeOf(s));
    EXPECT_NE(a.data, s.data);                       // the output does not alias the input
    EXPECT_EQ(CV_32FC2, s.type());
    EXPECT_EQ(0, cvtest::norm(Mat(1, 48, CV_32FC1, a.data), Mat(1, 48, CV_32FC1, s.data), NORM_INF));

    Mat u, f, r;
    cv::unsqueeze(a, u, {0, -1});
    EXPECT_EQ(std::vector<int>({1, 2, 1, 3, 4, 1}), shapeOf(u));
    cv::flatten(a, f, 1);
    EXPECT_EQ(std::vector<int>({2, 12}), shapeOf(f));
    cv::reshape(a, r, {-1, 6});
    EXPECT_EQ(std::vector<int>({4, 6}), shapeOf(r));
    EXPECT_EQ(0, cvtest::norm(Mat(1, 48, CV_32FC1, a.data), Mat(1, 48, CV_32FC1, r.data), NORM_INF));

    // a preallocated destination of the new shape gets a copy
    Mat pre({2, 3, 4}, CV_32FC2, Scalar::all(0));
    uchar* p0 = pre.data;
    cv::squeeze(a, pre);
    EXPECT_EQ(p0, pre.data);
    EXPECT_EQ(0, cvtest::norm(Mat(1, 48, CV_32FC1, a.data), Mat(1, 48, CV_32FC1, pre.data), NORM_INF));

    // a non-continuous source is copied
    Mat big({4, 6}, CV_8U), roi = big(Rect(1, 1, 4, 2)), fr;
    rng.fill(big, RNG::UNIFORM, 0, 256);
    cv::flatten(roi, fr);
    ASSERT_EQ(std::vector<int>({8}), shapeOf(fr));
    for (int i = 0; i < 8; i++)
        EXPECT_EQ(roi.at<uchar>(i/4, i%4), fr.at<uchar>(i));

    // in place: only the header changes
    Mat b({1, 5}, CV_16S, Scalar(3));
    const uchar* bdata = b.data;
    cv::squeeze(b, b);
    EXPECT_EQ(std::vector<int>({5}), shapeOf(b));
    EXPECT_EQ(bdata, b.data);

    // to and from 0-d
    Mat one({1, 1}, CV_32F, Scalar(2)), sc;
    cv::squeeze(one, sc);
    EXPECT_EQ(0, sc.dims);
    EXPECT_EQ(2.f, sc.at<float>(0));
}

}} // namespace
