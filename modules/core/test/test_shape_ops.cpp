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
    EXPECT_EQ(MatShape({1}), MatShape::scalar().flatten());
    EXPECT_ANY_THROW(MatShape::scalar().flatten(1));

    EXPECT_EQ(MatShape({6, 20}), u.reshape(MatShape({6, -1})));
    EXPECT_EQ(MatShape({2, 3, 20}), u.reshape(MatShape({0, 0, -1})));
    EXPECT_ANY_THROW(MatShape({0, 4}).reshape(MatShape({0, -1})));   // -1 next to a zero-size axis is ambiguous
    EXPECT_EQ(MatShape({0, 7}), MatShape({0, 4}).reshape(MatShape({0, 7}), true));
    EXPECT_ANY_THROW(u.reshape(MatShape({7, -1})));
    EXPECT_ANY_THROW(u.reshape(MatShape({-1, -1})));
    EXPECT_ANY_THROW(u.reshape(MatShape({121})));
}

// A block ('NC1HWC0') shape survives these operations as long as N and C are untouched.
TEST(Core_MatShape, shape_ops_block_layout)
{
    const MatShape nchw({2, 12, 5, 5}, DATA_LAYOUT_NCHW);
    const MatShape blk = nchw.toLayout(DATA_LAYOUT_BLOCK, 8);
    ASSERT_EQ(MatShape({2, 2, 5, 5, 8}, DATA_LAYOUT_BLOCK, 12), blk);
    ASSERT_EQ((size_t)800, blk.total());   // padded: C 12 rounds up to 2*8

    // the spatial axes may move
    const MatShape flat = blk.flatten(2);
    EXPECT_EQ(MatShape({2, 2, 25, 8}, DATA_LAYOUT_BLOCK, 12), flat);
    EXPECT_EQ(DATA_LAYOUT_BLOCK, flat.layout);
    EXPECT_EQ(12, flat.C);
    EXPECT_EQ(blk.total(), flat.total());
    EXPECT_EQ(nchw.flatten(2), flat.toLayout(DATA_LAYOUT_NCHW));

    EXPECT_EQ(MatShape({2, 2, 1, 5, 5, 8}, DATA_LAYOUT_BLOCK, 12), blk.unsqueeze({2}));
    EXPECT_EQ(MatShape({2, 2, 25, 8}, DATA_LAYOUT_BLOCK, 12), blk.reshape(MatShape({0, 0, -1})));
    EXPECT_EQ(MatShape({2, 2, 5, 5, 8}, DATA_LAYOUT_BLOCK, 12), blk.squeeze());

    const MatShape blk1 = MatShape({2, 12, 1, 5}, DATA_LAYOUT_NCHW).toLayout(DATA_LAYOUT_BLOCK, 8);
    EXPECT_EQ(MatShape({2, 2, 5, 8}, DATA_LAYOUT_BLOCK, 12), blk1.squeeze());

    // moving N or C would repack the padded C0 lanes
    EXPECT_ANY_THROW(blk.flatten());
    EXPECT_ANY_THROW(blk.flatten(0, 1));
    EXPECT_ANY_THROW(blk.unsqueeze({0}));
    EXPECT_ANY_THROW(blk.reshape(MatShape({2, 6, 50})));
    EXPECT_ANY_THROW(blk.reshape(MatShape({4, 12, 25})));
    EXPECT_ANY_THROW(blk.reshape(MatShape({-1})));

    const MatShape blkN1 = MatShape({1, 12, 5, 5}, DATA_LAYOUT_NCHW).toLayout(DATA_LAYOUT_BLOCK, 8);
    EXPECT_ANY_THROW(blkN1.squeeze({0}));

    // the counts a caller addresses are logical: C == 12, not the padded C1 * C0 == 16
    EXPECT_EQ(MatShape({2, 12, 25}, DATA_LAYOUT_NCHW), blk.toLayout(DATA_LAYOUT_NCHW).flatten(2));
    EXPECT_ANY_THROW(blk.reshape(MatShape({2, 16, 25})));
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

    // a preallocated strided destination of the target shape gets a copy, not an assert
    Mat canvas(10, 10, CV_8U, Scalar(0)), view = canvas(Rect(0, 0, 4, 5));
    Mat src5x4({1, 5, 4}, CV_8U);
    rng.fill(src5x4, RNG::UNIFORM, 0, 256);
    cv::squeeze(src5x4, view);
    ASSERT_EQ(std::vector<int>({5, 4}), shapeOf(view));
    EXPECT_FALSE(view.isContinuous());
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 4; j++)
            EXPECT_EQ(src5x4.at<uchar>(0, i, j), view.at<uchar>(i, j));

    // to and from 0-d
    Mat one({1, 1}, CV_32F, Scalar(2)), sc;
    cv::squeeze(one, sc);
    EXPECT_EQ(0, sc.dims);
    EXPECT_EQ(2.f, sc.at<float>(0));
}

}} // namespace
