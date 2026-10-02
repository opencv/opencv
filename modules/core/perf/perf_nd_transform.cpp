// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"

namespace opencv_test {
using namespace perf;

// declare.in() only handles 2D arrays
static void fillRandom(Mat& m)
{
    CV_Assert(m.isContinuous());
    Mat bytes(1, (int)(m.total()*m.elemSize()), CV_8U, m.data);
    randu(bytes, 0, 256);
}

typedef tuple<std::vector<int>, std::vector<int>, perf::MatType> TransposeNDParams;
typedef TestBaseWithParam<TransposeNDParams> TransposeNDPerf;

PERF_TEST_P_(TransposeNDPerf, transposeND)
{
    const std::vector<int> shape = get<0>(GetParam()), order = get<1>(GetParam());
    const int type = get<2>(GetParam());
    Mat src(shape, type), dst;
    fillRandom(src);
    cv::transposeND(src, order, dst);

    TEST_CYCLE() cv::transposeND(src, order, dst);

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, TransposeNDPerf, testing::Values(
    // NCHW <-> NHWC
    make_tuple(std::vector<int>{1, 64, 128, 128}, std::vector<int>{0, 2, 3, 1}, perf::MatType(CV_32FC1)),
    make_tuple(std::vector<int>{1, 64, 128, 128}, std::vector<int>{0, 2, 3, 1}, perf::MatType(CV_16FC1)),
    make_tuple(std::vector<int>{1, 3, 640, 640}, std::vector<int>{0, 2, 3, 1}, perf::MatType(CV_32FC1)),
    make_tuple(std::vector<int>{1, 640, 640, 3}, std::vector<int>{0, 3, 1, 2}, perf::MatType(CV_32FC1)),
    make_tuple(std::vector<int>{1, 640, 640, 3}, std::vector<int>{0, 3, 1, 2}, perf::MatType(CV_8UC1)),
    // batched matrix transpose and attention head reordering
    make_tuple(std::vector<int>{8, 512, 256}, std::vector<int>{0, 2, 1}, perf::MatType(CV_32FC1)),
    make_tuple(std::vector<int>{2, 196, 8, 64}, std::vector<int>{0, 2, 1, 3}, perf::MatType(CV_32FC1)),
    make_tuple(std::vector<int>{2, 16, 8, 16, 32}, std::vector<int>{0, 4, 2, 1, 3}, perf::MatType(CV_32FC1)),
    // multi-channel input
    make_tuple(std::vector<int>{480, 640}, std::vector<int>{1, 0}, perf::MatType(CV_8UC3)),
    make_tuple(std::vector<int>{4, 240, 320}, std::vector<int>{2, 0, 1}, perf::MatType(CV_8UC3))
));

typedef tuple<std::vector<int>, int, int> ConcatNDParams;
typedef TestBaseWithParam<ConcatNDParams> ConcatNDPerf;

PERF_TEST_P_(ConcatNDPerf, concatND)
{
    const std::vector<int> shape = get<0>(GetParam());
    const int axis = get<1>(GetParam()), n = get<2>(GetParam());
    std::vector<Mat> src(n);
    for (int k = 0; k < n; k++)
    {
        src[k].create(shape, CV_32F);
        fillRandom(src[k]);
    }
    Mat dst;
    cv::concatND(src, axis, dst);

    TEST_CYCLE() cv::concatND(src, axis, dst);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(ConcatNDPerf, splitND)
{
    std::vector<int> shape = get<0>(GetParam());
    const int axis = get<1>(GetParam()), n = get<2>(GetParam());
    std::vector<int> sizes(n, shape[axis]);
    shape[axis] *= n;
    Mat src(shape, CV_32F);
    fillRandom(src);
    std::vector<Mat> dst;
    cv::splitND(src, axis, sizes, dst);

    TEST_CYCLE() cv::splitND(src, axis, sizes, dst);

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, ConcatNDPerf, testing::Values(
    make_tuple(std::vector<int>{1, 64, 80, 80}, 1, 4),
    make_tuple(std::vector<int>{1, 64, 80, 80}, 3, 2),
    make_tuple(std::vector<int>{1, 128, 40, 40}, 2, 3),
    make_tuple(std::vector<int>{1, 16, 8, 8}, 1, 3)
));

typedef tuple<std::vector<int>, std::vector<int> > TileNDParams;
typedef TestBaseWithParam<TileNDParams> TileNDPerf;

PERF_TEST_P_(TileNDPerf, tileND)
{
    const std::vector<int> shape = get<0>(GetParam()), repeats = get<1>(GetParam());
    Mat src(shape, CV_32F), dst;
    fillRandom(src);
    cv::tileND(src, repeats, dst);

    TEST_CYCLE() cv::tileND(src, repeats, dst);

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, TileNDPerf, testing::Values(
    make_tuple(std::vector<int>{1, 64, 1, 128}, std::vector<int>{1, 1, 64, 1}),
    make_tuple(std::vector<int>{8, 64}, std::vector<int>{16, 4}),
    make_tuple(std::vector<int>{1, 3, 160, 160}, std::vector<int>{1, 1, 2, 2})
));

typedef tuple<int, int> SliceNDParams;  // axis, step
typedef TestBaseWithParam<SliceNDParams> SliceNDPerf;

PERF_TEST_P_(SliceNDPerf, sliceND)
{
    const int axis = get<0>(GetParam()), step = get<1>(GetParam());
    Mat src({64, 128, 128}, CV_32F), dst;
    fillRandom(src);
    std::vector<int> starts(axis + 1, 0), ends(src.size.p, src.size.p + axis + 1), steps(axis + 1, 1);
    steps[axis] = step;
    if (step < 0)
    {
        starts[axis] = src.size[axis] - 1;
        ends[axis] = -1;
    }
    else
    {
        starts[axis] = step == 1 ? 10 : 0;
        ends[axis] = step == 1 ? src.size[axis] - 10 : src.size[axis];
    }
    cv::sliceND(src, starts, ends, steps, dst);

    TEST_CYCLE() cv::sliceND(src, starts, ends, steps, dst);

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/**/, SliceNDPerf, testing::Combine(
    testing::Values(0, 1, 2),
    testing::Values(1, 2, -1)
));

PERF_TEST_P_(TransposeNDPerf, flipND)
{
    const std::vector<int> shape = get<0>(GetParam());
    const int type = get<2>(GetParam());
    Mat src(shape, type), dst;
    fillRandom(src);
    const int axis = (int)shape.size() - 1;

    TEST_CYCLE() cv::flipND(src, dst, axis);

    SANITY_CHECK_NOTHING();
}

} // namespace
