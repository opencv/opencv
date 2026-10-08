// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, Intel Corporation, all rights reserved.
#include "perf_precomp.hpp"
#include <numeric>
#include "opencv2/core/softfloat.hpp"

namespace opencv_test
{
using namespace perf;

// Shared BinaryOpTest coverage.
#define BINARY_OP_TYPES CV_8UC1, CV_8UC3, CV_8UC4, CV_8SC1, CV_16UC1, CV_16SC1, CV_16SC2, \
                        CV_16SC3, CV_16SC4, CV_32SC1, CV_32FC1, CV_64FC1
// Integer types accepting a scale factor.
#define ARITHM_SCALE_TYPES (perf::MatType)CV_8UC1, (perf::MatType)CV_16UC1, (perf::MatType)CV_16SC1
// Floating-point-only ops (cv::pow / sqrt).
#define ARITHM_FLOAT_TYPES CV_32FC1, CV_64FC1

using BroadcastTest = perf::TestBaseWithParam<std::tuple<std::vector<int>, perf::MatType, std::vector<int>>>;
typedef Size_MatType BinaryOpTest;

PERF_TEST_P_(BroadcastTest, basic)
{
    std::vector<int> shape_src = get<0>(GetParam());
    int dt_type = get<1>(GetParam());
    std::vector<int> shape_dst = get<2>(GetParam());

    cv::Mat src(static_cast<int>(shape_src.size()), shape_src.data(), dt_type);
    cv::Mat dst(static_cast<int>(shape_dst.size()), shape_dst.data(), dt_type);

    cv::randu(src, -1.f, 1.f);

    TEST_CYCLE() cv::broadcast(src, shape_dst, dst);

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/*nothing*/ , BroadcastTest,
    testing::Combine(
        testing::Values(std::vector<int>{1, 100, 800},
                        std::vector<int>{10, 1, 800},
                        std::vector<int>{10, 100, 1}),
        testing::Values(CV_32FC1),
        testing::Values(std::vector<int>{10, 100, 800})
    )
);

PERF_TEST_P_(BinaryOpTest, min)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::min(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, minScalarDouble)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::min(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, minScalarSameType)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) < CV_32S)
    {
        b = Scalar(1, 0, 3, 4); // don't pass non-integer values for 8U/8S/16U/16S processing
    }
    else if (CV_MAT_DEPTH(type) == CV_32S)
    {
        b = Scalar(1, 0, -3, 4); // don't pass non-integer values for 32S processing
    }

    TEST_CYCLE() cv::min(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, max)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::max(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, maxScalarDouble)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::max(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, maxScalarSameType)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) < CV_32S)
    {
        b = Scalar(1, 0, 3, 4); // don't pass non-integer values for 8U/8S/16U/16S processing
    }
    else if (CV_MAT_DEPTH(type) == CV_32S)
    {
        b = Scalar(1, 0, -3, 4); // don't pass non-integer values for 32S processing
    }

    TEST_CYCLE() cv::max(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, absdiff)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: absdiff can be without saturation on 32S
        a /= 2;
        b /= 2;
    }

    TEST_CYCLE() cv::absdiff(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, absdiffScalarDouble)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: absdiff can be without saturation on 32S
        a /= 2;
        b /= 2;
    }

    TEST_CYCLE() cv::absdiff(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, absdiffScalarSameType)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) < CV_32S)
    {
        b = Scalar(1, 0, 3, 4); // don't pass non-integer values for 8U/8S/16U/16S processing
    }
    else if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: absdiff can be without saturation on 32S
        a /= 2;
        b = Scalar(1, 0, -3, 4); // don't pass non-integer values for 32S processing
    }

    TEST_CYCLE() cv::absdiff(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, add)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);
    declare.time(50);

    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: add can be without saturation on 32S
        a /= 2;
        b /= 2;
    }

    TEST_CYCLE() cv::add(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, addScalarDouble)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: add can be without saturation on 32S
        a /= 2;
        b /= 2;
    }

    TEST_CYCLE() cv::add(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, addScalarSameType)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) < CV_32S)
    {
        b = Scalar(1, 0, 3, 4); // don't pass non-integer values for 8U/8S/16U/16S processing
    }
    else if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: add can be without saturation on 32S
        a /= 2;
        b = Scalar(1, 0, -3, 4); // don't pass non-integer values for 32S processing
    }

    TEST_CYCLE() cv::add(a, b, c, noArray(), type);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, subtract)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Mat b = Mat(sz, type);
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: subtract can be without saturation on 32S
        a /= 2;
        b /= 2;
    }

    TEST_CYCLE() cv::subtract(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, subtractScalarDouble)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: subtract can be without saturation on 32S
        a /= 2;
        b /= 2;
    }

    TEST_CYCLE() cv::subtract(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, subtractScalarSameType)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type);
    cv::Scalar b;
    cv::Mat c = Mat(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) < CV_32S)
    {
        b = Scalar(1, 0, 3, 4); // don't pass non-integer values for 8U/8S/16U/16S processing
    }
    else if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //see ticket 1529: subtract can be without saturation on 32S
        a /= 2;
        b = Scalar(1, 0, -3, 4); // don't pass non-integer values for 32S processing
    }

    TEST_CYCLE() cv::subtract(a, b, c, noArray(), type);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, multiply)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a(sz, type), b(sz, type), c(sz, type);

    declare.in(a, b, WARMUP_RNG).out(c);
    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //According to docs, saturation is not applied when result is 32bit integer
        a /= (2 << 16);
        b /= (2 << 16);
    }

    TEST_CYCLE() cv::multiply(a, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, multiplyScale)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a(sz, type), b(sz, type), c(sz, type);
    double scale = 0.5;

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        //According to docs, saturation is not applied when result is 32bit integer
        a /= (2 << 16);
        b /= (2 << 16);
    }

    TEST_CYCLE() cv::multiply(a, b, c, scale);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, multiplyScalar)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a(sz, type), c(sz, type);
    cv::Scalar s = 1.333;

    declare.in(a, WARMUP_RNG).out(c);
    if (CV_MAT_DEPTH(type) == CV_32S)
    {
        a /= (2 << 16);
        s = 2.0;
    }

    TEST_CYCLE() cv::multiply(a, s, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, divide)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a(sz, type), b(sz, type), c(sz, type);
    double scale = 0.5;

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::divide(a, b, c, scale);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, reciprocal)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat b(sz, type), c(sz, type);
    double scale = 0.5;

    declare.in(b, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::divide(scale, b, c);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, transpose2d)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    Size tsz = Size(sz.height, sz.width);
    cv::Mat a(sz, type), b(tsz, type);;

    declare.in(a, WARMUP_RNG).out(b);

    TEST_CYCLE() cv::transpose(a, b);

    SANITY_CHECK_NOTHING();
}

#define TRANSPOSE_TYPES CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC1, CV_16UC3, CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4

typedef Size_MatType TransposeMultiChannelTest;

PERF_TEST_P_(TransposeMultiChannelTest, transpose2d_multichannel)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    Size tsz = Size(sz.height, sz.width);
    cv::Mat a(sz, type), b(tsz, type);

    declare.in(a, WARMUP_RNG).out(b);

    TEST_CYCLE() cv::transpose(a, b);

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/*nothing*/ , TransposeMultiChannelTest,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p),
        testing::Values(TRANSPOSE_TYPES)
    )
);

typedef Size_MatType TransposeInplaceTest;

PERF_TEST_P_(TransposeInplaceTest, transpose2d_inplace)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a(sz, type), b(sz, type);

    declare.in(a, WARMUP_RNG).out(b);

    while (next())
    {
        a.copyTo(b);
        startTimer();
        cv::transpose(b, b);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/*nothing*/ , TransposeInplaceTest,
    testing::Combine(
        testing::Values(cv::Size(1280, 1280), cv::Size(1920, 1920)),
        testing::Values(TRANSPOSE_TYPES)
    )
);

static cv::Mat makeTransposeNDOutput(const cv::Mat& src, const std::vector<int>& order)
{
    std::vector<int> new_sz(order.size());
    for (size_t i = 0; i < order.size(); ++i)
        new_sz[i] = src.size[order[i]];
    return Mat(static_cast<int>(new_sz.size()), new_sz.data(), src.type());
}

static cv::Mat makeTransposeNDInput3D(Size sz, int type)
{
    const int channels = CV_MAT_CN(type);
    int dims[] = { sz.height, sz.width, channels };
    return Mat(3, dims, CV_MAKETYPE(CV_MAT_DEPTH(type), 1));
}

PERF_TEST_P_(BinaryOpTest, transposeND_identity_order)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type).reshape(1);

    std::vector<int> order(a.dims);
    std::iota(order.begin(), order.end(), 0);

    cv::Mat b = makeTransposeNDOutput(a, order);

    declare.in(a, WARMUP_RNG).out(b);

    TEST_CYCLE() cv::transposeND(a, order, b);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, transposeND_2d_swap_order)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = Mat(sz, type).reshape(1);

    std::vector<int> order{1, 0};

    cv::Mat b = makeTransposeNDOutput(a, order);

    declare.in(a, WARMUP_RNG).out(b);

    TEST_CYCLE() cv::transposeND(a, order, b);

    SANITY_CHECK_NOTHING();
}


PERF_TEST_P_(BinaryOpTest, transposeND_generic_keep_tail_order)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = makeTransposeNDInput3D(sz, type);

    std::vector<int> order{1, 0, 2};

    cv::Mat b = makeTransposeNDOutput(a, order);

    randu(a, 0, 255);
    declare.in(a).out(b);

    TEST_CYCLE() cv::transposeND(a, order, b);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BinaryOpTest, transposeND_generic_move_tail_order)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    cv::Mat a = makeTransposeNDInput3D(sz, type);

    std::vector<int> order{2, 0, 1};

    cv::Mat b = makeTransposeNDOutput(a, order);

    randu(a, 0, 255);
    declare.in(a).out(b);

    TEST_CYCLE() cv::transposeND(a, order, b);

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/*nothing*/ , BinaryOpTest,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p),
        testing::Values(BINARY_OP_TYPES)
    )
);

///////////// Mixed type arithmetics ////////

typedef perf::TestBaseWithParam<std::tuple<cv::Size, std::tuple<perf::MatType, perf::MatType>>> ArithmMixedTest;

PERF_TEST_P_(ArithmMixedTest, add)
{
    auto p = GetParam();
    Size sz = get<0>(p);
    int srcType = get<0>(get<1>(p));
    int dstType = get<1>(get<1>(p));

    cv::Mat a = Mat(sz, srcType);
    cv::Mat b = Mat(sz, srcType);
    cv::Mat c = Mat(sz, dstType);

    declare.in(a, b, WARMUP_RNG).out(c);
    declare.time(50);

    if (CV_MAT_DEPTH(dstType) == CV_32S)
    {
        //see ticket 1529: add can be without saturation on 32S
        a /= 2;
        b /= 2;
    }

    TEST_CYCLE() cv::add(a, b, c, /* mask */ noArray(), dstType);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(ArithmMixedTest, addScalarDouble)
{
    auto p = GetParam();
    Size sz = get<0>(p);
    int srcType = get<0>(get<1>(p));
    int dstType = get<1>(get<1>(p));

    cv::Mat a = Mat(sz, srcType);
    cv::Scalar b;
    cv::Mat c = Mat(sz, dstType);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(dstType) == CV_32S)
    {
        //see ticket 1529: add can be without saturation on 32S
        a /= 2;
        b /= 2;
    }

    TEST_CYCLE() cv::add(a, b, c, /* mask */ noArray(), dstType);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(ArithmMixedTest, addScalarSameType)
{
    auto p = GetParam();
    Size sz = get<0>(p);
    int srcType = get<0>(get<1>(p));
    int dstType = get<1>(get<1>(p));

    cv::Mat a = Mat(sz, srcType);
    cv::Scalar b;
    cv::Mat c = Mat(sz, dstType);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(dstType) < CV_32S)
    {
        b = Scalar(1, 0, 3, 4); // don't pass non-integer values for 8U/8S/16U/16S processing
    }
    else if (CV_MAT_DEPTH(dstType) == CV_32S)
    {
        //see ticket 1529: add can be without saturation on 32S
        a /= 2;
        b = Scalar(1, 0, -3, 4); // don't pass non-integer values for 32S processing
    }

    TEST_CYCLE() cv::add(a, b, c, /* mask */ noArray(), dstType);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(ArithmMixedTest, subtract)
{
    auto p = GetParam();
    Size sz = get<0>(p);
    int srcType = get<0>(get<1>(p));
    int dstType = get<1>(get<1>(p));

    cv::Mat a = Mat(sz, srcType);
    cv::Mat b = Mat(sz, srcType);
    cv::Mat c = Mat(sz, dstType);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(dstType) == CV_32S)
    {
        //see ticket 1529: subtract can be without saturation on 32S
        a /= 2;
        b /= 2;
    }

    TEST_CYCLE() cv::subtract(a, b, c, /* mask */ noArray(), dstType);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(ArithmMixedTest, subtractScalarDouble)
{
    auto p = GetParam();
    Size sz = get<0>(p);
    int srcType = get<0>(get<1>(p));
    int dstType = get<1>(get<1>(p));

    cv::Mat a = Mat(sz, srcType);
    cv::Scalar b;
    cv::Mat c = Mat(sz, dstType);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(dstType) == CV_32S)
    {
        //see ticket 1529: subtract can be without saturation on 32S
        a /= 2;
        b /= 2;
    }

    TEST_CYCLE() cv::subtract(a, b, c, /* mask */ noArray(), dstType);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(ArithmMixedTest, subtractScalarSameType)
{
    auto p = GetParam();
    Size sz = get<0>(p);
    int srcType = get<0>(get<1>(p));
    int dstType = get<1>(get<1>(p));

    cv::Mat a = Mat(sz, srcType);
    cv::Scalar b;
    cv::Mat c = Mat(sz, dstType);

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(dstType) < CV_32S)
    {
        b = Scalar(1, 0, 3, 4); // don't pass non-integer values for 8U/8S/16U/16S processing
    }
    else if (CV_MAT_DEPTH(dstType) == CV_32S)
    {
        //see ticket 1529: subtract can be without saturation on 32S
        a /= 2;
        b = Scalar(1, 0, -3, 4); // don't pass non-integer values for 32S processing
    }

    TEST_CYCLE() cv::subtract(a, b, c, /* mask */ noArray(), dstType);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(ArithmMixedTest, multiply)
{
    auto p = GetParam();
    Size sz = get<0>(p);
    int srcType = get<0>(get<1>(p));
    int dstType = get<1>(get<1>(p));

    cv::Mat a(sz, srcType), b(sz, srcType), c(sz, dstType);

    declare.in(a, b, WARMUP_RNG).out(c);
    if (CV_MAT_DEPTH(dstType) == CV_32S)
    {
        //According to docs, saturation is not applied when result is 32bit integer
        a /= (2 << 16);
        b /= (2 << 16);
    }

    TEST_CYCLE() cv::multiply(a, b, c, /* scale */ 1.0, dstType);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(ArithmMixedTest, multiplyScale)
{
    auto p = GetParam();
    Size sz = get<0>(p);
    int srcType = get<0>(get<1>(p));
    int dstType = get<1>(get<1>(p));

    cv::Mat a(sz, srcType), b(sz, srcType), c(sz, dstType);
    double scale = 0.5;

    declare.in(a, b, WARMUP_RNG).out(c);

    if (CV_MAT_DEPTH(dstType) == CV_32S)
    {
        //According to docs, saturation is not applied when result is 32bit integer
        a /= (2 << 16);
        b /= (2 << 16);
    }

    TEST_CYCLE() cv::multiply(a, b, c, scale, dstType);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(ArithmMixedTest, divide)
{
    auto p = GetParam();
    Size sz = get<0>(p);
    int srcType = get<0>(get<1>(p));
    int dstType = get<1>(get<1>(p));

    cv::Mat a(sz, srcType), b(sz, srcType), c(sz, dstType);
    double scale = 0.5;

    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::divide(a, b, c, scale, dstType);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(ArithmMixedTest, reciprocal)
{
    auto p = GetParam();
    Size sz = get<0>(p);
    int srcType = get<0>(get<1>(p));
    int dstType = get<1>(get<1>(p));

    cv::Mat b(sz, srcType), c(sz, dstType);
    double scale = 0.5;

    declare.in(b, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::divide(scale, b, c, dstType);

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/*nothing*/ , ArithmMixedTest,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p),
        testing::Values(std::tuple<perf::MatType, perf::MatType>{CV_8U, CV_16U},
                        std::tuple<perf::MatType, perf::MatType>{CV_8S, CV_16S},
                        std::tuple<perf::MatType, perf::MatType>{CV_8U, CV_32F},
                        std::tuple<perf::MatType, perf::MatType>{CV_8S, CV_32F}
            )
    )
);

typedef perf::TestBaseWithParam<std::tuple<cv::Size, perf::MatType>> ArithmScaleTest;

PERF_TEST_P_(ArithmScaleTest, add_scale)
{
    auto p = GetParam();
    Size sz = get<0>(p);
    int src1Type = get<1>(p);

    cv::Mat a(sz, src1Type), b(sz, CV_16UC1), c(sz, CV_32FC1);
    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::add(a, b, c, noArray(), CV_32F);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(ArithmScaleTest, subtract_scale)
{
    auto p = GetParam();
    Size sz = get<0>(p);
    int src1Type = get<1>(p);

    cv::Mat a(sz, src1Type), b(sz, CV_16UC1), c(sz, CV_32FC1);
    declare.in(a, b, WARMUP_RNG).out(c);

    TEST_CYCLE() cv::subtract(a, b, c, noArray(), CV_32F);

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/*nothing*/ , ArithmScaleTest,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p),
        testing::Values(ARITHM_SCALE_TYPES)
    )
);

typedef perf::TestBaseWithParam<std::tuple<cv::Size, int, bool>> SqrtFixture;
PERF_TEST_P_(SqrtFixture, Sqrt) {
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    bool inverse = get<2>(GetParam());

    Mat src(sz, type), dst(sz, type);
    randu(src, FLT_EPSILON, 1000);
    declare.in(src).out(dst);

    TEST_CYCLE() cv::pow(src, inverse ? -0.5 : 0.5, dst);

    SANITY_CHECK_NOTHING();
}
INSTANTIATE_TEST_CASE_P(/*nothing*/ , SqrtFixture,
    testing::Combine(
        testing::Values(TYPICAL_MAT_SIZES),
        testing::Values(ARITHM_FLOAT_TYPES),
        testing::Bool()
    )
);

typedef perf::TestBaseWithParam<std::tuple<cv::Size, int, double>> PowFixture;
PERF_TEST_P_(PowFixture, Pow) {
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    double power = get<2>(GetParam());

    Mat src(sz, type), dst(sz, type);
    randu(src, FLT_EPSILON, 1000);
    declare.in(src).out(dst);

    TEST_CYCLE() cv::pow(src, power, dst);

    SANITY_CHECK_NOTHING();
}
INSTANTIATE_TEST_CASE_P(/*nothing*/ , PowFixture,
    testing::Combine(
        testing::Values(sz1080p),
        testing::Values(ARITHM_FLOAT_TYPES),
        testing::Values(-6.0, -5.0, -4.5, -4.0, -3.0, -2.0, -1.0, -0.7, -0.5, 0.0,
                        0.5, 0.7, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0)
    )
);

///////////// Rotate ////////////////////////

typedef perf::TestBaseWithParam<std::tuple<cv::Size, int, perf::MatType>> RotateTest;

PERF_TEST_P_(RotateTest, rotate)
{
    Size sz        = get<0>(GetParam());
    int rotatecode = get<1>(GetParam());
    int type       = get<2>(GetParam());
    cv::Mat a(sz, type), b(sz, type);

    declare.in(a, WARMUP_RNG).out(b);

    TEST_CYCLE() cv::rotate(a, b, rotatecode);

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/*nothing*/ , RotateTest,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p),
        testing::Values(ROTATE_180, ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE),
        testing::Values(CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4, CV_8SC1, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4, CV_32SC1, CV_32FC1)
    )
);


///////////// PatchNaNs ////////////////////////

template<typename _Tp>
_Tp randomNan(RNG& rng);

template<>
float randomNan(RNG& rng)
{
    uint32_t r = rng.next();
    Cv32suf v;
    v.u = r;
    // exp & set a bit to avoid zero mantissa
    v.u = v.u | 0x7f800001;
    return v.f;
}

template<>
double randomNan(RNG& rng)
{
    uint32_t r0 = rng.next();
    uint32_t r1 = rng.next();
    Cv64suf v;
    v.u = (uint64_t(r0) << 32) | uint64_t(r1);
    // exp &set a bit to avoid zero mantissa
    v.u = v.u | 0x7ff0000000000001;
    return v.f;
}

typedef Size_MatType PatchNaNsFixture;

PERF_TEST_P_(PatchNaNsFixture, PatchNaNs)
{
    const Size_MatType_t params = GetParam();
    Size srcSize = get<0>(params);
    const int type = get<1>(params), cn = CV_MAT_CN(type);

    Mat src(srcSize, type);
    declare.in(src, WARMUP_RNG).out(src);

    // generating NaNs
    {
        srcSize.width *= cn;
        RNG& rng = theRNG();
        for (int y = 0; y < srcSize.height; ++y)
        {
            float  *const ptrf = src.ptr<float>(y);
            for (int x = 0; x < srcSize.width; ++x)
            {
                ptrf[x] = (x + y) % 2 == 0 ? randomNan<float >(rng) : ptrf[x];
            }
        }
    }

    TEST_CYCLE() cv::patchNaNs(src, 17.7);

    SANITY_CHECK(src);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/ , PatchNaNsFixture,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p, sz2160p),
        testing::Values(CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4)
    )
);

//////////////EXP////////////

typedef Size_MatType ExpFixture;

PERF_TEST_P(ExpFixture, Exp,
    testing::Combine(testing::Values(TYPICAL_MAT_SIZES), testing::Values(CV_32F, CV_64F)))
{
    cv::Size size = std::get<0>(GetParam());
    int type = std::get<1>(GetParam());

    cv::Mat src(size, type);
    cv::Mat dst(size, type);

    declare.in(src).out(dst);

    cv::randu(src, -5.0, 5.0);

    TEST_CYCLE()
    {
        cv::exp(src, dst);
    }

    SANITY_CHECK_NOTHING();
}

//////////////LOG////////////

typedef Size_MatType LogFixture;

PERF_TEST_P(LogFixture, Log,
    testing::Combine(testing::Values(TYPICAL_MAT_SIZES), testing::Values(CV_32F, CV_64F)))
{
    cv::Size size = std::get<0>(GetParam());
    int type = std::get<1>(GetParam());

    cv::Mat src(size, type);
    cv::Mat dst(size, type);

    declare.in(src).out(dst);

    cv::randu(src, 1e-5, 1e5);

    TEST_CYCLE()
    {
        cv::log(src, dst);
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
