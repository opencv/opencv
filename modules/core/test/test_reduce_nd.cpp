// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include <cmath>
#include <random>

namespace opencv_test { namespace {

static const int reduceOps[] = { REDUCE_SUM, REDUCE_AVG, REDUCE_MAX, REDUCE_MIN, REDUCE_SUM2, REDUCE_PROD,
                                 REDUCE_L1, REDUCE_L2, REDUCE_LOG_SUM, REDUCE_LOG_SUM_EXP };
static const int reduceDepths[] = { CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F, CV_16F, CV_64S };

static std::vector<int> shapeOf(const Mat& m) { return std::vector<int>(m.size.p, m.size.p + m.dims); }

static double readAsDouble(const Mat& m, const int* idx, int c)
{
    const uchar* p = m.ptr(idx) + c*m.elemSize1();
    switch (m.depth())
    {
    case CV_8U: return *p;
    case CV_8S: return *(const schar*)p;
    case CV_16U: return *(const ushort*)p;
    case CV_16S: return *(const short*)p;
    case CV_32S: return *(const int*)p;
    case CV_32F: return *(const float*)p;
    case CV_64F: return *(const double*)p;
    case CV_16F: return (float)*(const hfloat*)p;
    case CV_64S: return (double)*(const int64_t*)p;
    }
    CV_Error(Error::StsBadArg, "unsupported depth");
}

template<typename F> static void forEachIndex(const std::vector<int>& shape, F f)
{
    const int dims = (int)shape.size();
    for (int s : shape)
        if (s == 0)
            return;
    std::vector<int> idx(dims, 0);
    for (;;)
    {
        f(idx.data());
        int k = dims - 1;
        for (; k >= 0; k--)
        {
            if (++idx[k] < shape[k])
                break;
            idx[k] = 0;
        }
        if (k < 0)
            break;
    }
}

// Naive reference in double: a CV_64F array in the keepdims layout, src.channels() channels.
static Mat refReduce(const Mat& src, const std::vector<bool>& reduced, int op)
{
    const int dims = src.dims, cn = src.channels();
    std::vector<int> outShape = shapeOf(src), redShape = shapeOf(src);
    for (int i = 0; i < dims; i++)
    {
        if (reduced[i]) outShape[i] = 1;
        else redShape[i] = 1;
    }
    Mat dst(outShape, CV_64FC(cn));
    std::vector<int> sidx(dims);
    forEachIndex(outShape, [&](const int* oidx)
    {
        for (int c = 0; c < cn; c++)
        {
            double acc = op == REDUCE_PROD ? 1 : op == REDUCE_MAX ? -INFINITY : op == REDUCE_MIN ? INFINITY : 0;
            double mx = -INFINITY;
            double count = 0;
            if (op == REDUCE_LOG_SUM_EXP)
                forEachIndex(redShape, [&](const int* ridx)
                {
                    for (int i = 0; i < dims; i++) sidx[i] = reduced[i] ? ridx[i] : oidx[i];
                    mx = std::max(mx, readAsDouble(src, sidx.data(), c));
                });
            forEachIndex(redShape, [&](const int* ridx)
            {
                for (int i = 0; i < dims; i++) sidx[i] = reduced[i] ? ridx[i] : oidx[i];
                double x = readAsDouble(src, sidx.data(), c);
                count++;
                switch (op)
                {
                case REDUCE_SUM: case REDUCE_AVG: case REDUCE_LOG_SUM: acc += x; break;
                case REDUCE_SUM2: case REDUCE_L2: acc += x*x; break;
                case REDUCE_L1: acc += std::abs(x); break;
                case REDUCE_PROD: acc *= x; break;
                case REDUCE_MAX: acc = std::max(acc, x); break;
                case REDUCE_MIN: acc = std::min(acc, x); break;
                case REDUCE_LOG_SUM_EXP: acc += std::exp(x - mx); break;
                }
            });
            if (op == REDUCE_AVG) acc = count > 0 ? acc/count : 0;
            else if (op == REDUCE_L2) acc = std::sqrt(acc);
            else if (op == REDUCE_LOG_SUM) acc = std::log(acc);
            else if (op == REDUCE_LOG_SUM_EXP) acc = mx + std::log(acc);
            ((double*)dst.ptr(oidx))[c] = acc;
        }
    });
    return dst;
}

// values that keep every operation well-defined and finite
static void fillForOp(RNG& rng, Mat& m, int op)
{
    const int depth = m.depth();
    const bool isFloat = depth == CV_32F || depth == CV_64F || depth == CV_16F;
    const bool isUnsigned = depth == CV_8U || depth == CV_16U;
    double lo, hi;
    if (op == REDUCE_PROD) { lo = isFloat ? 0.9 : 1; hi = isFloat ? 1.1 : 2; }
    else if (op == REDUCE_LOG_SUM) { lo = isFloat ? 0.1 : 1; hi = isFloat ? 10 : 50; }
    else if (op == REDUCE_LOG_SUM_EXP) { lo = -5; hi = 5; }
    else if (isFloat) { lo = -10; hi = 10; }
    else { lo = isUnsigned ? 0 : -100; hi = 100; }
    Mat tmp(shapeOf(m), CV_64FC(m.channels()));
    Mat flat(1, (int)(tmp.total()*tmp.channels()), CV_64F, tmp.data);
    rng.fill(flat, RNG::UNIFORM, lo, hi);
    if (!isFloat)
        for (int i = 0; i < flat.cols; i++)
        {
            double& v = flat.at<double>(i);
            v = std::floor(v);
            if (op == REDUCE_PROD)
                v = std::min(std::max(v, 1.), 2.);
        }
    tmp.convertTo(m, depth);
}

static void checkReduce(const Mat& src, const Mat& dst, const std::vector<bool>& reduced, int op,
                        bool keepdims, int ddepth, const std::string& info)
{
    Mat ref = refReduce(src, reduced, op);
    std::vector<int> keepShape = shapeOf(ref), outShape;
    for (int i = 0; i < src.dims; i++)
        if (!reduced[i] || keepdims)
            outShape.push_back(keepShape[i]);
    ASSERT_EQ(outShape, shapeOf(dst)) << info;
    ASSERT_EQ(CV_MAKETYPE(ddepth, src.channels()), dst.type()) << info;

    Mat refD;
    ref.convertTo(refD, ddepth);         // the expected result, saturated like the output
    auto flat64 = [](const Mat& m) {
        Mat c = m.isContinuous() ? m : m.clone(), r;
        Mat(1, (int)(c.total()*c.channels()), c.depth(), c.data).convertTo(r, CV_64F);
        return r;
    };
    Mat a = flat64(refD), b = flat64(dst), refRaw = flat64(ref);

    const bool exact = op == REDUCE_MAX || op == REDUCE_MIN;
    const bool intOut = ddepth != CV_32F && ddepth != CV_64F && ddepth != CV_16F;
    const double relTol = ddepth == CV_16F ? 2e-3 : ddepth == CV_64F ? 1e-9 : 2e-5;
    for (int i = 0; i < a.cols; i++)
    {
        double e = a.at<double>(i), v = b.at<double>(i), raw = refRaw.at<double>(i);
        if (std::isnan(e) && std::isnan(v)) continue;
        if (std::isinf(e) || std::isinf(v)) { ASSERT_EQ(e, v) << info << " at " << i; continue; }
        double tol = exact ? 0 : intOut ? 1 : relTol*(std::abs(raw) + 1);
        ASSERT_LE(std::abs(e - v), tol) << info << " at " << i << ": expected " << e << ", got " << v;
    }
}

static Mat randomArray(RNG& rng, const std::vector<int>& shape, int type, bool roi)
{
    if (!roi)
        return Mat(shape, type);
    std::vector<int> big = shape;
    std::vector<Range> ranges;
    for (size_t i = 0; i < shape.size(); i++)
    {
        int before = rng.uniform(0, 2);
        big[i] += before + rng.uniform(0, 2);
        ranges.push_back(Range(before, before + shape[i]));
    }
    return Mat(big, type)(ranges);
}

TEST(Core_ReduceND, random)
{
    RNG& rng = theRNG();
    int checked = 0;
    for (int iter = 0; iter < 400; iter++)
    {
        const int dims = rng.uniform(1, 6);
        std::vector<int> shape(dims);
        for (int i = 0; i < dims; i++)
            shape[i] = rng.uniform(1, dims <= 2 ? 30 : 7);
        const int op = reduceOps[rng.uniform(0, 10)];
        const int depth = reduceDepths[rng.uniform(0, 9)];
        const int cn = rng.uniform(0, 4) == 0 ? rng.uniform(2, 4) : 1;
        Mat src = randomArray(rng, shape, CV_MAKETYPE(depth, cn), rng.uniform(0, 2) != 0);
        fillForOp(rng, src, op);

        std::vector<bool> reduced(dims, false);
        std::vector<int> axes;
        const bool all = rng.uniform(0, 5) == 0;
        for (int i = 0; i < dims; i++)
            if (all || rng.uniform(0, 2))
            {
                reduced[i] = true;
                axes.push_back(rng.uniform(0, 2) ? i : i - dims);
            }
        if (all)
            axes.clear();
        else if (axes.empty())
        {
            reduced[0] = true;
            axes.push_back(0);
        }
        std::shuffle(axes.begin(), axes.end(), std::mt19937(rng.next()));
        if (op == REDUCE_PROD)
        {
            // keep integer products in range
            int64 count = 1;
            for (int i = 0; i < dims; i++) if (reduced[i]) count *= shape[i];
            if (count > 20 && depth != CV_32F && depth != CV_64F && depth != CV_16F)
                continue;
        }
        const bool keepdims = rng.uniform(0, 2) != 0;
        const int ddepth = rng.uniform(0, 3) == 0 ? (rng.uniform(0, 2) ? CV_32F : CV_64F) : depth;

        Mat dst;
        cv::reduce(src, dst, ReduceParams(op, axes, keepdims, ddepth == depth ? -1 : ddepth));
        std::string info = cv::format("iter %d: %s %s cn=%d op=%d axes=%s keepdims=%d ddepth=%d%s", iter,
                                      typeToString(depth).c_str(), MatShape(shape).str().c_str(), cn, op,
                                      MatShape(axes).str().c_str(), (int)keepdims, ddepth,
                                      src.isContinuous() ? "" : " roi");
        checkReduce(src, dst, reduced, op, keepdims, ddepth, info);
        if (HasFatalFailure())
            return;
        checked++;
    }
    EXPECT_GE(checked, 300);
}

typedef testing::TestWithParam<tuple<std::vector<int>, std::vector<int>, int> > Core_ReduceND_Large;

// large enough for the parallel paths, including split reductions
TEST_P(Core_ReduceND_Large, accuracy)
{
    const std::vector<int> shape = get<0>(GetParam()), axes = get<1>(GetParam());
    const int op = get<2>(GetParam());
    RNG& rng = theRNG();
    for (int depth : {CV_32F, CV_8U})
    {
        Mat src(shape, depth);
        fillForOp(rng, src, op);
        std::vector<bool> reduced(shape.size(), axes.empty());
        for (int a : axes) reduced[a] = true;
        Mat dst;
        cv::reduce(src, dst, ReduceParams(op, axes, true, CV_64F));
        checkReduce(src, dst, reduced, op, true, CV_64F, typeToString(depth));
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Core_ReduceND_Large, testing::Values(
    make_tuple(std::vector<int>{1, 64, 80, 80}, std::vector<int>{1}, (int)REDUCE_AVG),
    make_tuple(std::vector<int>{1, 64, 80, 80}, std::vector<int>{2, 3}, (int)REDUCE_SUM),
    make_tuple(std::vector<int>{600, 700}, std::vector<int>{}, (int)REDUCE_SUM),
    make_tuple(std::vector<int>{600, 700}, std::vector<int>{}, (int)REDUCE_MAX),
    make_tuple(std::vector<int>{100000, 3}, std::vector<int>{0}, (int)REDUCE_SUM2),
    make_tuple(std::vector<int>{3, 100000}, std::vector<int>{1}, (int)REDUCE_L2),
    make_tuple(std::vector<int>{8, 50, 300}, std::vector<int>{0, 2}, (int)REDUCE_LOG_SUM_EXP),
    make_tuple(std::vector<int>{8, 50, 300}, std::vector<int>{1}, (int)REDUCE_MIN)
));

TEST(Core_ReduceND, reduce_matches_2d)
{
    RNG& rng = theRNG();
    Mat src(37, 53, CV_32FC2);
    rng.fill(src, RNG::UNIFORM, -1, 1);
    for (int op : {REDUCE_SUM, REDUCE_AVG, REDUCE_MAX, REDUCE_MIN, REDUCE_SUM2})
        for (int dim = 0; dim < 2; dim++)
        {
            // 2D cv::reduce keeps the input depth for max/min
            const int ddepth = op == REDUCE_MAX || op == REDUCE_MIN ? -1 : CV_64F;
            Mat a, b;
            cv::reduce(src, a, dim, op, ddepth);
            cv::reduce(src, b, ReduceParams(op, {dim}, true, ddepth));
            EXPECT_LE(cvtest::norm(a, b, NORM_INF), 1e-9) << op << " " << dim;
        }
}

TEST(Core_ReduceND, reduce_nd_input)
{
    RNG& rng = theRNG();
    Mat src({4, 5, 6}, CV_32F), a, b;
    rng.fill(src, RNG::UNIFORM, -1, 1);
    cv::reduce(src, a, 1, REDUCE_MAX);
    cv::reduce(src, b, {REDUCE_MAX, {1}});
    ASSERT_EQ(std::vector<int>({4, 1, 6}), shapeOf(a));
    EXPECT_EQ(0, cvtest::norm(a, b, NORM_INF));
    // the new operations also work through cv::reduce on 2D input
    Mat c, d, m2(3, 4, CV_32F, Scalar(2));
    cv::reduce(m2, c, 1, REDUCE_PROD);
    EXPECT_EQ(16.f, c.at<float>(0));
}

TEST(Core_ReduceND, special_cases)
{
    Mat a({3, 0, 4}, CV_32F), d;
    cv::reduce(a, d, {REDUCE_SUM, {1}});                  // empty reduction -> identity
    ASSERT_EQ(std::vector<int>({3, 1, 4}), shapeOf(d));
    EXPECT_EQ(0, cvtest::norm(d, NORM_INF));
    cv::reduce(a, d, {REDUCE_PROD, {1}});
    EXPECT_EQ(1., cvtest::norm(d, NORM_INF));

    Mat s(0, nullptr, CV_32F);                            // 0-dimensional input
    s.at<float>(0) = -3.f;
    cv::reduce(s, d, {REDUCE_L1});
    EXPECT_EQ(0, d.dims);
    EXPECT_EQ(3.f, d.at<float>(0));

    Mat m({2, 3}, CV_32F, Scalar(1));
    cv::reduce(m, d, {REDUCE_SUM, {}, false});           // all axes, no keepdims -> scalar
    EXPECT_EQ(0, d.dims);
    EXPECT_EQ(6.f, d.at<float>(0));

    Mat f({2, 3}, CV_32F);
    f.setTo(-INFINITY);
    cv::reduce(f, d, {REDUCE_LOG_SUM_EXP, {1}});
    EXPECT_TRUE(std::isinf(d.at<float>(0)) && d.at<float>(0) < 0);
    f.setTo(1000.f);                                      // would overflow exp() without the shift
    cv::reduce(f, d, {REDUCE_LOG_SUM_EXP, {1}});
    EXPECT_NEAR(1000. + std::log(3.), d.at<float>(0), 1e-3);

    EXPECT_ANY_THROW(cv::reduce(m, d, {REDUCE_SUM, {2}}));
    EXPECT_ANY_THROW(cv::reduce(m, d, {REDUCE_SUM, {0, -2}}));
    EXPECT_ANY_THROW(cv::reduce(m, d, {42, {0}}));
}

}} // namespace
