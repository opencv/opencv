// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include <numeric>
#include <random>

namespace opencv_test { namespace {

// element sizes 1, 2, 3, 4, 5, 6, 8, 16, 24 and 32 bytes
static const int ndTypes[] = { CV_8UC1, CV_16FC1, CV_8UC3, CV_32FC1, CV_8UC(5), CV_16UC3,
                               CV_64FC1, CV_32SC2, CV_32FC4, CV_64FC3, CV_64FC4 };

static int randomType(RNG& rng)
{
    return ndTypes[rng.uniform(0, (int)(sizeof(ndTypes)/sizeof(ndTypes[0])))];
}

static std::vector<int> randomShape(RNG& rng, int dims, int maxSize)
{
    std::vector<int> shape(dims);
    for (int i = 0; i < dims; i++)
        shape[i] = rng.uniform(1, maxSize + 1);
    return shape;
}

// Random bytes; with roi == true the result is a sub-array of a bigger array.
static Mat randomArray(RNG& rng, const std::vector<int>& shape, int type, bool roi)
{
    const int dims = (int)shape.size();
    for (int i = 0; i < dims; i++)
        if (shape[i] == 0)
            roi = false;  // Mat(m, ranges) does not accept empty ranges
    std::vector<int> bigShape = shape;
    std::vector<Range> ranges(dims);
    for (int i = 0; i < dims; i++)
    {
        int before = roi ? rng.uniform(0, 3) : 0, after = roi ? rng.uniform(0, 3) : 0;
        bigShape[i] = shape[i] + before + after;
        ranges[i] = Range(before, before + shape[i]);
    }
    Mat big(bigShape, type);
    if (big.total() > 0)
    {
        Mat bytes(1, (int)(big.total()*big.elemSize()), CV_8U, big.data);
        rng.fill(bytes, RNG::UNIFORM, 0, 256);
    }
    return roi ? big(ranges) : big;
}

static std::vector<int> shapeOf(const Mat& m)
{
    return std::vector<int>(m.size.p, m.size.p + m.dims);
}

template<typename F> static void forEachIndex(const std::vector<int>& shape, F f)
{
    const int dims = (int)shape.size();
    for (int i = 0; i < dims; i++)
        if (shape[i] == 0)
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

static void expectSame(const Mat& expected, const Mat& actual)
{
    ASSERT_EQ(expected.type(), actual.type());
    ASSERT_EQ(shapeOf(expected), shapeOf(actual));
    const size_t esz = expected.elemSize();
    forEachIndex(shapeOf(expected), [&](const int* idx)
    {
        if (memcmp(expected.ptr(idx), actual.ptr(idx), esz) != 0)
        {
            std::vector<int> v(idx, idx + expected.dims);
            ADD_FAILURE() << "mismatch at " << Mat(v).t();
            throw std::runtime_error("mismatch");
        }
    });
}

#define EXPECT_SAME(expected, actual) \
    do { try { expectSame(expected, actual); } catch (const std::runtime_error&) {} } while (0)

static std::string describe(const Mat& m)
{
    return cv::format("%s %s%s", typeToString(m.type()).c_str(),
                      MatShape(shapeOf(m)).str().c_str(), m.isContinuous() ? "" : " (roi)");
}

static Mat refTranspose(const Mat& src, const std::vector<int>& order)
{
    const int dims = src.dims;
    std::vector<int> outShape(dims);
    for (int i = 0; i < dims; i++)
        outShape[i] = src.size[order[i]];
    Mat dst(outShape, src.type());
    std::vector<int> sidx(dims);
    forEachIndex(outShape, [&](const int* idx)
    {
        for (int i = 0; i < dims; i++)
            sidx[order[i]] = idx[i];
        memcpy(dst.ptr(idx), src.ptr(sidx.data()), src.elemSize());
    });
    return dst;
}

static Mat refFlip(const Mat& src, int axis)
{
    Mat dst(shapeOf(src), src.type());
    std::vector<int> sidx(src.dims);
    forEachIndex(shapeOf(src), [&](const int* idx)
    {
        sidx.assign(idx, idx + src.dims);
        sidx[axis] = src.size[axis] - 1 - idx[axis];
        memcpy(dst.ptr(idx), src.ptr(sidx.data()), src.elemSize());
    });
    return dst;
}

static Mat refConcat(const std::vector<Mat>& src, int axis)
{
    std::vector<int> outShape = shapeOf(src[0]);
    outShape[axis] = 0;
    for (const Mat& m : src)
        outShape[axis] += m.size[axis];
    Mat dst(outShape, src[0].type());
    int ofs = 0;
    for (const Mat& m : src)
    {
        std::vector<int> didx(m.dims);
        forEachIndex(shapeOf(m), [&](const int* idx)
        {
            didx.assign(idx, idx + m.dims);
            didx[axis] += ofs;
            memcpy(dst.ptr(didx.data()), m.ptr(idx), m.elemSize());
        });
        ofs += m.size[axis];
    }
    return dst;
}

static Mat refTile(const Mat& src, const std::vector<int>& repeats)
{
    std::vector<int> outShape(src.dims);
    for (int i = 0; i < src.dims; i++)
        outShape[i] = src.size[i]*repeats[i];
    Mat dst(outShape, src.type());
    std::vector<int> sidx(src.dims);
    forEachIndex(outShape, [&](const int* idx)
    {
        for (int i = 0; i < src.dims; i++)
            sidx[i] = idx[i] % src.size[i];
        memcpy(dst.ptr(idx), src.ptr(sidx.data()), src.elemSize());
    });
    return dst;
}

static Mat refSlice(const Mat& src, const std::vector<int>& starts,
                    const std::vector<int>& ends, const std::vector<int>& steps)
{
    std::vector<int> outShape = shapeOf(src);
    for (size_t i = 0; i < starts.size(); i++)
    {
        int s = starts[i], e = ends[i], st = steps.empty() ? 1 : steps[i];
        int count = 0;
        if (st > 0)
            for (int k = s; k < e; k += st) count++;
        else
            for (int k = s; k > e; k += st) count++;
        outShape[i] = count;
    }
    Mat dst(outShape, src.type());
    std::vector<int> sidx(src.dims);
    forEachIndex(outShape, [&](const int* idx)
    {
        for (int i = 0; i < src.dims; i++)
        {
            bool sliced = i < (int)starts.size();
            int st = sliced && !steps.empty() ? steps[i] : 1;
            sidx[i] = (sliced ? starts[i] : 0) + idx[i]*st;
        }
        memcpy(dst.ptr(idx), src.ptr(sidx.data()), src.elemSize());
    });
    return dst;
}

TEST(Core_TransposeND, random)
{
    RNG& rng = theRNG();
    for (int iter = 0; iter < 300; iter++)
    {
        int dims = rng.uniform(1, 7);
        Mat src = randomArray(rng, randomShape(rng, dims, dims <= 3 ? 12 : 5), randomType(rng), rng.uniform(0, 2) != 0);
        std::vector<int> order(dims);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), std::mt19937(rng.next()));

        Mat dst;
        cv::transposeND(src, order, dst);
        SCOPED_TRACE(describe(src) + " order " + MatShape(order).str());
        EXPECT_SAME(refTranspose(src, order), dst);
    }
}

typedef testing::TestWithParam<tuple<std::vector<int>, std::vector<int>, perf::MatType> > Core_TransposeND_Large;

// shapes big enough to use the parallel and blocked-transpose paths
TEST_P(Core_TransposeND_Large, accuracy)
{
    const std::vector<int> shape = get<0>(GetParam());
    const std::vector<int> order = get<1>(GetParam());
    const int type = get<2>(GetParam());
    RNG& rng = theRNG();
    for (int roi = 0; roi < 2; roi++)
    {
        Mat src = randomArray(rng, shape, type, roi != 0);
        Mat dst;
        cv::transposeND(src, order, dst);
        SCOPED_TRACE(describe(src));
        EXPECT_SAME(refTranspose(src, order), dst);
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Core_TransposeND_Large, testing::Values(
    make_tuple(std::vector<int>{1, 32, 70, 90}, std::vector<int>{0, 2, 3, 1}, perf::MatType(CV_32FC1)),
    make_tuple(std::vector<int>{1, 150, 170, 3}, std::vector<int>{0, 3, 1, 2}, perf::MatType(CV_32FC1)),
    make_tuple(std::vector<int>{2, 3, 150, 170}, std::vector<int>{0, 2, 3, 1}, perf::MatType(CV_32FC1)),
    make_tuple(std::vector<int>{1, 4, 100, 130}, std::vector<int>{0, 2, 3, 1}, perf::MatType(CV_8UC1)),
    make_tuple(std::vector<int>{1, 100, 130, 2}, std::vector<int>{0, 3, 1, 2}, perf::MatType(CV_16FC1)),
    make_tuple(std::vector<int>{3, 50, 60, 4}, std::vector<int>{3, 1, 2, 0}, perf::MatType(CV_64FC1)),
    make_tuple(std::vector<int>{4, 130, 200}, std::vector<int>{0, 2, 1}, perf::MatType(CV_8UC1)),
    make_tuple(std::vector<int>{4, 130, 200}, std::vector<int>{0, 2, 1}, perf::MatType(CV_16FC1)),
    make_tuple(std::vector<int>{2, 60, 8, 64}, std::vector<int>{0, 2, 1, 3}, perf::MatType(CV_32FC1)),
    make_tuple(std::vector<int>{300, 200}, std::vector<int>{1, 0}, perf::MatType(CV_8UC(5))),
    make_tuple(std::vector<int>{3, 40, 50, 60}, std::vector<int>{3, 1, 0, 2}, perf::MatType(CV_8UC3)),
    make_tuple(std::vector<int>{2, 16, 8, 16, 32}, std::vector<int>{0, 4, 2, 1, 3}, perf::MatType(CV_64FC1))
));

// a user-provided row step that is not a multiple of the element size
TEST(Core_TransposeND, odd_step)
{
    RNG& rng = theRNG();
    const int sizes[] = {30, 40, 3};
    const size_t esz = 4, steps[] = {40*3*esz + 2, 3*esz};  // CV_16UC2: esz == 4
    std::vector<uchar> buf(sizes[0]*steps[0] + 16);
    Mat bytes(1, (int)buf.size(), CV_8U, buf.data());
    rng.fill(bytes, RNG::UNIFORM, 0, 256);
    Mat src3(3, sizes, CV_16UC2, buf.data() + 2, steps);
    Mat dst;
    cv::transposeND(src3, {2, 0, 1}, dst);
    EXPECT_SAME(refTranspose(src3, {2, 0, 1}), dst);
}

TEST(Core_TransposeND, inplace)
{
    RNG& rng = theRNG();
    Mat a = randomArray(rng, {6, 6, 6}, CV_32FC3, false);
    Mat expected = refTranspose(a, {2, 0, 1});
    cv::transposeND(a, {2, 0, 1}, a);
    EXPECT_SAME(expected, a);
}

// dnn preallocates the outputs with Mat::fit(), which keeps the buffer for an empty shape;
// transposeND must not reallocate or release it
TEST(Core_TransposeND, empty_keeps_buffer)
{
    Mat src({0, 5}, CV_32F);
    Mat buf(1, 16, CV_32F), dst = buf;
    dst.fit(std::vector<int>{5, 0}, CV_32F);
    cv::transposeND(src, {1, 0}, dst);
    EXPECT_EQ(buf.u, dst.u);
    EXPECT_EQ(std::vector<int>({5, 0}), shapeOf(dst));
}

TEST(Core_TransposeND, invalid_order)
{
    Mat a({2, 3, 4}, CV_32F, Scalar(0)), b;
    EXPECT_ANY_THROW(cv::transposeND(a, {0, 1}, b));
    EXPECT_ANY_THROW(cv::transposeND(a, {0, 1, 1}, b));
    EXPECT_ANY_THROW(cv::transposeND(a, {0, 1, 3}, b));
}

TEST(Core_FlipND, random)
{
    RNG& rng = theRNG();
    for (int iter = 0; iter < 200; iter++)
    {
        int dims = rng.uniform(1, 6);
        Mat src = randomArray(rng, randomShape(rng, dims, dims <= 3 ? 40 : 6), randomType(rng), rng.uniform(0, 2) != 0);
        int axis = rng.uniform(-dims, dims);
        Mat dst;
        cv::flipND(src, dst, axis);
        SCOPED_TRACE(describe(src) + cv::format(" axis %d", axis));
        EXPECT_SAME(refFlip(src, (axis + dims) % dims), dst);
    }
}

TEST(Core_FlipND, inplace)
{
    RNG& rng = theRNG();
    for (int axis = 0; axis < 3; axis++)
    {
        Mat a = randomArray(rng, {5, 7, 9}, CV_8UC3, false);
        Mat expected = refFlip(a, axis);
        cv::flipND(a, a, axis);
        EXPECT_SAME(expected, a);
    }
}

TEST(Core_FlipND, inplace_roi)
{
    RNG& rng = theRNG();
    for (int axis = 0; axis < 3; axis++)
    {
        Mat a = randomArray(rng, {5, 7, 9}, CV_32FC2, true);
        Mat expected = refFlip(a, axis);
        cv::flipND(a, a, axis);
        EXPECT_SAME(expected, a);
    }
}

TEST(Core_ConcatND, random)
{
    RNG& rng = theRNG();
    for (int iter = 0; iter < 200; iter++)
    {
        int dims = rng.uniform(1, 6), type = randomType(rng);
        int axis = rng.uniform(0, dims);
        std::vector<int> shape = randomShape(rng, dims, dims <= 3 ? 20 : 5);
        int n = rng.uniform(1, 5);
        std::vector<Mat> src(n);
        for (int k = 0; k < n; k++)
        {
            shape[axis] = rng.uniform(0, 6);
            src[k] = randomArray(rng, shape, type, rng.uniform(0, 2) != 0);
        }
        Mat dst;
        cv::concatND(src, axis - (rng.uniform(0, 2) ? dims : 0), dst);
        SCOPED_TRACE(describe(src[0]) + cv::format(" x%d axis %d", n, axis));
        EXPECT_SAME(refConcat(src, axis), dst);
    }
}

TEST(Core_ConcatND, matches_hconcat_vconcat)
{
    RNG& rng = theRNG();
    std::vector<Mat> v = { randomArray(rng, {20, 30}, CV_8UC3, true), randomArray(rng, {20, 30}, CV_8UC3, false) };
    Mat h1, h2, v1, v2;
    cv::hconcat(v, h1);
    cv::vconcat(v, v1);
    cv::concatND(v, 1, h2);
    cv::concatND(v, 0, v2);
    EXPECT_SAME(h1, h2);
    EXPECT_SAME(v1, v2);
}

TEST(Core_ConcatND, large)
{
    RNG& rng = theRNG();
    std::vector<Mat> src = { randomArray(rng, {1, 64, 40, 40}, CV_32F, false),
                             randomArray(rng, {1, 32, 40, 40}, CV_32F, true),
                             randomArray(rng, {1, 16, 40, 40}, CV_32F, false) };
    Mat dst;
    cv::concatND(src, 1, dst);
    EXPECT_SAME(refConcat(src, 1), dst);
}

TEST(Core_ConcatND, invalid)
{
    Mat a({2, 3, 4}, CV_32F, Scalar(0)), b({2, 5, 4}, CV_32F, Scalar(0)), c({2, 3, 4}, CV_8U, Scalar(0)), d;
    EXPECT_ANY_THROW(cv::concatND(std::vector<Mat>{a, b}, 2, d));
    EXPECT_ANY_THROW(cv::concatND(std::vector<Mat>{a, c}, 0, d));
    EXPECT_ANY_THROW(cv::concatND(std::vector<Mat>{a, a}, 3, d));
    EXPECT_NO_THROW(cv::concatND(std::vector<Mat>{a, b}, 1, d));
}

// an input that aliases a part of dst written by another input must be read first
TEST(Core_ConcatND, input_aliases_dst)
{
    RNG& rng = theRNG();
    Mat dst = randomArray(rng, {6, 4}, CV_32F, false);
    Mat other = randomArray(rng, {3, 4}, CV_32F, false);
    std::vector<Mat> src = { other, dst.rowRange(0, 3) };
    Mat expected;
    cv::vconcat(std::vector<Mat>{ other.clone(), src[1].clone() }, expected);
    cv::concatND(src, 0, dst);
    EXPECT_SAME(expected, dst);
}

TEST(Core_SplitND, random)
{
    RNG& rng = theRNG();
    for (int iter = 0; iter < 200; iter++)
    {
        int dims = rng.uniform(1, 6);
        Mat src = randomArray(rng, randomShape(rng, dims, dims <= 3 ? 20 : 5), randomType(rng), rng.uniform(0, 2) != 0);
        int axis = rng.uniform(0, dims);
        std::vector<int> sizes;
        for (int left = src.size[axis]; left > 0; )
        {
            int s = std::min(left, rng.uniform(0, 5));
            sizes.push_back(s);
            left -= s;
        }
        if (sizes.empty())
            sizes.push_back(0);
        std::vector<Mat> dst;
        cv::splitND(src, axis, sizes, dst);
        SCOPED_TRACE(describe(src) + cv::format(" axis %d", axis));
        ASSERT_EQ(sizes.size(), dst.size());
        int ofs = 0;
        for (size_t k = 0; k < sizes.size(); k++)
        {
            std::vector<int> starts(axis + 1, 0), ends = shapeOf(src);
            ends.resize(axis + 1);
            starts[axis] = ofs;
            ends[axis] = ofs + sizes[k];
            EXPECT_SAME(refSlice(src, starts, ends, {}), dst[k]);
            ofs += sizes[k];
        }
        Mat back;
        cv::concatND(dst, axis, back);
        EXPECT_SAME(src, back);
    }
}

TEST(Core_SplitND, invalid)
{
    Mat a({2, 6, 4}, CV_32F, Scalar(0));
    std::vector<Mat> d;
    EXPECT_ANY_THROW(cv::splitND(a, 1, {2, 3}, d));
    EXPECT_ANY_THROW(cv::splitND(a, 1, {7, -1}, d));
    EXPECT_ANY_THROW(cv::splitND(a, 3, {2}, d));
    EXPECT_NO_THROW(cv::splitND(a, -2, {2, 4}, d));
}

TEST(Core_TileND, random)
{
    RNG& rng = theRNG();
    for (int iter = 0; iter < 200; iter++)
    {
        int dims = rng.uniform(1, 5);
        Mat src = randomArray(rng, randomShape(rng, dims, dims <= 2 ? 16 : 5), randomType(rng), rng.uniform(0, 2) != 0);
        std::vector<int> repeats(dims);
        for (int i = 0; i < dims; i++)
            repeats[i] = rng.uniform(0, 10) == 0 ? 0 : rng.uniform(1, 4);
        Mat dst;
        cv::tileND(src, repeats, dst);
        SCOPED_TRACE(describe(src) + " repeats " + MatShape(repeats).str());
        EXPECT_SAME(refTile(src, repeats), dst);
    }
}

TEST(Core_TileND, matches_repeat)
{
    RNG& rng = theRNG();
    Mat src = randomArray(rng, {7, 11}, CV_16UC3, true), a, b;
    cv::repeat(src, 3, 5, a);
    cv::tileND(src, {3, 5}, b);
    EXPECT_SAME(a, b);
}

TEST(Core_TileND, large)
{
    RNG& rng = theRNG();
    Mat src = randomArray(rng, {1, 64, 1, 128}, CV_32F, false), dst;
    cv::tileND(src, {1, 1, 64, 1}, dst);
    EXPECT_SAME(refTile(src, {1, 1, 64, 1}), dst);
}

TEST(Core_SliceND, random)
{
    RNG& rng = theRNG();
    for (int iter = 0; iter < 300; iter++)
    {
        int dims = rng.uniform(1, 6);
        Mat src = randomArray(rng, randomShape(rng, dims, dims <= 3 ? 20 : 6), randomType(rng), rng.uniform(0, 2) != 0);
        int nslices = rng.uniform(0, dims + 1);
        std::vector<int> starts(nslices), ends(nslices), steps;
        bool withSteps = rng.uniform(0, 3) != 0;
        for (int i = 0; i < nslices; i++)
        {
            int n = src.size[i];
            int step = withSteps ? rng.uniform(1, 4)*(rng.uniform(0, 2) ? 1 : -1) : 1;
            if (withSteps)
                steps.push_back(step);
            if (step > 0)
            {
                starts[i] = rng.uniform(0, n + 1);
                ends[i] = rng.uniform(0, n + 1);
            }
            else
            {
                starts[i] = rng.uniform(-1, n);
                ends[i] = rng.uniform(-1, n);
            }
        }
        Mat dst;
        cv::sliceND(src, starts, ends, steps, dst);
        SCOPED_TRACE(describe(src) + " starts " + MatShape(starts).str() + " ends " + MatShape(ends).str() +
                     " steps " + MatShape(steps).str());
        EXPECT_SAME(refSlice(src, starts, ends, steps), dst);
    }
}

TEST(Core_SliceND, reverse_whole_axis)
{
    RNG& rng = theRNG();
    Mat src = randomArray(rng, {4, 9, 5}, CV_32FC2, true), dst;
    cv::sliceND(src, {0, 8}, {4, -1}, {1, -1}, dst);
    EXPECT_SAME(refFlip(src, 1), dst);
}

TEST(Core_NDTransform, scalar)
{
    Mat a(0, nullptr, CV_32F);
    a.at<float>(0) = 3.5f;
    ASSERT_EQ(0, a.dims);
    Mat t, r, s;
    cv::transposeND(a, std::vector<int>(), t);
    cv::tileND(a, std::vector<int>(), r);
    cv::sliceND(a, std::vector<int>(), std::vector<int>(), std::vector<int>(), s);
    for (const Mat& m : {t, r, s})
    {
        EXPECT_EQ(0, m.dims);
        EXPECT_EQ((size_t)1, m.total());
        EXPECT_EQ(3.5f, m.at<float>(0));
    }
    EXPECT_ANY_THROW(cv::flipND(a, t, 0));
    EXPECT_ANY_THROW(cv::concatND(std::vector<Mat>{a, a}, 0, t));
}

TEST(Core_SliceND, invalid)
{
    Mat a({4, 5}, CV_32F, Scalar(0)), d;
    EXPECT_ANY_THROW(cv::sliceND(a, {0}, {4}, {0}, d));
    EXPECT_ANY_THROW(cv::sliceND(a, {0}, {5}, {}, d));
    EXPECT_ANY_THROW(cv::sliceND(a, {-1}, {4}, {1}, d));
    EXPECT_ANY_THROW(cv::sliceND(a, {0, 0, 0}, {1, 1, 1}, {}, d));
    EXPECT_NO_THROW(cv::sliceND(a, {3}, {-1}, {-1}, d));
}

}} // namespace
