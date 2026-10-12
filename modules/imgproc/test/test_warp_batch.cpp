// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the
// top-level directory of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

enum { OP_AFFINE = 0, OP_PERSPECTIVE = 1 };

static Mat makeTestImage(int type, Size sz, int seed)
{
    Mat img(sz, type);
    RNG rng(seed);
    rng.fill(img, RNG::UNIFORM, 0, 255);
    return img;
}

// Mild rotation, scale and shift, plus a small projective term for perspective.
static Mat makeTransform(int op, Size sz, int seed)
{
    RNG rng(seed);
    const Point2f center(sz.width*0.5f, sz.height*0.5f);
    Mat M = getRotationMatrix2D(center, rng.uniform(-30., 30.), rng.uniform(0.7, 1.3));
    M.at<double>(0, 2) += rng.uniform(-3., 3.);
    M.at<double>(1, 2) += rng.uniform(-3., 3.);
    if (op == OP_AFFINE)
        return M;
    Mat P = Mat::eye(3, 3, CV_64F);
    M.copyTo(P.rowRange(0, 2));
    P.at<double>(2, 0) = rng.uniform(-1e-3, 1e-3);
    P.at<double>(2, 1) = rng.uniform(-1e-3, 1e-3);
    return P;
}

static void warpOne(int op, InputArray src, OutputArray dst, InputArray M, Size dsize,
                    int flags, int border, const Scalar& borderValue = Scalar())
{
    if (op == OP_AFFINE)
        warpAffine(src, dst, M, dsize, flags, border, borderValue);
    else
        warpPerspective(src, dst, M, dsize, flags, border, borderValue);
}

static Mat plane(const Mat& t, int p)
{
    return Mat(t.size[1], t.size[2], t.type(), const_cast<uchar*>(t.ptr(p)), t.step[1]);
}

// The batch runs every image through the kernel its own call would pick, so the match is exact.
TEST(Warp_Batch, TensorMatchesSingleImage)
{
    const int types[] = { CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4, CV_16UC1, CV_16SC3, CV_32FC1, CV_32FC4, CV_64FC1 };
    const int interpolations[] = { INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4 };
    const int borders[] = { BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT_101 };
    const Size from(37, 29), to(41, 23);
    const int N = 4;

    for (int op : { OP_AFFINE, OP_PERSPECTIVE })
        for (int type : types)
            for (int interp : interpolations)
                for (int border : borders)
                    for (bool perImage : { false, true })
                    {
                        int srcSizes[] = { N, from.height, from.width };
                        Mat src(3, srcSizes, type);
                        RNG(0xbeef + interp).fill(src, RNG::UNIFORM, 0, 255);

                        std::vector<Mat> Ms;
                        for (int i = 0; i < (perImage ? N : 1); i++)
                            Ms.push_back(makeTransform(op, from, 100 + i));

                        Mat dst;
                        const Scalar bval(7, 11, 13, 17);
                        if (perImage)
                            warpOne(op, src, dst, Ms, to, interp, border, bval);
                        else
                            warpOne(op, src, dst, Ms[0], to, interp, border, bval);
                        ASSERT_EQ(3, dst.dims);
                        ASSERT_EQ(N, dst.size[0]);

                        for (int p = 0; p < N; p++)
                        {
                            Mat expected;
                            warpOne(op, plane(src, p), expected, Ms[perImage ? p : 0], to, interp, border, bval);
                            EXPECT_EQ(0, cvtest::norm(expected, plane(dst, p), NORM_INF))
                                << "op=" << op << " type=" << typeToString(type) << " interp=" << interp
                                << " border=" << border << " perImage=" << perImage << " plane=" << p;
                        }
                    }
}

// 4-D input is rejected as ambiguous; NCHW data goes through as an (N*C, H, W) view.
TEST(Warp_Batch, NCHWAsPlaneBatch)
{
    const int N = 3, C = 2;
    const Size from(48, 40), to(32, 36);
    int srcSizes[] = { N, C, from.height, from.width };
    Mat src(4, srcSizes, CV_32F);
    RNG(5).fill(src, RNG::UNIFORM, 0, 255);
    int planeSizes[] = { N*C, from.height, from.width };
    const Mat planes = src.reshape(1, 3, planeSizes);

    for (int op : { OP_AFFINE, OP_PERSPECTIVE })
    {
        std::vector<Mat> Ms;
        for (int p = 0; p < N*C; p++)
            Ms.push_back(makeTransform(op, from, 7 + p / C));

        Mat dst;
        EXPECT_THROW(warpOne(op, src, dst, Ms[0], to, INTER_LINEAR, BORDER_CONSTANT), cv::Exception);

        warpOne(op, planes, dst, Ms, to, INTER_LINEAR, BORDER_CONSTANT);
        ASSERT_EQ(3, dst.dims);
        EXPECT_EQ(N*C, dst.size[0]);
        EXPECT_EQ(to.height, dst.size[1]);
        EXPECT_EQ(to.width, dst.size[2]);

        for (int p = 0; p < N*C; p++)
        {
            Mat expected;
            warpOne(op, plane(planes, p), expected, Ms[p], to, INTER_LINEAR, BORDER_CONSTANT);
            EXPECT_EQ(0, cvtest::norm(expected, plane(dst, p), NORM_INF)) << "op=" << op << " plane=" << p;
        }
    }
}

// Per-image matrices as vector<Mat>, K x R x 3 Mat, or K x R 3-channel Mat (Python's (K, R, 3)).
TEST(Warp_Batch, MatrixLayouts)
{
    const int N = 3;
    const Size sz(30, 26);
    int srcSizes[] = { N, sz.height, sz.width };
    Mat src(3, srcSizes, CV_8UC3);
    RNG(9).fill(src, RNG::UNIFORM, 0, 255);

    for (int op : { OP_AFFINE, OP_PERSPECTIVE })
    {
        const int R = op == OP_AFFINE ? 2 : 3;
        std::vector<Mat> Ms;
        for (int n = 0; n < N; n++)
            Ms.push_back(makeTransform(op, sz, 21 + n));

        Mat reference;
        warpOne(op, src, reference, Ms, sz, INTER_LINEAR, BORDER_REPLICATE);

        int stackSizes[] = { N, R, 3 };
        Mat stacked(3, stackSizes, CV_64F);
        for (int n = 0; n < N; n++)
            Ms[n].copyTo(Mat(R, 3, CV_64F, stacked.ptr<double>(n)));
        Mat viaStacked;
        warpOne(op, src, viaStacked, stacked, sz, INTER_LINEAR, BORDER_REPLICATE);
        EXPECT_EQ(0, cvtest::norm(reference, viaStacked, NORM_INF)) << "op=" << op;

        Mat channels(N, R, CV_64FC3, stacked.data);
        Mat viaChannels;
        warpOne(op, src, viaChannels, channels, sz, INTER_LINEAR, BORDER_REPLICATE);
        EXPECT_EQ(0, cvtest::norm(reference, viaChannels, NORM_INF)) << "op=" << op;

        // float matrices are widened to double, as in the single-image call
        std::vector<Mat> Ms32(N);
        for (int n = 0; n < N; n++)
            Ms[n].convertTo(Ms32[n], CV_32F);
        Mat via32, expected32;
        warpOne(op, src, via32, Ms32, sz, INTER_LINEAR, BORDER_REPLICATE);
        for (int n = 0; n < N; n++)
        {
            warpOne(op, plane(src, n), expected32, Ms32[n], sz, INTER_LINEAR, BORDER_REPLICATE);
            EXPECT_EQ(0, cvtest::norm(expected32, plane(via32, n), NORM_INF)) << "op=" << op << " n=" << n;
        }
    }
}

TEST(Warp_Batch, InverseMap)
{
    const int N = 2;
    const Size sz(33, 27);
    int srcSizes[] = { N, sz.height, sz.width };
    Mat src(3, srcSizes, CV_8UC1);
    RNG(11).fill(src, RNG::UNIFORM, 0, 255);

    for (int op : { OP_AFFINE, OP_PERSPECTIVE })
    {
        std::vector<Mat> Ms;
        for (int n = 0; n < N; n++)
            Ms.push_back(makeTransform(op, sz, 31 + n));

        Mat dst;
        warpOne(op, src, dst, Ms, sz, INTER_LINEAR | WARP_INVERSE_MAP, BORDER_CONSTANT);
        for (int n = 0; n < N; n++)
        {
            Mat expected;
            warpOne(op, plane(src, n), expected, Ms[n], sz, INTER_LINEAR | WARP_INVERSE_MAP, BORDER_CONSTANT);
            EXPECT_EQ(0, cvtest::norm(expected, plane(dst, n), NORM_INF)) << "op=" << op << " n=" << n;
        }
    }
}

// Elements of a vector<Mat> batch may differ in size; an empty dsize keeps each one's own.
TEST(Warp_Batch, RaggedVector)
{
    std::vector<Mat> srcs = {
        makeTestImage(CV_8UC3, Size(100, 100), 1),
        makeTestImage(CV_8UC3, Size(64, 200), 2),
        makeTestImage(CV_8UC3, Size(300, 40), 3),
        makeTestImage(CV_8UC3, Size(100, 100), 4),
        makeTestImage(CV_8UC3, Size(5, 3), 5),
    };
    const int n = (int)srcs.size();

    for (int op : { OP_AFFINE, OP_PERSPECTIVE })
        for (int interp : { INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4 })
            for (Size dsize : { Size(48, 64), Size() })
                for (bool perImage : { false, true })
                {
                    std::vector<Mat> Ms;
                    for (int i = 0; i < (perImage ? n : 1); i++)
                        Ms.push_back(makeTransform(op, srcs[i].size(), 40 + i));

                    std::vector<Mat> actual;
                    if (perImage)
                        warpOne(op, srcs, actual, Ms, dsize, interp, BORDER_REFLECT);
                    else
                        warpOne(op, srcs, actual, Ms[0], dsize, interp, BORDER_REFLECT);
                    ASSERT_EQ(srcs.size(), actual.size());

                    for (int i = 0; i < n; i++)
                    {
                        Mat expected;
                        warpOne(op, srcs[i], expected, Ms[perImage ? i : 0], dsize, interp, BORDER_REFLECT);
                        EXPECT_EQ(dsize.empty() ? srcs[i].size() : dsize, actual[i].size());
                        EXPECT_EQ(0, cvtest::norm(expected, actual[i], NORM_INF))
                            << "op=" << op << " interp=" << interp << " dsize=" << dsize
                            << " perImage=" << perImage << " index=" << i;
                    }
                }
}

// Hundreds of tiny images: scheduler chunks span many images each.
TEST(Warp_Batch, ManySmallImages)
{
    const int N = 300;
    const Size sz(9, 7);
    int srcSizes[] = { N, sz.height, sz.width };
    Mat src(3, srcSizes, CV_8UC1);
    RNG(13).fill(src, RNG::UNIFORM, 0, 255);

    std::vector<Mat> Ms;
    for (int i = 0; i < N; i++)
        Ms.push_back(makeTransform(OP_AFFINE, sz, 1000 + i));

    Mat dst;
    warpAffine(src, dst, Ms, Size(11, 5));
    for (int p = 0; p < N; p++)
    {
        Mat expected;
        warpAffine(plane(src, p), expected, Ms[p], Size(11, 5));
        EXPECT_EQ(0, cvtest::norm(expected, plane(dst, p), NORM_INF)) << "plane=" << p;
    }
}

// BORDER_TRANSPARENT keeps whatever dst held, for a batch as for a single image.
TEST(Warp_Batch, TransparentBorder)
{
    const int N = 3;
    const Size sz(40, 30);
    int sizes[] = { N, sz.height, sz.width };
    Mat src(3, sizes, CV_8UC3), dst(3, sizes, CV_8UC3);
    RNG(17).fill(src, RNG::UNIFORM, 0, 255);
    dst.setTo(Scalar(1, 2, 3));
    Mat M = makeTransform(OP_AFFINE, sz, 3);
    M.at<double>(0, 2) += 15;

    warpAffine(src, dst, M, sz, INTER_LINEAR, BORDER_TRANSPARENT);
    for (int p = 0; p < N; p++)
    {
        Mat expected(sz, CV_8UC3, Scalar(1, 2, 3));
        warpAffine(plane(src, p), expected, M, sz, INTER_LINEAR, BORDER_TRANSPARENT);
        EXPECT_EQ(0, cvtest::norm(expected, plane(dst, p), NORM_INF)) << "plane=" << p;
    }
}

// dst aliasing src reads the source before it is overwritten.
TEST(Warp_Batch, InPlace)
{
    const int N = 2;
    const Size sz(35, 25);
    int sizes[] = { N, sz.height, sz.width };
    Mat src(3, sizes, CV_8UC1);
    RNG(19).fill(src, RNG::UNIFORM, 0, 255);
    const Mat M = makeTransform(OP_AFFINE, sz, 5);

    Mat expected;
    warpAffine(src, expected, M, sz);
    Mat inplace = src.clone();
    warpAffine(inplace, inplace, M, sz);
    EXPECT_EQ(0, cvtest::norm(expected.reshape(1, N*sz.height), inplace.reshape(1, N*sz.height), NORM_INF));

    std::vector<Mat> v = { makeTestImage(CV_8UC1, sz, 1), makeTestImage(CV_8UC1, sz, 2) };
    std::vector<Mat> vExpected;
    warpAffine(v, vExpected, M, sz);
    warpAffine(v, v, M, sz);
    for (int i = 0; i < N; i++)
        EXPECT_EQ(0, cvtest::norm(vExpected[i], v[i], NORM_INF)) << "index=" << i;
}

TEST(Warp_Batch, RejectsInvalidInputs)
{
    const Size sz(20, 20);
    int sizes[] = { 3, sz.height, sz.width };
    Mat src(3, sizes, CV_8UC1, Scalar(0)), dst;
    const Mat M = makeTransform(OP_AFFINE, sz, 1);

    std::vector<Mat> twoMs = { M, M };
    EXPECT_THROW(warpAffine(src, dst, twoMs, sz), cv::Exception);             // 2 matrices, 3 images
    EXPECT_THROW(warpAffine(src, dst, Mat::eye(3, 3, CV_64F), sz), cv::Exception);
    EXPECT_THROW(warpPerspective(src, dst, M, sz), cv::Exception);           // 2x3 for a 3x3 op
    EXPECT_THROW(warpAffine(src, dst, Mat(M.size(), CV_32S, Scalar(0)), sz), cv::Exception);

    std::vector<Mat> mixed = { makeTestImage(CV_8UC1, sz, 1), makeTestImage(CV_8UC3, sz, 2) };
    std::vector<Mat> vdst;
    EXPECT_THROW(warpAffine(mixed, vdst, M, sz), cv::Exception);
}

// A vector<UMat> batch goes image by image through the OpenCL path.
TEST(Warp_Batch, UMatVector)
{
    std::vector<UMat> srcs(2);
    makeTestImage(CV_8UC3, Size(120, 90), 11).copyTo(srcs[0]);
    makeTestImage(CV_8UC3, Size(90, 120), 22).copyTo(srcs[1]);
    const Size dsize(64, 48);

    for (int op : { OP_AFFINE, OP_PERSPECTIVE })
    {
        std::vector<Mat> Ms = { makeTransform(op, Size(120, 90), 1), makeTransform(op, Size(90, 120), 2) };
        std::vector<UMat> actual;
        warpOne(op, srcs, actual, Ms, dsize, INTER_LINEAR, BORDER_CONSTANT);
        ASSERT_EQ(srcs.size(), actual.size());
        for (size_t i = 0; i < srcs.size(); i++)
        {
            UMat expected;
            warpOne(op, srcs[i], expected, Ms[i], dsize, INTER_LINEAR, BORDER_CONSTANT);
            EXPECT_EQ(0, cvtest::norm(expected.getMat(ACCESS_READ), actual[i].getMat(ACCESS_READ), NORM_INF))
                << "op=" << op << " index=" << i;
        }
    }
}

}} // namespace
