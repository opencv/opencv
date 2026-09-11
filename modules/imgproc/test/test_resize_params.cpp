// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the
// top-level directory of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

static Mat makeTestImage(int type, Size sz, int seed)
{
    Mat img(sz, type);
    RNG rng(seed);
    rng.fill(img, RNG::UNIFORM, 0, 255);
    return img;
}

static double referenceSrcCoord(int dst, double scale, int outLen, ResizeCoord mode)
{
    switch (mode)
    {
    case ResizeCoord::PYTORCH_HALF_PIXEL:
        return outLen > 1 ? (dst + 0.5) * scale - 0.5 : 0.0;
    case ResizeCoord::TF_HALF_PIXEL_FOR_NN:
        return (dst + 0.5) * scale;
    case ResizeCoord::ASYMMETRIC:
    case ResizeCoord::ALIGN_CORNERS:
        return dst * scale;
    case ResizeCoord::HALF_PIXEL:
    case ResizeCoord::HALF_PIXEL_SYMMETRIC:
    default:
        return (dst + 0.5) * scale - 0.5;
    }
}

static void referenceCubicWeights(float x, float A, float w[4])
{
    w[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    w[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    w[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    w[3] = 1.f - w[0] - w[1] - w[2];
}

static float referenceCubic1D(const std::vector<float>& src, double dstCoordScale, int dst,
                              int outLen, ResizeCoord mode, bool excludeOutside, float A)
{
    const int inLen = (int)src.size();
    double srcCoord = referenceSrcCoord(dst, dstCoordScale, outLen, mode);
    int i = (int)std::floor(srcCoord);
    float w[4];
    referenceCubicWeights((float)(srcCoord - i), A, w);

    float acc = 0.f, sw = 0.f;
    for (int k = -1; k <= 2; k++)
    {
        int idx = i + k;
        bool valid = idx >= 0 && idx < inLen;
        float wk = w[k + 1];
        if (excludeOutside && !valid) wk = 0.f;
        else if (!valid) idx = std::min(std::max(idx, 0), inLen - 1);
        sw += wk;
        acc += wk * src[idx];
    }
    return sw != 0.f ? acc / sw : acc;
}

TEST(Resize_Params, BackwardCompat)
{
    const int interpolations[] = { INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4 };
    const Size srcSize(173, 121), dstSize(64, 96);

    for (int interp : interpolations)
    {
        Mat src = makeTestImage(CV_8UC3, srcSize, 12345 + interp);

        Mat expected, actual;
        resize(src, expected, dstSize, 0, 0, interp);

        ResizeParams params(dstSize, 0, 0, interp);
        resize(src, actual, params);

        EXPECT_EQ(0, cvtest::norm(expected, actual, NORM_INF))
            << "interpolation=" << interp;
    }

    {
        const Size bitExactSrcSize(342, 256), bitExactDstSize(171, 128);
        Mat src = makeTestImage(CV_8UC3, bitExactSrcSize, 777);

        Mat expectedLinear, actualLinear;
        resize(src, expectedLinear, bitExactDstSize, 0, 0, INTER_LINEAR_EXACT);
        ResizeParams linearParams(bitExactDstSize, 0, 0, INTER_LINEAR);
        linearParams.bitExact = true;
        resize(src, actualLinear, linearParams);
        EXPECT_EQ(0, cvtest::norm(expectedLinear, actualLinear, NORM_INF));

        Mat expectedNearest, actualNearest;
        resize(src, expectedNearest, bitExactDstSize, 0, 0, INTER_NEAREST_EXACT);
        ResizeParams nearestParams(bitExactDstSize, 0, 0, INTER_NEAREST);
        nearestParams.bitExact = true;
        resize(src, actualNearest, nearestParams);
        EXPECT_EQ(0, cvtest::norm(expectedNearest, actualNearest, NORM_INF));
    }

    {
        // The batch kinds accept the classic argument list too, and both spellings must agree.
        const int N = 2, C = 3, inH = 20, inW = 16, outH = 8, outW = 12;
        int srcSizes[] = { N, C, inH, inW };
        Mat ndSrc(4, srcSizes, CV_32F);
        RNG rng(99);
        rng.fill(ndSrc, RNG::UNIFORM, 0, 255);

        Mat viaClassic, viaParams;
        resize(ndSrc, viaClassic, Size(outW, outH), 0, 0, INTER_LINEAR);
        resize(ndSrc, viaParams, ResizeParams(Size(outW, outH)));
        ASSERT_EQ(viaClassic.dims, 4);
        EXPECT_EQ(0, cvtest::norm(viaClassic, viaParams, NORM_INF));

        const Size vecDstSize(48, 64);
        std::vector<Mat> srcs = {
            makeTestImage(CV_8UC3, Size(100, 100), 4),
            makeTestImage(CV_8UC3, Size(64, 200), 5),
        };
        std::vector<Mat> viaClassicVec, viaParamsVec;
        resize(srcs, viaClassicVec, vecDstSize, 0, 0, INTER_LINEAR);
        resize(srcs, viaParamsVec, ResizeParams(vecDstSize));
        ASSERT_EQ(srcs.size(), viaClassicVec.size());
        ASSERT_EQ(srcs.size(), viaParamsVec.size());
        for (size_t i = 0; i < srcs.size(); i++)
            EXPECT_EQ(0, cvtest::norm(viaClassicVec[i], viaParamsVec[i], NORM_INF)) << "index=" << i;
    }
}

// Reference is a one-element batch: only the single-image path can reach the HAL.
TEST(Resize_Params, BatchMatchesSingleImage)
{
    const int interpolations[] = { INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4 };
    const int types[] = { CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC3, CV_32FC1, CV_32FC4, CV_64FC1 };
    // Up, down, and an integer 2x down for the INTER_AREA fast path.
    const Size sizes[][2] = { { Size(37, 29), Size(64, 51) }, { Size(64, 51), Size(37, 29) },
                              { Size(64, 48), Size(32, 24) } };
    const int N = 5;

    for (int type : types)
        for (int interp : interpolations)
            for (const Size* sz : sizes)
            {
                const Size from = sz[0], to = sz[1];
                int srcSizes[] = { N, from.height, from.width };
                Mat src(3, srcSizes, type);
                RNG(0xbeef + interp).fill(src, RNG::UNIFORM, 0, 255);

                ResizeParams params(to, 0, 0, interp);
                Mat dst;
                resize(src, dst, params);
                ASSERT_EQ(dst.dims, 3);

                const size_t srcPlaneBytes = from.area() * src.elemSize();
                const size_t dstPlaneBytes = to.area() * dst.elemSize();
                std::vector<Mat> plane(1);
                for (int p = 0; p < N; p++)
                {
                    plane[0] = Mat(from, type, src.data + p * srcPlaneBytes);
                    Mat actual(to, type, dst.data + p * dstPlaneBytes);
                    std::vector<Mat> alone;
                    resize(plane, alone, params);
                    EXPECT_EQ(0, cvtest::norm(alone[0], actual, NORM_INF))
                        << "type=" << typeToString(type) << " interpolation=" << interp
                        << " " << from << "->" << to << " plane=" << p;
                }
            }
}

// A ragged batch shares tables only between matching elements; row counts differ.
TEST(Resize_Params, RaggedBatch)
{
    std::vector<Mat> srcs = {
        makeTestImage(CV_8UC3, Size(100, 100), 1),
        makeTestImage(CV_8UC3, Size(64, 200), 2),
        makeTestImage(CV_8UC3, Size(300, 40), 3),
        makeTestImage(CV_8UC3, Size(100, 100), 4),  // repeats the first geometry, reuses its plan
    };

    // dsize fixed for all elements...
    std::vector<Mat> actual;
    resize(srcs, actual, ResizeParams(Size(48, 64)));
    ASSERT_EQ(srcs.size(), actual.size());
    for (size_t i = 0; i < srcs.size(); i++)
    {
        std::vector<Mat> one(1, srcs[i]), alone;
        resize(one, alone, ResizeParams(Size(48, 64)));
        EXPECT_EQ(0, cvtest::norm(alone[0], actual[i], NORM_INF)) << "index=" << i;
    }

    // ...and dsize derived per element from fx/fy, which makes the outputs differently sized.
    ResizeParams scaled;
    scaled.fx = scaled.fy = 0.5;
    resize(srcs, actual, scaled);
    ASSERT_EQ(srcs.size(), actual.size());
    for (size_t i = 0; i < srcs.size(); i++)
    {
        EXPECT_EQ(Size(srcs[i].cols/2, srcs[i].rows/2), actual[i].size()) << "index=" << i;
        std::vector<Mat> one(1, srcs[i]), alone;
        resize(one, alone, scaled);
        EXPECT_EQ(0, cvtest::norm(alone[0], actual[i], NORM_INF)) << "index=" << i;
    }
}

// Asking for the source size back is a copy, for a batch too.
TEST(Resize_Params, BatchIdentitySize)
{
    const int N = 3;
    const Size sz(21, 17);
    int srcSizes[] = { N, sz.height, sz.width };
    Mat src(3, srcSizes, CV_8UC3);
    RNG(7).fill(src, RNG::UNIFORM, 0, 255);

    const int interpolations[] = { INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4 };
    for (int interp : interpolations)
    {
        Mat dst;
        resize(src, dst, ResizeParams(sz, 0, 0, interp));
        EXPECT_EQ(0, cvtest::norm(src.reshape(1, N * sz.height), dst.reshape(1, N * sz.height), NORM_INF))
            << "interpolation=" << interp;
    }
}

// CV_64F must be interpolated in double, not narrowed to float on the way.
TEST(Resize_Params, DoublePrecisionCoordMode)
{
    const int inW = 9, outW = 5;
    Mat src(1, inW, CV_64FC1);
    // Values whose differences are far below float resolution: a float accumulator collapses them.
    for (int x = 0; x < inW; x++)
        src.at<double>(0, x) = 1.0 + x * 1e-12;

    const int interpolations[] = { INTER_LINEAR, INTER_CUBIC };
    for (int interp : interpolations)
    {
        ResizeParams params(Size(outW, 1), 0, 0, interp);
        params.coordMode = ResizeCoord::HALF_PIXEL;

        Mat out;
        resize(src, out, params);
        ASSERT_EQ(out.depth(), CV_64F);

        // Detail survives and values stay near the input; cubic may overshoot slightly.
        double lo, hi;
        minMaxLoc(out, &lo, &hi);
        EXPECT_GT(hi - lo, 1e-13) << "interpolation=" << interp << ": detail lost to float";
        EXPECT_LT(std::abs(hi - 1.0), 1e-10) << "interpolation=" << interp;
        EXPECT_LT(std::abs(lo - 1.0), 1e-10) << "interpolation=" << interp;
    }
}

TEST(Resize_Params, RejectsInvalidInputs)
{
    Mat src = makeTestImage(CV_8UC3, Size(64, 64), 1), dst;

    ResizeParams bitExactCubic(Size(32, 32), 0, 0, INTER_CUBIC);
    bitExactCubic.bitExact = true;
    EXPECT_THROW(resize(src, dst, bitExactCubic), cv::Exception);

    ResizeParams legacyExactSentinel(Size(32, 32), 0, 0, INTER_LINEAR_EXACT);
    EXPECT_THROW(resize(src, dst, legacyExactSentinel), cv::Exception);

    ResizeParams coordModeArea(Size(32, 32), 0, 0, INTER_AREA);
    coordModeArea.coordMode = ResizeCoord::HALF_PIXEL;
    EXPECT_THROW(resize(src, dst, coordModeArea), cv::Exception);

    ResizeParams coordModeLanczos(Size(32, 32), 0, 0, INTER_LANCZOS4);
    coordModeLanczos.coordMode = ResizeCoord::HALF_PIXEL;
    EXPECT_THROW(resize(src, dst, coordModeLanczos), cv::Exception);

    ResizeParams coordModeBitExact(Size(32, 32));
    coordModeBitExact.coordMode = ResizeCoord::HALF_PIXEL;
    coordModeBitExact.bitExact = true;
    EXPECT_THROW(resize(src, dst, coordModeBitExact), cv::Exception);

    ResizeParams antialias(Size(32, 32));
    antialias.antialias = true;
    EXPECT_THROW(resize(src, dst, antialias), cv::Exception);

    ResizeParams antialiasWithCoordMode(Size(32, 32));
    antialiasWithCoordMode.coordMode = ResizeCoord::HALF_PIXEL;
    antialiasWithCoordMode.antialias = true;
    EXPECT_THROW(resize(src, dst, antialiasWithCoordMode), cv::Exception);

    int ndSizes[] = { 2, 3, 10, 10 };
    Mat ndSrc(4, ndSizes, CV_32F), ndDst;
    EXPECT_THROW(resize(ndSrc, ndDst, ResizeParams()), cv::Exception);   // neither dsize nor fx/fy

    std::vector<Mat> mismatched = {
        makeTestImage(CV_8UC3, Size(64, 64), 1),
        makeTestImage(CV_8UC1, Size(64, 64), 2),
    };
    std::vector<Mat> vecDst;
    EXPECT_THROW(resize(mismatched, vecDst, ResizeParams(Size(32, 32))), cv::Exception);

    std::vector<Mat> uniform = { makeTestImage(CV_8UC3, Size(64, 64), 3) };
    EXPECT_THROW(resize(uniform, vecDst, ResizeParams()), cv::Exception);
}

TEST(Resize_Params, CoordModeMath)
{
    {
        const ResizeCoord modes[] = {
            ResizeCoord::HALF_PIXEL, ResizeCoord::PYTORCH_HALF_PIXEL,
            ResizeCoord::ASYMMETRIC, ResizeCoord::ALIGN_CORNERS,
            ResizeCoord::TF_HALF_PIXEL_FOR_NN, ResizeCoord::HALF_PIXEL_SYMMETRIC
        };
        const int inW = 17, outW = 6;

        Mat ramp(1, inW, CV_32FC1);
        for (int x = 0; x < inW; x++)
            ramp.at<float>(0, x) = (float)x;

        for (ResizeCoord mode : modes)
        {
            double scale = (mode == ResizeCoord::ALIGN_CORNERS && outW > 1)
                             ? (double)(inW - 1) / (outW - 1)
                             : (double)inW / outW;

            ResizeParams params;
            params.interpolation = INTER_LINEAR;
            params.coordMode = mode;

            params.dsize = Size(outW, 1);

            Mat out;
            resize(ramp, out, params);
            ASSERT_EQ(out.size(), Size(outW, 1));

            for (int x = 0; x < outW; x++)
            {
                double expected = std::min(std::max(referenceSrcCoord(x, scale, outW, mode), 0.0), (double)(inW - 1));
                EXPECT_NEAR(expected, out.at<float>(0, x), 1e-3)
                    << "mode=" << (int)mode << " x=" << x;
            }
        }
    }

    {
        const int inW = 67, inH = 5, outW = 41, outH = 3, cn = 3;
        Mat src(inH, inW, CV_32FC(cn));
        for (int y = 0; y < inH; y++)
            for (int x = 0; x < inW; x++)
                for (int c = 0; c < cn; c++)
                    src.at<Vec3f>(y, x)[c] = (float)x + (float)y * 1000.f + (float)c;

        ResizeParams params;
        params.interpolation = INTER_LINEAR;
        params.coordMode = ResizeCoord::HALF_PIXEL;

        params.dsize = Size(outW, outH);

        Mat out;
        resize(src, out, params);
        ASSERT_EQ(out.size(), Size(outW, outH));

        const double scaleX = (double)inW / outW, scaleY = (double)inH / outH;
        for (int oy = 0; oy < outH; oy++)
            for (int ox = 0; ox < outW; ox++)
            {
                double sx = std::min(std::max(referenceSrcCoord(ox, scaleX, outW, ResizeCoord::HALF_PIXEL), 0.0), (double)(inW - 1));
                double sy = std::min(std::max(referenceSrcCoord(oy, scaleY, outH, ResizeCoord::HALF_PIXEL), 0.0), (double)(inH - 1));
                for (int c = 0; c < cn; c++)
                {
                    float expected = (float)(sx + sy * 1000.0 + c);
                    EXPECT_NEAR(expected, out.at<Vec3f>(oy, ox)[c], 5e-2)
                        << "oy=" << oy << " ox=" << ox << " c=" << c;
                }
            }
    }

    {
        const int inW = 17, outW = 6;
        Mat ramp(1, inW, CV_32FC1);
        for (int x = 0; x < inW; x++)
            ramp.at<float>(0, x) = (float)x;

        ResizeParams params;
        params.interpolation = INTER_NEAREST;
        params.coordMode = ResizeCoord::HALF_PIXEL;
        params.nearestMode = ResizeNearest::ROUND_PREFER_FLOOR;

        params.dsize = Size(outW, 1);

        Mat out;
        resize(ramp, out, params);

        double scale = (double)inW / outW;
        for (int x = 0; x < outW; x++)
        {
            double src = std::min(std::max(referenceSrcCoord(x, scale, outW, ResizeCoord::HALF_PIXEL), 0.0), (double)(inW - 1));
            double frac = src - std::floor(src);
            int expectedIdx = (std::abs(frac - 0.5) <= 1e-6) ? (int)std::floor(src) : cvRound(src);
            EXPECT_FLOAT_EQ((float)expectedIdx, out.at<float>(0, x)) << "x=" << x;
        }
    }

    {
        Mat ramp(1, 8, CV_32FC1);
        for (int x = 0; x < 8; x++)
            ramp.at<float>(0, x) = (float)x;

        struct { ResizeNearest mode; float expected; } cases[] = {
            { ResizeNearest::FLOOR, 1.f },
            { ResizeNearest::CEIL, 2.f },
            { ResizeNearest::ROUND_PREFER_CEIL, 2.f },
            { ResizeNearest::ROUND_PREFER_FLOOR, 1.f },
        };

        for (const auto& c : cases)
        {
            ResizeParams params;
            params.interpolation = INTER_NEAREST;
            params.coordMode = ResizeCoord::TF_HALF_PIXEL_FOR_NN;
            params.nearestMode = c.mode;

            params.dsize = Size(8, 1);

            Mat out;
            resize(ramp, out, params);
            EXPECT_FLOAT_EQ(c.expected, out.at<float>(0, 1))
                << "nearestMode=" << (int)c.mode;
        }
    }

    {
        Mat src = (Mat_<float>(2, 4) << 1, 2, 3, 4, 5, 6, 7, 8);

        ResizeParams params;
        params.interpolation = INTER_LINEAR;
        params.coordMode = ResizeCoord::ALIGN_CORNERS;

        params.dsize = Size(2, 1);
        params.fx = params.fy = 0.6;

        Mat out;
        resize(src, out, params);
        ASSERT_EQ(out.size(), Size(2, 1));
        EXPECT_NEAR(1.f, out.at<float>(0, 0), 1e-4);
        EXPECT_NEAR(3.142857f, out.at<float>(0, 1), 1e-4);
    }

    {
        Mat ramp(1, 4, CV_32FC1);
        for (int x = 0; x < 4; x++)
            ramp.at<float>(0, x) = (float)x;

        ResizeParams symmetric(Size(2, 1), 0.6, 0.6);
        symmetric.coordMode = ResizeCoord::HALF_PIXEL_SYMMETRIC;

        ResizeParams halfPixel = symmetric;
        halfPixel.coordMode = ResizeCoord::HALF_PIXEL;

        // With the true scale known, half_pixel_symmetric shifts the sampling grid...
        Mat outSymmetric, outHalfPixel;
        resize(ramp, outSymmetric, symmetric);
        resize(ramp, outHalfPixel, halfPixel);
        EXPECT_GT(cv::norm(outSymmetric, outHalfPixel, NORM_INF), 1e-3);

        // ...but with the scale re-derived from dsize alone the two modes coincide.
        symmetric.fx = symmetric.fy = 0;
        halfPixel.fx = halfPixel.fy = 0;
        Mat outSymmetricNoFx, outHalfPixelNoFx;
        resize(ramp, outSymmetricNoFx, symmetric);
        resize(ramp, outHalfPixelNoFx, halfPixel);
        EXPECT_EQ(0, cv::norm(outSymmetricNoFx, outHalfPixelNoFx, NORM_INF));
    }

    {
        const std::vector<float> src = { 3.f, 1.f, 4.f, 1.f, 5.f, 9.f, 2.f, 6.f };
        const int inW = (int)src.size(), outW = 5;
        Mat srcMat(1, inW, CV_32FC1, (void*)src.data());

        struct { ResizeCoord mode; bool excludeOutside; } cases[] = {
            { ResizeCoord::HALF_PIXEL, false },
            { ResizeCoord::HALF_PIXEL, true },
            { ResizeCoord::ASYMMETRIC, false },
            { ResizeCoord::PYTORCH_HALF_PIXEL, true },
        };

        for (auto& c : cases)
        {
            ResizeParams params;
            params.interpolation = INTER_CUBIC;
            params.coordMode = c.mode;
            params.excludeOutside = c.excludeOutside;

            params.dsize = Size(outW, 1);

            Mat out;
            resize(srcMat, out, params);
            ASSERT_EQ(out.cols, outW);

            double scale = (double)inW / outW;
            for (int x = 0; x < outW; x++)
            {
                float expected = referenceCubic1D(src, scale, x, outW, c.mode, c.excludeOutside, -0.75f);
                EXPECT_NEAR(expected, out.at<float>(0, x), 1e-3)
                    << "mode=" << (int)c.mode << " excludeOutside=" << c.excludeOutside << " x=" << x;
            }
        }
    }

    {
        const std::vector<float> src = { 10.f, 0.f, 0.f, 0.f, 0.f, 10.f };
        Mat srcMat(1, (int)src.size(), CV_32FC1, (void*)src.data());

        ResizeParams clamped(Size(12, 1), 0, 0, INTER_CUBIC);
        clamped.coordMode = ResizeCoord::ASYMMETRIC;
        clamped.excludeOutside = false;

        ResizeParams excluded = clamped;
        excluded.excludeOutside = true;

        Mat outClamped, outExcluded;
        resize(srcMat, outClamped, clamped);
        resize(srcMat, outExcluded, excluded);
        EXPECT_GT(cv::norm(outClamped, outExcluded, NORM_INF), 1e-3);
    }
}

TEST(Resize_Params, BatchMechanics)
{
    {
        const int N = 2, C = 3, inH = 20, inW = 16, outH = 8, outW = 12;
        int srcSizes[] = { N, C, inH, inW };
        Mat src(4, srcSizes, CV_32F);
        RNG rng(42);
        rng.fill(src, RNG::UNIFORM, 0, 255);

        Mat dst;
        resize(src, dst, ResizeParams(Size(outW, outH)));
        ASSERT_EQ(dst.dims, 4);
        EXPECT_EQ(dst.size[0], N);
        EXPECT_EQ(dst.size[1], C);
        EXPECT_EQ(dst.size[2], outH);
        EXPECT_EQ(dst.size[3], outW);

        Mat src2D = src.reshape(1, N * C * inH);
        Mat dst2D = dst.reshape(1, N * C * outH);
        for (int p = 0; p < N * C; p++)
        {
            // See Resize_Params.BatchMatchesSingleImage for why the reference is a batch of one.
            std::vector<Mat> plane(1, src2D.rowRange(p * inH, (p + 1) * inH)), alone;
            resize(plane, alone, ResizeParams(Size(outW, outH)));
            Mat actualPlane = dst2D.rowRange(p * outH, (p + 1) * outH);
            EXPECT_EQ(0, cvtest::norm(alone[0], actualPlane, NORM_INF)) << "plane=" << p;
        }
    }

    {
        const int N = 2, inH = 9, inW = 13, outH = 4, outW = 5;
        int srcSizes[] = { N, inH, inW };
        Mat src(3, srcSizes, CV_32F);
        Mat srcFill = src.reshape(1, N * inH);
        for (int n = 0; n < N; n++)
            for (int y = 0; y < inH; y++)
            {
                float* row = srcFill.ptr<float>(n * inH + y);
                for (int x = 0; x < inW; x++)
                    row[x] = (float)x + (float)y * 1000.f;
            }

        ResizeParams params;
        params.interpolation = INTER_LINEAR;
        params.coordMode = ResizeCoord::HALF_PIXEL;

        params.dsize = Size(outW, outH);

        Mat dst;
        resize(src, dst, params);
        ASSERT_EQ(dst.dims, 3);
        EXPECT_EQ(dst.size[0], N);
        EXPECT_EQ(dst.size[1], outH);
        EXPECT_EQ(dst.size[2], outW);

        Mat dstView = dst.reshape(1, N * outH);
        const double scaleX = (double)inW / outW, scaleY = (double)inH / outH;
        for (int n = 0; n < N; n++)
            for (int oy = 0; oy < outH; oy++)
            {
                const float* row = dstView.ptr<float>(n * outH + oy);
                for (int ox = 0; ox < outW; ox++)
                {
                    double sx = std::min(std::max(referenceSrcCoord(ox, scaleX, outW, ResizeCoord::HALF_PIXEL), 0.0), (double)(inW - 1));
                    double sy = std::min(std::max(referenceSrcCoord(oy, scaleY, outH, ResizeCoord::HALF_PIXEL), 0.0), (double)(inH - 1));
                    float expected = (float)(sx + sy * 1000.0);
                    EXPECT_NEAR(expected, row[ox], 5e-2) << "n=" << n << " oy=" << oy << " ox=" << ox;
                }
            }
    }

    {
        // The UMat batch still goes image by image through the OpenCL path.
        const Size dstSize(48, 64);
        std::vector<UMat> srcs(2);
        makeTestImage(CV_8UC3, Size(120, 90), 11).copyTo(srcs[0]);
        makeTestImage(CV_8UC3, Size(90, 120), 22).copyTo(srcs[1]);

        std::vector<UMat> actual;
        resize(srcs, actual, ResizeParams(dstSize));

        ASSERT_EQ(srcs.size(), actual.size());
        for (size_t i = 0; i < srcs.size(); i++)
        {
            UMat expected;
            resize(srcs[i], expected, dstSize, 0, 0, INTER_LINEAR);
            EXPECT_EQ(0, cvtest::norm(expected.getMat(ACCESS_READ), actual[i].getMat(ACCESS_READ), NORM_INF)) << "index=" << i;
        }
    }
}

}} // namespace
