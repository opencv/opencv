// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

static const int allOperators[] = {
    ALPHA_COMPOSITE_CLEAR, ALPHA_COMPOSITE_SOURCE, ALPHA_COMPOSITE_DEST,
    ALPHA_COMPOSITE_OVER, ALPHA_COMPOSITE_DEST_OVER,
    ALPHA_COMPOSITE_IN, ALPHA_COMPOSITE_DEST_IN,
    ALPHA_COMPOSITE_OUT, ALPHA_COMPOSITE_DEST_OUT,
    ALPHA_COMPOSITE_ATOP, ALPHA_COMPOSITE_DEST_ATOP,
    ALPHA_COMPOSITE_XOR, ALPHA_COMPOSITE_PLUS
};

// Porter-Duff weights straight from the W3C table, independently of the fixed-point implementation.
static void referenceWeights(int op, double as, double ad, double& fa, double& fb)
{
    switch (op)
    {
    case ALPHA_COMPOSITE_CLEAR:     fa = 0;      fb = 0;      break;
    case ALPHA_COMPOSITE_SOURCE:    fa = 1;      fb = 0;      break;
    case ALPHA_COMPOSITE_DEST:      fa = 0;      fb = 1;      break;
    case ALPHA_COMPOSITE_OVER:      fa = 1;      fb = 1 - as; break;
    case ALPHA_COMPOSITE_DEST_OVER: fa = 1 - ad; fb = 1;      break;
    case ALPHA_COMPOSITE_IN:        fa = ad;     fb = 0;      break;
    case ALPHA_COMPOSITE_DEST_IN:   fa = 0;      fb = as;     break;
    case ALPHA_COMPOSITE_OUT:       fa = 1 - ad; fb = 0;      break;
    case ALPHA_COMPOSITE_DEST_OUT:  fa = 0;      fb = 1 - as; break;
    case ALPHA_COMPOSITE_ATOP:      fa = ad;     fb = 1 - as; break;
    case ALPHA_COMPOSITE_DEST_ATOP: fa = 1 - ad; fb = as;     break;
    case ALPHA_COMPOSITE_XOR:       fa = 1 - ad; fb = 1 - as; break;
    case ALPHA_COMPOSITE_PLUS:      fa = 1;      fb = 1;      break;
    default: CV_Error(Error::StsBadArg, "unknown operator");
    }
}

// Takes straight-alpha inputs and returns the premultiplied result; straight color is unstable
// when the composited alpha is near zero, so comparisons are done in premultiplied space.
static void referenceCompositePremultiplied(const Mat& overlay, const Mat& background, int op, Mat& dst)
{
    CV_Assert(overlay.type() == CV_8UC4);
    int bgChannels = background.channels();
    dst.create(background.size(), background.type());

    for (int y = 0; y < overlay.rows; ++y)
    {
        const uchar* ov = overlay.ptr<uchar>(y);
        const uchar* bg = background.ptr<uchar>(y);
        uchar* d = dst.ptr<uchar>(y);
        for (int x = 0; x < overlay.cols; ++x, ov += 4, bg += bgChannels, d += bgChannels)
        {
            double as = ov[3] / 255.0;
            double ad = (bgChannels == 4) ? bg[3] / 255.0 : 1.0;
            double fa = 0, fb = 0;
            referenceWeights(op, as, ad, fa, fb);

            for (int c = 0; c < 3; ++c)
            {
                double premul = ov[c] * as * fa + bg[c] * ad * fb;
                d[c] = saturate_cast<uchar>(cvRound(premul));
            }
            if (bgChannels == 4)
                d[3] = saturate_cast<uchar>(cvRound(255.0 * (as * fa + ad * fb)));
        }
    }
}

static double maxAbsDiff(const Mat& a, const Mat& b)
{
    Mat diff;
    absdiff(a, b, diff);
    double maxVal = 0;
    std::vector<Mat> planes;
    split(diff, planes);
    for (size_t i = 0; i < planes.size(); ++i)
    {
        double planeMax = 0;
        minMaxLoc(planes[i], nullptr, &planeMax);
        maxVal = std::max(maxVal, planeMax);
    }
    return maxVal;
}

// Fixed-point rounding vs. the float reference differs by ~2; expected, not a bug.
static const double kRoundingTolerance = 2.0;

typedef testing::TestWithParam<std::tuple<int, bool, int>> Imgproc_AlphaComposite_Correctness;

TEST_P(Imgproc_AlphaComposite_Correctness, MatchesReference)
{
    int bgChannels = std::get<0>(GetParam());
    bool premultiplied = std::get<1>(GetParam());
    int op = std::get<2>(GetParam());

    Size size(157, 83);
    Mat overlay(size, CV_8UC4), background(size, CV_8UC(bgChannels));
    randu(overlay, 0, 256);
    randu(background, 0, 256);

    // Fresh, non-aliased Mats regardless of backend; reference() keeps the original straight values.
    Mat overlayInput;
    if (premultiplied)
        cvtColor(overlay, overlayInput, COLOR_RGBA2mRGBA);
    else
        overlayInput = overlay;

    // premultiplied=true needs an already-premultiplied background too, when it has alpha.
    Mat backgroundInput;
    if (premultiplied && bgChannels == 4)
        cvtColor(background, backgroundInput, COLOR_RGBA2mRGBA);
    else
        backgroundInput = background;

    Mat dst;
    alphaComposite(overlayInput, backgroundInput, dst, op, premultiplied);

    ASSERT_EQ(dst.type(), background.type());
    ASSERT_EQ(dst.size(), background.size());

    Mat dstPremul;
    if (bgChannels == 4 && !premultiplied)
        cvtColor(dst, dstPremul, COLOR_RGBA2mRGBA);
    else
        dstPremul = dst;

    Mat ref;
    referenceCompositePremultiplied(overlay, background, op, ref);
    EXPECT_LE(maxAbsDiff(dstPremul, ref), kRoundingTolerance);
}

INSTANTIATE_TEST_CASE_P(Imgproc, Imgproc_AlphaComposite_Correctness,
                        testing::Combine(testing::Values(3, 4), testing::Bool(),
                                         testing::ValuesIn(allOperators)));

TEST(Imgproc_AlphaComposite, NoFringingAtBoundariesAndEdges)
{
    Size size(64, 48);

    { // Garbage RGB behind alpha=0 mimics the classic leak from skipping premultiplication.
        Mat overlay(size, CV_8UC4, Scalar(255, 255, 255, 0));
        Mat background(size, CV_8UC3);
        randu(background, 0, 256);
        Mat dst;
        alphaComposite(overlay, background, dst);
        EXPECT_EQ(0.0, cv::norm(dst, background, NORM_INF));
    }

    { // Alpha=255 must reproduce the overlay's color exactly, regardless of the background.
        Mat overlay(size, CV_8UC4);
        randu(overlay, 0, 256);
        std::vector<Mat> channels;
        split(overlay, channels);
        channels[3].setTo(255);
        merge(channels, overlay);
        Mat background(size, CV_8UC3);
        randu(background, 0, 256);
        Mat dst, overlayColor;
        alphaComposite(overlay, background, dst);
        cvtColor(overlay, overlayColor, COLOR_BGRA2BGR);
        EXPECT_EQ(0.0, cv::norm(dst, overlayColor, NORM_INF));
    }

    { // Alpha ramp models an antialiased edge; blended values must stay within [background, overlay].
        const int width = 256, height = 8;
        Mat overlay(height, width, CV_8UC4);
        Mat background(height, width, CV_8UC3, Scalar(10, 200, 30));
        for (int y = 0; y < height; ++y)
        {
            Vec4b* row = overlay.ptr<Vec4b>(y);
            for (int x = 0; x < width; ++x)
                row[x] = Vec4b(240, 20, 220, saturate_cast<uchar>(x));
        }

        Mat dst;
        alphaComposite(overlay, background, dst);

        for (int y = 0; y < height; ++y)
        {
            const Vec4b* ov = overlay.ptr<Vec4b>(y);
            const Vec3b* bg = background.ptr<Vec3b>(y);
            const Vec3b* d = dst.ptr<Vec3b>(y);
            for (int x = 0; x < width; ++x)
                for (int c = 0; c < 3; ++c)
                {
                    // Rounding in two steps can shift the sum up to 1 outside [lo, hi].
                    int lo = std::min(ov[x][c], bg[x][c]);
                    int hi = std::max(ov[x][c], bg[x][c]);
                    ASSERT_GE((int)d[x][c], lo - 1) << "x=" << x << " c=" << c;
                    ASSERT_LE((int)d[x][c], hi + 1) << "x=" << x << " c=" << c;
                }
        }
        // Ramp endpoints must match exactly -- no rounding slack at alpha=0 or 255.
        Vec3b firstPixel = dst.at<Vec3b>(0, 0);
        Vec3b bgPixel = background.at<Vec3b>(0, 0);
        for (int c = 0; c < 3; ++c)
            EXPECT_EQ((int)firstPixel[c], (int)bgPixel[c]);

        Vec3b lastPixel = dst.at<Vec3b>(0, width - 1);
        EXPECT_EQ((int)lastPixel[0], 240);
        EXPECT_EQ((int)lastPixel[1], 20);
        EXPECT_EQ((int)lastPixel[2], 220);
    }
}

// Cases where the operator collapses to one of its operands, so no rounding slack is allowed.
TEST(Imgproc_AlphaComposite, OperatorIdentities)
{
    Size size(63, 37);
    Mat overlay(size, CV_8UC4), background(size, CV_8UC3), dst;
    randu(overlay, 0, 256);
    randu(background, 0, 256);

    Mat opaqueOverlay = overlay.clone(), opaqueColor;
    std::vector<Mat> channels;
    split(opaqueOverlay, channels);
    channels[3].setTo(255);
    merge(channels, opaqueOverlay);
    cvtColor(opaqueOverlay, opaqueColor, COLOR_BGRA2BGR);

    alphaComposite(overlay, background, dst, ALPHA_COMPOSITE_DEST);
    EXPECT_EQ(0.0, cv::norm(dst, background, NORM_INF));

    alphaComposite(overlay, background, dst, ALPHA_COMPOSITE_CLEAR);
    EXPECT_EQ(0.0, cv::norm(dst, Mat::zeros(size, CV_8UC3), NORM_INF));

    // An opaque source covers an opaque destination for every "keep the source" operator.
    alphaComposite(opaqueOverlay, background, dst, ALPHA_COMPOSITE_SOURCE);
    EXPECT_EQ(0.0, cv::norm(dst, opaqueColor, NORM_INF));
    alphaComposite(opaqueOverlay, background, dst, ALPHA_COMPOSITE_IN);
    EXPECT_EQ(0.0, cv::norm(dst, opaqueColor, NORM_INF));
    alphaComposite(opaqueOverlay, background, dst, ALPHA_COMPOSITE_ATOP);
    EXPECT_EQ(0.0, cv::norm(dst, opaqueColor, NORM_INF));

    // ... and the complementary operators erase everything on an opaque destination.
    alphaComposite(opaqueOverlay, background, dst, ALPHA_COMPOSITE_OUT);
    EXPECT_EQ(0.0, cv::norm(dst, Mat::zeros(size, CV_8UC3), NORM_INF));
    alphaComposite(opaqueOverlay, background, dst, ALPHA_COMPOSITE_DEST_OUT);
    EXPECT_EQ(0.0, cv::norm(dst, Mat::zeros(size, CV_8UC3), NORM_INF));
    alphaComposite(opaqueOverlay, background, dst, ALPHA_COMPOSITE_XOR);
    EXPECT_EQ(0.0, cv::norm(dst, Mat::zeros(size, CV_8UC3), NORM_INF));

    // A transparent source leaves an opaque destination alone.
    Mat transparentOverlay(size, CV_8UC4, Scalar(200, 100, 50, 0));
    alphaComposite(transparentOverlay, background, dst, ALPHA_COMPOSITE_OVER);
    EXPECT_EQ(0.0, cv::norm(dst, background, NORM_INF));
    alphaComposite(transparentOverlay, background, dst, ALPHA_COMPOSITE_DEST_OVER);
    EXPECT_EQ(0.0, cv::norm(dst, background, NORM_INF));

    // PLUS onto black is the premultiplied source.
    Mat black(size, CV_8UC3, Scalar::all(0)), overlayPremul, overlayPremulColor;
    cvtColor(overlay, overlayPremul, COLOR_RGBA2mRGBA);
    cvtColor(overlayPremul, overlayPremulColor, COLOR_BGRA2BGR);
    alphaComposite(overlay, black, dst, ALPHA_COMPOSITE_PLUS);
    EXPECT_EQ(0.0, cv::norm(dst, overlayPremulColor, NORM_INF));
}

// A 4-channel destination keeps the composited alpha, which a 3-channel one cannot.
TEST(Imgproc_AlphaComposite, CompositedAlphaOnFourChannelBackground)
{
    Size size(40, 24);
    Mat overlay(size, CV_8UC4, Scalar(200, 100, 50, 128));
    Mat background(size, CV_8UC4, Scalar(10, 20, 30, 64));
    Mat dst;

    alphaComposite(overlay, background, dst, ALPHA_COMPOSITE_OVER);
    // 128 + 64*(1 - 128/255) = 159.8
    EXPECT_NEAR(dst.at<Vec4b>(0, 0)[3], 160, 1);

    alphaComposite(overlay, background, dst, ALPHA_COMPOSITE_DEST_OUT);
    // 64*(1 - 128/255) = 31.9
    EXPECT_NEAR(dst.at<Vec4b>(0, 0)[3], 32, 1);

    alphaComposite(overlay, background, dst, ALPHA_COMPOSITE_IN);
    // 128 * 64/255 = 32.1
    EXPECT_NEAR(dst.at<Vec4b>(0, 0)[3], 32, 1);

    alphaComposite(overlay, background, dst, ALPHA_COMPOSITE_PLUS);
    EXPECT_EQ(dst.at<Vec4b>(0, 0)[3], 192);
}

TEST(Imgproc_AlphaComposite, InPlaceDestinationAliasingBackground)
{
    Size size(80, 60);
    Mat overlay(size, CV_8UC4);
    randu(overlay, 0, 256);

    for (size_t i = 0; i < sizeof(allOperators) / sizeof(allOperators[0]); ++i)
    {
        int op = allOperators[i];
        SCOPED_TRACE(cv::format("op=%d", op));

        Mat background3(size, CV_8UC3), expected3;
        randu(background3, 0, 256);
        Mat inPlace3 = background3.clone();
        alphaComposite(overlay, background3, expected3, op);
        alphaComposite(overlay, inPlace3, inPlace3, op); // dst aliases background
        EXPECT_EQ(0.0, cv::norm(inPlace3, expected3, NORM_INF));

        Mat background4(size, CV_8UC4), expected4;
        randu(background4, 0, 256);
        Mat inPlace4 = background4.clone();
        alphaComposite(overlay, background4, expected4, op);
        alphaComposite(overlay, inPlace4, inPlace4, op);
        EXPECT_EQ(0.0, cv::norm(inPlace4, expected4, NORM_INF));
    }
}

TEST(Imgproc_AlphaComposite, RejectsInvalidInputs)
{
    Mat overlay4(32, 32, CV_8UC4, Scalar::all(255));
    Mat overlay3(32, 32, CV_8UC3, Scalar::all(255));
    Mat background3(32, 32, CV_8UC3, Scalar::all(0));
    Mat background2(32, 32, CV_8UC2, Scalar::all(0));
    Mat backgroundSmall(16, 16, CV_8UC3, Scalar::all(0));
    Mat overlayFloat(32, 32, CV_32FC4, Scalar::all(1.f));
    Mat dst;

    EXPECT_THROW(alphaComposite(overlay3, background3, dst), cv::Exception);
    EXPECT_THROW(alphaComposite(overlay4, background2, dst), cv::Exception);
    EXPECT_THROW(alphaComposite(overlay4, backgroundSmall, dst), cv::Exception);
    EXPECT_THROW(alphaComposite(overlayFloat, background3, dst), cv::Exception);
    EXPECT_THROW(alphaComposite(overlay4, background3, dst, -1), cv::Exception);
    EXPECT_THROW(alphaComposite(overlay4, background3, dst, ALPHA_COMPOSITE_PLUS + 1), cv::Exception);
}

}} // namespace opencv_test / anonymous namespace
