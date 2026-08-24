// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

// Independent reference for the over operator -- avoids reusing alphaComposite's own rounding.
static void referenceOver(const Mat& overlay, const Mat& background, Mat& dst)
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
            for (int c = 0; c < 3; ++c)
            {
                double blended = ov[c] * as + bg[c] * (1.0 - as);
                d[c] = saturate_cast<uchar>(cvRound(blended));
            }
            if (bgChannels == 4)
            {
                double ad = bg[3] / 255.0;
                double ao = as + ad * (1.0 - as);
                d[3] = saturate_cast<uchar>(cvRound(ao * 255.0));
            }
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

typedef testing::TestWithParam<std::tuple<int, bool>> Imgproc_AlphaComposite_Correctness;

TEST_P(Imgproc_AlphaComposite_Correctness, MatchesReference)
{
    int bgChannels = std::get<0>(GetParam());
    bool premultiplied = std::get<1>(GetParam());

    Size size(157, 83);
    Mat overlay(size, CV_8UC4), background(size, CV_8UC(bgChannels)), dst, ref;
    randu(overlay, 0, 256);
    randu(background, 0, 256);

    Mat overlayInput = overlay;
    if (premultiplied)
        cvtColor(overlay, overlayInput, COLOR_RGBA2mRGBA);

    alphaComposite(overlayInput, background, dst, premultiplied);
    referenceOver(overlay, background, ref);

    ASSERT_EQ(dst.type(), background.type());
    ASSERT_EQ(dst.size(), background.size());
    EXPECT_LE(maxAbsDiff(dst, ref), kRoundingTolerance);
}

INSTANTIATE_TEST_CASE_P(Imgproc, Imgproc_AlphaComposite_Correctness,
                        testing::Combine(testing::Values(3, 4), testing::Bool()));

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

TEST(Imgproc_AlphaComposite, InPlaceDestinationAliasingBackground)
{
    Size size(80, 60);
    Mat overlay(size, CV_8UC4);
    randu(overlay, 0, 256);

    Mat background3(size, CV_8UC3), expected3;
    randu(background3, 0, 256);
    alphaComposite(overlay, background3, expected3);
    alphaComposite(overlay, background3, background3); // dst aliases background
    EXPECT_EQ(0.0, cv::norm(background3, expected3, NORM_INF));

    Mat background4(size, CV_8UC4), expected4;
    randu(background4, 0, 256);
    alphaComposite(overlay, background4, expected4);
    alphaComposite(overlay, background4, background4);
    EXPECT_EQ(0.0, cv::norm(background4, expected4, NORM_INF));
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
}

}} // namespace opencv_test / anonymous namespace
