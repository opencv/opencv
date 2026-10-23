// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

struct HalfColorCase
{
    int code, scn, dcn;
};

std::vector<HalfColorCase> halfColorCases()
{
    std::vector<HalfColorCase> cases;
    for (int scn : {3, 4})
    {
        for (int code : {COLOR_RGB2GRAY, COLOR_BGR2GRAY})
            cases.push_back({code, scn, 1});
        for (int code : {COLOR_RGB2HSV, COLOR_BGR2HSV, COLOR_RGB2HSV_FULL, COLOR_BGR2HSV_FULL,
                         COLOR_RGB2HLS, COLOR_BGR2HLS, COLOR_RGB2HLS_FULL, COLOR_BGR2HLS_FULL})
            cases.push_back({code, scn, 3});
        for (int dcn : {3, 4})
            for (int code : {COLOR_RGB2BGR, COLOR_RGB2RGBA})
                cases.push_back({code, scn, dcn});
    }
    for (int dcn : {3, 4})
    {
        cases.push_back({COLOR_GRAY2RGB, 1, dcn});
        for (int code : {COLOR_HSV2RGB, COLOR_HSV2BGR, COLOR_HSV2RGB_FULL, COLOR_HSV2BGR_FULL,
                         COLOR_HLS2RGB, COLOR_HLS2BGR, COLOR_HLS2RGB_FULL, COLOR_HLS2BGR_FULL})
            cases.push_back({code, 3, dcn});
    }
    return cases;
}

void expectHalfColorNear(const Mat& actual, const Mat& expected)
{
    ASSERT_EQ(expected.type(), actual.type());
    ASSERT_EQ(expected.size(), actual.size());
    for (int y = 0; y < actual.rows; ++y)
    {
        const hfloat* a = actual.ptr<hfloat>(y);
        const hfloat* e = expected.ptr<hfloat>(y);
        for (int x = 0; x < actual.cols * actual.channels(); ++x)
        {
            float value = e[x];
            ushort bits;
            memcpy(&bits, e + x, sizeof(bits));
            float ulp = std::abs(float(hfloatFromBits((bits & 0x7fff) + 1)) - std::abs(value));
            ASSERT_LE(std::abs(float(a[x]) - value), std::max(ulp, 1e-6f)) << y << ", " << x;
        }
    }
}

using HalfColorParams = std::tuple<int, int>;
typedef testing::TestWithParam<HalfColorParams> Imgproc_HalfColor;

TEST_P(Imgproc_HalfColor, accuracy_and_storage)
{
    const int width = std::get<0>(GetParam()), layout = std::get<1>(GetParam());
    RNG rng(0x30361);
    for (const HalfColorCase& c : halfColorCases())
    {
        SCOPED_TRACE(format("code=%d scn=%d dcn=%d", c.code, c.scn, c.dcn));
        Mat parent(7, width + 4, CV_MAKETYPE(CV_32F, c.scn));
        rng.fill(parent, RNG::UNIFORM, 0.f, 1.f);
        if (c.code == COLOR_HSV2RGB || c.code == COLOR_HSV2BGR ||
            c.code == COLOR_HSV2RGB_FULL || c.code == COLOR_HSV2BGR_FULL ||
            c.code == COLOR_HLS2RGB || c.code == COLOR_HLS2BGR ||
            c.code == COLOR_HLS2RGB_FULL || c.code == COLOR_HLS2BGR_FULL)
        {
            for (int y = 0; y < parent.rows; ++y)
                for (int x = 0; x < parent.cols; ++x)
                    parent.ptr<float>(y)[3*x] *= 360.f;
        }
        Mat halfParent;
        parent.convertTo(halfParent, CV_16F);
        Rect roi(1, 1, width, 5);
        Mat src = layout ? halfParent(roi) : halfParent(roi).clone();
        Mat original = halfParent.clone(), represented, expectedFloat, expected;
        src.convertTo(represented, CV_32F);
        cvtColor(represented, expectedFloat, c.code, c.dcn);
        expectedFloat.convertTo(expected, CV_16F);

        Mat destinationParent(7, width + 4, CV_MAKETYPE(CV_16F, c.dcn), Scalar::all(-7));
        Mat dst = layout == 1 ? destinationParent(roi) : Mat();
        uchar* storage = dst.data;
        if (layout == 2)
        {
            dst = src;
            cvtColor(dst, dst, c.code, c.dcn);
            if (c.scn == c.dcn)
                EXPECT_EQ(src.data, dst.data);
        }
        else
        {
            cvtColor(src, dst, c.code, c.dcn);
            if (layout == 1)
                EXPECT_EQ(storage, dst.data);
        }
        expectHalfColorNear(dst, expected);
        if (layout != 2 || c.scn != c.dcn)
        {
            EXPECT_EQ(0, cv::norm(src, original(roi), NORM_INF));
            EXPECT_EQ(0, cv::norm(halfParent, original, NORM_INF));
        }
        else
        {
            Mat mask(halfParent.size(), CV_8U, Scalar::all(255));
            mask(roi).setTo(0);
            EXPECT_EQ(0, cv::norm(halfParent, original, NORM_INF, mask));
        }
        if (layout == 1)
        {
            Mat mask(destinationParent.size(), CV_8U, Scalar::all(255));
            mask(roi).setTo(0);
            Mat guard(destinationParent.size(), destinationParent.type(), Scalar::all(-7));
            EXPECT_EQ(0, cv::norm(destinationParent, guard, NORM_INF, mask));
        }
    }
}

INSTANTIATE_TEST_CASE_P(Imgproc, Imgproc_HalfColor,
    testing::Combine(testing::Values(1, 3, 7, 16, 17, 33, 257), testing::Values(0, 1, 2)));

TEST(Imgproc_HalfColor, primaries_and_hue_sectors)
{
    Mat rgb = Mat_<Vec3f>({1, 8}, {Vec3f(0, 0, 0), Vec3f(1, 1, 1), Vec3f(1, 0, 0),
        Vec3f(1, 1, 0), Vec3f(0, 1, 0), Vec3f(0, 1, 1), Vec3f(0, 0, 1), Vec3f(1, 0, 1)});
    rgb.convertTo(rgb, CV_16F);
    for (int code : {COLOR_RGB2HSV, COLOR_RGB2HLS})
    {
        Mat converted, restored, reference;
        cvtColor(rgb, converted, code);
        converted.convertTo(reference, CV_32F);
        for (int x = 2; x < 8; ++x)
            EXPECT_FLOAT_EQ(float((x - 2) * 60), reference.at<Vec3f>(0, x)[0]);
        EXPECT_FLOAT_EQ(0.f, reference.at<Vec3f>(0, 0)[0]);
        EXPECT_FLOAT_EQ(0.f, reference.at<Vec3f>(0, 1)[0]);
        cvtColor(converted, restored, code == COLOR_RGB2HSV ? COLOR_HSV2RGB : COLOR_HLS2RGB);
        EXPECT_EQ(0, cv::norm(rgb, restored, NORM_INF));
    }
    const Vec3f colors[] = {Vec3f(1, 0, 0), Vec3f(1, 1, 0), Vec3f(0, 1, 0),
        Vec3f(0, 1, 1), Vec3f(0, 0, 1), Vec3f(1, 0, 1), Vec3f(1, 0, 0)};
    for (int code : {COLOR_HSV2RGB, COLOR_HLS2RGB})
    {
        Mat hue(1, 7, CV_32FC3);
        for (int x = 0; x < hue.cols; ++x)
            hue.at<Vec3f>(0, x) = Vec3f(float(x * 60), code == COLOR_HSV2RGB ? 1.f : 0.5f, 1.f);
        hue.convertTo(hue, CV_16F);
        Mat restored;
        cvtColor(hue, restored, code);
        restored.convertTo(restored, CV_32F);
        for (int x = 0; x < hue.cols; ++x)
            for (int c = 0; c < 3; ++c)
                EXPECT_NEAR(colors[x][c], restored.at<Vec3f>(0, x)[c], 1e-6f);
    }
    Mat gray, gray32;
    cvtColor(rgb, gray, COLOR_RGB2GRAY);
    gray.convertTo(gray32, CV_32F);
    EXPECT_FLOAT_EQ(float(hfloat(0.299f)), gray32.at<float>(0, 2));
    EXPECT_FLOAT_EQ(float(hfloat(0.587f)), gray32.at<float>(0, 4));
    EXPECT_FLOAT_EQ(float(hfloat(0.114f)), gray32.at<float>(0, 6));
}

TEST(Imgproc_HalfColor, channel_copy_all_encodings)
{
    Mat gray(1, 65536, CV_16F);
    for (int x = 0; x < gray.cols; ++x)
    {
        ushort bits = static_cast<ushort>(x);
        memcpy(gray.ptr<hfloat>() + x, &bits, sizeof(bits));
    }
    Mat rgba, bgra, restored;
    cvtColor(gray, rgba, COLOR_GRAY2RGBA);
    cvtColor(rgba, bgra, COLOR_RGBA2BGRA);
    cvtColor(bgra, restored, COLOR_BGRA2BGR);
    for (int x = 0; x < gray.cols; ++x)
    {
        for (int c = 0; c < 3; ++c)
            EXPECT_EQ(0, memcmp(gray.ptr<hfloat>() + x, restored.ptr<hfloat>() + 3*x + c, sizeof(hfloat)));
        EXPECT_EQ(1.f, float(rgba.ptr<hfloat>()[4*x + 3]));
    }
}

TEST(Imgproc_HalfColor, large_image_and_umat)
{
    RNG rng(0x30361);
    for (const HalfColorCase& c : halfColorCases())
    {
        SCOPED_TRACE(format("code=%d scn=%d dcn=%d", c.code, c.scn, c.dcn));
        Mat src(257, 513, CV_MAKETYPE(CV_32F, c.scn));
        rng.fill(src, RNG::UNIFORM, 0.f, 1.f);
        src.convertTo(src, CV_16F);
        Mat represented, referenceFloat, reference, dst;
        src.convertTo(represented, CV_32F);
        cvtColor(represented, referenceFloat, c.code, c.dcn);
        referenceFloat.convertTo(reference, CV_16F);
        cvtColor(src, dst, c.code, c.dcn);
        expectHalfColorNear(dst, reference);
        UMat input, output;
        src.copyTo(input);
        cvtColor(input, output, c.code, c.dcn);
        EXPECT_EQ(0, cv::norm(dst, output.getMat(ACCESS_READ), NORM_INF));
    }
}

TEST(Imgproc_HalfColor, gray_finite_boundaries)
{
    Mat src = Mat_<Vec3f>({5, 1}, {Vec3f(0, -0.f, 0), Vec3f(1.f / 16777216, 0, 0),
        Vec3f(0, 1.f / 16384, 0), Vec3f(65504, 65504, 65504), Vec3f(-1, 2, 0.5f)});
    src.convertTo(src, CV_16F);
    Mat represented;
    src.convertTo(represented, CV_32F);
    for (int code : {COLOR_RGB2GRAY, COLOR_BGR2GRAY})
    {
        Mat expected, actual;
        cvtColor(represented, expected, code);
        expected.convertTo(expected, CV_16F);
        cvtColor(src, actual, code);
        EXPECT_EQ(0, cv::norm(actual, expected, NORM_INF));
    }
}

}} // namespace
