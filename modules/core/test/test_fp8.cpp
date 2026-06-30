// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

// The two FP8 depths and their wrapper types share one set of expectations.
// Values chosen to be exactly representable (so round-trips are bit-exact) plus
// the special/overflow cases that distinguish the formats.

TEST(Core_FP8, type_basics)
{
    const int depths[] = { CV_8F_E4M3FN, CV_8F_E4M3FNUZ };
    for (int d : depths)
    {
        EXPECT_EQ(CV_ELEM_SIZE1(d), 1) << "depth " << d;
        Mat m(3, 4, CV_MAKETYPE(d, 1));
        EXPECT_EQ(m.depth(), d);
        EXPECT_EQ(m.channels(), 1);
        EXPECT_EQ(m.elemSize(), (size_t)1);
        EXPECT_EQ(m.elemSize1(), (size_t)1);
        EXPECT_EQ(m.total(), (size_t)12);
        // depthToString should not return null for a registered depth
        EXPECT_NE(cv::depthToString(d), (const char*)NULL);
    }
    Mat c3(2, 2, CV_8FC(3));
    EXPECT_EQ(c3.channels(), 3);
    EXPECT_EQ(c3.elemSize(), (size_t)3);
}

TEST(Core_FP8, scalar_roundtrip_exact)
{
    // {0, .5, 1, 1.5, 2, 3, 4, 6} and negatives are exact in every FP8 format here.
    const float exact[] = { 0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -2.5f, -0.75f };
    for (float v : exact)
    {
        EXPECT_EQ((float)cv::fp8_t(v),   v) << v;
        EXPECT_EQ((float)cv::fp8a_t(v), v) << v;
    }
    // round-to-nearest-even onto the grid
    EXPECT_EQ((float)cv::fp8_t(1.234f), 1.25f);   // 3 mantissa bits
}

TEST(Core_FP8, format_specific_limits)
{
    // max finite values
    EXPECT_EQ((float)cv::fp8_t(448.f),   448.f);
    EXPECT_EQ((float)cv::fp8a_t(240.f), 240.f);

    // overflow: these formats have no inf -> overflow to NaN
    EXPECT_TRUE(cvIsNaN((float)cv::fp8_t(1e6f)));
    EXPECT_TRUE(cvIsNaN((float)cv::fp8a_t(1e6f)));
    // 448 exceeds the FNUZ E4M3 range (max 240) -> NaN
    EXPECT_TRUE(cvIsNaN((float)cv::fp8a_t(448.f)));

    // NaN propagates
    EXPECT_TRUE(cvIsNaN((float)cv::fp8_t(std::numeric_limits<float>::quiet_NaN())));

    // smallest E4M3FN subnormal is 2^-9
    EXPECT_EQ((float)cv::fp8_t(0.001953125f), 0.001953125f);
}

TEST(Core_FP8, mat_convert_roundtrip)
{
    float vals[] = { 0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -1.f, -4.f };
    Mat f(1, 10, CV_32F, vals);
    const int depths[] = { CV_8F_E4M3FN, CV_8F_E4M3FNUZ };
    for (int d : depths)
    {
        Mat q, back;
        f.convertTo(q, d);
        EXPECT_EQ(q.depth(), d);
        EXPECT_EQ(q.elemSize(), (size_t)1);
        q.convertTo(back, CV_32F);
        ASSERT_EQ(back.type(), CV_32FC1);
        for (int i = 0; i < 10; i++)
            EXPECT_EQ(back.at<float>(i), vals[i]) << "depth " << d << " idx " << i;
    }
}

TEST(Core_FP8, convert_from_and_to_other_types)
{
    // f16 -> fp8 -> f32 (f16 source is lossless into the conversion)
    Mat f32(1, 5, CV_32F);
    float v[] = { 0.5f, 1.f, 2.f, 4.f, -3.f };
    memcpy(f32.data, v, sizeof(v));
    Mat f16; f32.convertTo(f16, CV_16F);
    Mat q; f16.convertTo(q, CV_8F_E4M3FN);
    Mat back; q.convertTo(back, CV_32F);
    for (int i = 0; i < 5; i++)
        EXPECT_EQ(back.at<float>(i), v[i]);

    // fp8 -> int (saturate_cast rounds to nearest)
    Mat qi; f32.convertTo(qi, CV_8F_E4M3FN);
    Mat i32; qi.convertTo(i32, CV_32S);
    EXPECT_EQ(i32.at<int>(0), 0);   // 0.5 -> 0 (round to even)
    EXPECT_EQ(i32.at<int>(1), 1);
    EXPECT_EQ(i32.at<int>(2), 2);
    EXPECT_EQ(i32.at<int>(3), 4);
    EXPECT_EQ(i32.at<int>(4), -3);
}

TEST(Core_FP8, cross_fp8_conversion)
{
    float v[] = { 0.5f, 1.5f, 6.f, 100.f, -2.f };
    Mat f(1, 5, CV_32F, v);
    Mat e4m3, e4m3u, back;
    f.convertTo(e4m3, CV_8F_E4M3FN);
    e4m3.convertTo(e4m3u, CV_8F_E4M3FNUZ);   // FP8 -> FP8
    e4m3u.convertTo(back, CV_32F);
    // values <=6 are representable in both grids -> preserved exactly
    EXPECT_EQ(back.at<float>(0), 0.5f);
    EXPECT_EQ(back.at<float>(1), 1.5f);
    EXPECT_EQ(back.at<float>(2), 6.f);
    EXPECT_EQ(back.at<float>(4), -2.f);
}

TEST(Core_FP8, convert_scale)
{
    Mat f = (Mat_<float>(1, 4) << 1.f, 2.f, 3.f, 4.f);
    Mat q, back;
    f.convertTo(q, CV_8F_E4M3FN, 2.0, 1.0);   // 2x+1 -> {3,5,7,9}
    q.convertTo(back, CV_32F);
    EXPECT_EQ(back.at<float>(0), 3.f);   // 1.5*2,  exact
    EXPECT_EQ(back.at<float>(1), 5.f);   // 1.25*4, exact
    EXPECT_EQ(back.at<float>(2), 7.f);   // 1.75*4, exact
    EXPECT_EQ(back.at<float>(3), 9.f);   // 9 = 1.125*8 is exact in E4M3 (3 mantissa bits)
}

TEST(Core_FP8, set_scalar)
{
    Mat m(3, 3, CV_8F_E4M3FN);
    m.setTo(Scalar(2.5));
    Mat back; m.convertTo(back, CV_32F);
    for (int i = 0; i < 9; i++)
        EXPECT_EQ(back.at<float>(i), 2.5f);

    Mat z = Mat::zeros(2, 2, CV_8F_E4M3FNUZ);
    Mat zf; z.convertTo(zf, CV_32F);
    EXPECT_EQ(countNonZero(zf), 0);
}

}} // namespace
