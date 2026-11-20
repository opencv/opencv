// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef testing::TestWithParam<int> Pyramid_HalfFloat;

// widened = the half input widened back to float, which is lossless
static float pyrDownRef(const Mat& widened, int x, int y, int c, int cn)
{
    static const float k[5] = { 1.f, 4.f, 6.f, 4.f, 1.f };
    float s = 0.f;
    for (int i = -2; i <= 2; i++)
        for (int j = -2; j <= 2; j++)
            s += k[i + 2] * k[j + 2] * widened.ptr<float>(2 * y + i)[(2 * x + j) * cn + c];
    return s / 256.f;
}

// even output index takes [1,6,1] of the source, odd takes [4,4]; /64 overall
static float pyrUpRef(const Mat& widened, int X, int Y, int c, int cn)
{
    int sy[3], sx[3], wy[3], wx[3], ny, nx;
    if (Y % 2 == 0) { ny = 3; sy[0] = Y/2 - 1; sy[1] = Y/2; sy[2] = Y/2 + 1;
                              wy[0] = 1; wy[1] = 6; wy[2] = 1; }
    else            { ny = 2; sy[0] = Y/2;     sy[1] = Y/2 + 1;
                              wy[0] = 4; wy[1] = 4; }
    if (X % 2 == 0) { nx = 3; sx[0] = X/2 - 1; sx[1] = X/2; sx[2] = X/2 + 1;
                              wx[0] = 1; wx[1] = 6; wx[2] = 1; }
    else            { nx = 2; sx[0] = X/2;     sx[1] = X/2 + 1;
                              wx[0] = 4; wx[1] = 4; }

    float s = 0.f;
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            s += (float)(wy[i] * wx[j]) * widened.ptr<float>(sy[i])[sx[j] * cn + c];
    return s / 64.f;
}

TEST_P(Pyramid_HalfFloat, pyrDown_vs_reference)
{
    const int depth = GetParam();
    const double tol = depth == CV_16F ? 1e-3 : 8e-3;

    const Size size(64, 48);
    cv::RNG rng(53);

    for (int cn = 1; cn <= 4; cn++)
    {
        Mat src32(size, CV_MAKETYPE(CV_32F, cn)), src, widened, dst, dst32;
        rng.fill(src32, cv::RNG::UNIFORM, Scalar::all(0), Scalar::all(1));
        src32.convertTo(src, CV_MAKETYPE(depth, cn));
        src.convertTo(widened, CV_MAKETYPE(CV_32F, cn));

        SCOPED_TRACE(cv::format("depth=%d cn=%d", depth, cn));
        ASSERT_NO_THROW(cv::pyrDown(src, dst));
        ASSERT_EQ(depth, dst.depth());
        ASSERT_EQ(Size(size.width / 2, size.height / 2), dst.size());
        dst.convertTo(dst32, CV_MAKETYPE(CV_32F, cn));

        for (int y = 2; y < dst.rows - 2; y++)
            for (int x = 2; x < dst.cols - 2; x++)
                for (int c = 0; c < cn; c++)
                {
                    float want = pyrDownRef(widened, x, y, c, cn);
                    float got = dst32.ptr<float>(y)[x * cn + c];
                    ASSERT_NEAR(want, got, tol) << "at (" << x << "," << y << ") ch " << c;
                }
    }
}

TEST_P(Pyramid_HalfFloat, pyrUp_vs_reference)
{
    const int depth = GetParam();
    const double tol = depth == CV_16F ? 1e-3 : 8e-3;

    const Size size(32, 24);
    cv::RNG rng(59);

    for (int cn = 1; cn <= 4; cn++)
    {
        Mat src32(size, CV_MAKETYPE(CV_32F, cn)), src, widened, dst, dst32;
        rng.fill(src32, cv::RNG::UNIFORM, Scalar::all(0), Scalar::all(1));
        src32.convertTo(src, CV_MAKETYPE(depth, cn));
        src.convertTo(widened, CV_MAKETYPE(CV_32F, cn));

        SCOPED_TRACE(cv::format("depth=%d cn=%d", depth, cn));
        ASSERT_NO_THROW(cv::pyrUp(src, dst));
        ASSERT_EQ(depth, dst.depth());
        ASSERT_EQ(Size(size.width * 2, size.height * 2), dst.size());
        dst.convertTo(dst32, CV_MAKETYPE(CV_32F, cn));

        for (int y = 2; y < dst.rows - 3; y++)
            for (int x = 2; x < dst.cols - 3; x++)
                for (int c = 0; c < cn; c++)
                {
                    float want = pyrUpRef(widened, x, y, c, cn);
                    float got = dst32.ptr<float>(y)[x * cn + c];
                    ASSERT_NEAR(want, got, tol) << "at (" << x << "," << y << ") ch " << c;
                }
    }
}

// weights sum to 1, so a constant survives both directions exactly
TEST_P(Pyramid_HalfFloat, constant_is_exact)
{
    const int depth = GetParam();

    for (int cn = 1; cn <= 4; cn++)
    {
        SCOPED_TRACE(cv::format("depth=%d cn=%d", depth, cn));
        Mat src(Size(64, 48), CV_MAKETYPE(depth, cn));
        Mat c32(Size(64, 48), CV_MAKETYPE(CV_32F, cn), Scalar::all(0.5));
        c32.convertTo(src, CV_MAKETYPE(depth, cn));

        Mat down, up, down32, up32;
        ASSERT_NO_THROW(cv::pyrDown(src, down));
        ASSERT_NO_THROW(cv::pyrUp(src, up));
        ASSERT_EQ(depth, down.depth());
        ASSERT_EQ(depth, up.depth());
        ASSERT_EQ(Size(128, 96), up.size());

        down.convertTo(down32, CV_MAKETYPE(CV_32F, cn));
        up.convertTo(up32, CV_MAKETYPE(CV_32F, cn));

        Mat wantDown(down.size(), CV_MAKETYPE(CV_32F, cn), Scalar::all(0.5));
        Mat wantUp(up.size(), CV_MAKETYPE(CV_32F, cn), Scalar::all(0.5));
        EXPECT_EQ(0, cvtest::norm(down32, wantDown, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(up32, wantUp, NORM_INF));
    }
}

TEST_P(Pyramid_HalfFloat, buildPyramid_levels)
{
    const int depth = GetParam();
    Mat src(Size(64, 64), CV_MAKETYPE(depth, 1));
    Mat c32(Size(64, 64), CV_MAKETYPE(CV_32F, 1), Scalar::all(0.25));
    c32.convertTo(src, depth);

    std::vector<Mat> pyr;
    ASSERT_NO_THROW(cv::buildPyramid(src, pyr, 3));
    ASSERT_EQ(4u, pyr.size());
    for (size_t i = 0; i < pyr.size(); i++)
    {
        SCOPED_TRACE(cv::format("level %d", (int)i));
        EXPECT_EQ(depth, pyr[i].depth());
        EXPECT_EQ(Size(64 >> i, 64 >> i), pyr[i].size());
    }
}

INSTANTIATE_TEST_CASE_P(Imgproc, Pyramid_HalfFloat, testing::Values(CV_16F, CV_16BF));

TEST(Imgproc_PyrUp, pyrUp_regression_22184)
{
    Mat src(100,100,CV_16UC3,Scalar(255,255,255));
    Mat dst(100 * 2 + 1, 100 * 2 + 1, CV_16UC3, Scalar(0,0,0));
    pyrUp(src, dst, Size(dst.cols, dst.rows));
    double min_val = 0;
    minMaxLoc(dst, &min_val);
    ASSERT_GT(cvRound(min_val), 0);
}

TEST(Imgproc_PyrUp, pyrUp_regression_22194)
{
    Mat src(13, 13,CV_16UC3,Scalar(0,0,0));
    {
        int swidth = src.cols;
        int sheight = src.rows;
        int cn = src.channels();
        int count = 0;
        for (int y = 0; y < sheight; y++)
        {
            ushort *src_c = src.ptr<ushort>(y);
            for (int x = 0; x < swidth * cn; x++)
            {
                src_c[x] = (count++) % 10;
            }
        }
    }
    Mat dst(src.cols * 2 - 1, src.rows * 2 - 1, CV_16UC3, Scalar(0,0,0));
    pyrUp(src, dst, Size(dst.cols, dst.rows));

    {
        ushort *dst_c = dst.ptr<ushort>(dst.rows - 1);
        ASSERT_EQ(dst_c[0], 6);
        ASSERT_EQ(dst_c[1], 6);
        ASSERT_EQ(dst_c[2], 1);
    }
}

}
}
