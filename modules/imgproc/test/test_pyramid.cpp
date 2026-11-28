// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef testing::TestWithParam<int> Pyramid_HalfFloat;

// the FP32 run on the widened input is the same filter in float, so it is the
// reference for the whole image, borders and odd sizes included
TEST_P(Pyramid_HalfFloat, vs_fp32)
{
    const int depth = GetParam();
    const double tol = depth == CV_16F ? 1e-3 : 8e-3;
    const Size sizes[] = { Size(64, 48), Size(37, 23) };
    cv::RNG& rng = cv::theRNG();

    for (const Size& size : sizes)
        for (int cn = 1; cn <= 4; cn++)
        {
            SCOPED_TRACE(cv::format("depth=%d cn=%d size=%dx%d",
                                    depth, cn, size.width, size.height));
            Mat src32(size, CV_MAKETYPE(CV_32F, cn)), src, widened;
            rng.fill(src32, cv::RNG::UNIFORM, Scalar::all(0), Scalar::all(1));
            src32.convertTo(src, depth);
            src.convertTo(widened, CV_32F);
            const Size upSize(size.width*2 + size.width % 2, size.height*2 + size.height % 2);

            Mat down, up, down32, up32, refDown, refUp;
            ASSERT_NO_THROW(cv::pyrDown(src, down));
            ASSERT_NO_THROW(cv::pyrUp(src, up, upSize));
            ASSERT_EQ(depth, down.depth());
            ASSERT_EQ(depth, up.depth());
            cv::pyrDown(widened, refDown);
            cv::pyrUp(widened, refUp, upSize);
            down.convertTo(down32, CV_32F);
            up.convertTo(up32, CV_32F);
            EXPECT_LE(cvtest::norm(down32, refDown, NORM_INF), tol);
            EXPECT_LE(cvtest::norm(up32, refUp, NORM_INF), tol);
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
