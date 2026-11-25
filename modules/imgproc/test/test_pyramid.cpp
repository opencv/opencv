// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef testing::TestWithParam<Size> Imgproc_PyrUp_empty;

TEST_P(Imgproc_PyrUp_empty, no_crash)
{
    Mat src(GetParam(), CV_8UC1), dst;

    ASSERT_NO_THROW(pyrUp(src, dst));
    EXPECT_TRUE(dst.empty());
    // a non-empty destination size cannot be filled from an empty source
    EXPECT_THROW(pyrUp(src, dst, Size(1, 10)), cv::Exception);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Imgproc_PyrUp_empty, testing::Values(
    Size(0, 0), Size(0, 5), Size(5, 0)
));

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

static Mat pyrDownReference(const Mat& src, Size dsize, int borderType)
{
    static const int k[5] = { 1, 4, 6, 4, 1 };
    const int cn = src.channels();
    const int depth = src.depth();
    const int borderTypeNI = borderType & ~BORDER_ISOLATED;

    Mat padded, padded64;
    copyMakeBorder(src, padded, 2, 4, 2, 4, borderTypeNI);
    padded.convertTo(padded64, CV_64F);

    Mat horiz(padded64.rows, dsize.width, CV_64FC(cn), Scalar::all(0));
    for (int y = 0; y < padded64.rows; ++y)
    {
        const double* srow = padded64.ptr<double>(y);
        double* hrow = horiz.ptr<double>(y);
        for (int dx = 0; dx < dsize.width; ++dx)
        {
            const int sx_base = dx * 2;
            for (int c = 0; c < cn; ++c)
            {
                double sum = 0.0;
                for (int t = 0; t < 5; ++t)
                    sum += k[t] * srow[(sx_base + t) * cn + c];
                hrow[dx * cn + c] = sum;
            }
        }
    }

    Mat dst(dsize, src.type());
    for (int dy = 0; dy < dsize.height; ++dy)
    {
        const int sy_base = dy * 2;
        const int cols_cn = dsize.width * cn;
        for (int x = 0; x < cols_cn; ++x)
        {
            double sum = 0.0;
            for (int t = 0; t < 5; ++t)
                sum += k[t] * horiz.ptr<double>(sy_base + t)[x];

            if (depth == CV_8U)
                dst.ptr<uchar>(dy)[x] = saturate_cast<uchar>((cvRound(sum) + 128) >> 8);
            else if (depth == CV_16U)
                dst.ptr<ushort>(dy)[x] = saturate_cast<ushort>((cvRound(sum) + 128) >> 8);
            else if (depth == CV_16S)
                dst.ptr<short>(dy)[x] = saturate_cast<short>((cvRound(sum) + 128) >> 8);
            else if (depth == CV_32F)
                dst.ptr<float>(dy)[x] = static_cast<float>(sum / 256.0);
            else
                dst.ptr<double>(dy)[x] = sum / 256.0;
        }
    }
    return dst;
}

TEST(Imgproc_PyrDown, right_border_tail_32f)
{
    Mat src = (Mat_<float>(1, 4) << 0.f, 1.f, 2.f, 3.f);
    Mat dst;
    pyrDown(src, dst, Size(3, 1), BORDER_REPLICATE);

    Mat expected = (Mat_<float>(1, 3) << 0.375f, 1.9375f, 2.9375f);
    EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), 1e-6);
}

TEST(Imgproc_PyrDown, right_border_even_half_plus_one)
{
    const Size src_sizes[] = { Size(4, 4), Size(10, 10), Size(14, 14), Size(16, 16) };
    const int types[] = { CV_8UC1, CV_8UC3, CV_16UC3, CV_32FC1, CV_32FC3, CV_64FC1 };
    const int borders[] = { BORDER_REFLECT_101, BORDER_REPLICATE, BORDER_REFLECT };

    RNG rng(2025);
    for (size_t si = 0; si < sizeof(src_sizes) / sizeof(src_sizes[0]); ++si)
    {
        const Size ssize = src_sizes[si];
        const Size dsize(ssize.width / 2 + 1, ssize.height / 2 + 1);
        for (size_t ti = 0; ti < sizeof(types) / sizeof(types[0]); ++ti)
        {
            Mat src(ssize, types[ti]);
            rng.fill(src, RNG::UNIFORM, Scalar::all(0), Scalar::all(200));
            for (size_t bi = 0; bi < sizeof(borders) / sizeof(borders[0]); ++bi)
            {
                Mat dst;
                pyrDown(src, dst, dsize, borders[bi]);
                Mat expected = pyrDownReference(src, dsize, borders[bi]);
                const double tol = (CV_MAT_DEPTH(types[ti]) >= CV_32F) ? 1e-4 : 0.0;
                EXPECT_LE(cvtest::norm(dst, expected, NORM_INF), tol)
                    << "ssize=" << ssize << " type=" << types[ti] << " border=" << borders[bi];
            }
        }
    }
}

TEST(Imgproc_PyrDown, border_isolated)
{
    Mat parent(24, 24, CV_16UC3, Scalar::all(100));
    const Rect roi(2, 3, 14, 14);
    RNG rng(2026);
    rng.fill(parent(roi), RNG::UNIFORM, Scalar::all(0), Scalar::all(50));

    Mat isolated_clone = parent(roi).clone();
    const Size dsize(isolated_clone.cols / 2 + 1, isolated_clone.rows / 2 + 1);

    Mat dst_clone, dst_submat;
    pyrDown(isolated_clone, dst_clone, dsize, BORDER_DEFAULT);
    pyrDown(parent(roi), dst_submat, dsize, BORDER_DEFAULT | BORDER_ISOLATED);
    EXPECT_EQ(0.0, cvtest::norm(dst_submat, dst_clone, NORM_INF));

    UMat u_parent, u_dst_clone, u_dst_submat;
    parent.copyTo(u_parent);
    pyrDown(isolated_clone, u_dst_clone, dsize, BORDER_DEFAULT);
    pyrDown(u_parent(roi), u_dst_submat, dsize, BORDER_DEFAULT | BORDER_ISOLATED);
    EXPECT_EQ(0.0, cvtest::norm(u_dst_submat.getMat(ACCESS_READ), u_dst_clone.getMat(ACCESS_READ), NORM_INF));
    EXPECT_LE(cvtest::norm(u_dst_clone.getMat(ACCESS_READ), dst_clone, NORM_INF), 1.0);

    Mat tmp;
    EXPECT_THROW(pyrDown(isolated_clone, tmp, dsize, BORDER_CONSTANT | BORDER_ISOLATED), cv::Exception);
}

}
}
