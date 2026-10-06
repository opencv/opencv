// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

using namespace cv;

namespace opencv_test { namespace {

// Independent scalar reference of the documented formula (no library calls).
template<typename T>
static void refDepthTo3d(const Mat& depth, const Matx<T, 3, 3>& K, Mat& points3d)
{
    const T inv_fx = T(1) / K(0, 0);
    const T inv_fy = T(1) / K(1, 1);
    const T ox = K(0, 2);
    const T oy = K(1, 2);

    points3d.create(depth.size(), CV_MAKETYPE(DataType<T>::depth, 4));
    for (int y = 0; y < depth.rows; ++y)
    {
        Vec<T, 4>* point = points3d.ptr<Vec<T, 4> >(y);
        const T* z_row = depth.ptr<T>(y);
        const T y_val = (y - oy) * inv_fy;
        for (int x = 0; x < depth.cols; ++x)
        {
            T z = z_row[x];
            point[x][0] = ((x - ox) * inv_fx) * z;
            point[x][1] = y_val * z;
            point[x][2] = z;
            point[x][3] = 0;
        }
    }
}

// Bit-exact compare; NaN must match NaN position-wise, rest bit-for-bit.
template<typename T>
static void expectExact(const Mat& a, const Mat& b)
{
    ASSERT_EQ(a.type(), b.type());
    ASSERT_EQ(a.size(), b.size());
    for (int y = 0; y < a.rows; ++y)
    {
        const Vec<T, 4>* pa = a.ptr<Vec<T, 4> >(y);
        const Vec<T, 4>* pb = b.ptr<Vec<T, 4> >(y);
        for (int x = 0; x < a.cols; ++x)
        {
            for (int c = 0; c < 4; ++c)
            {
                T va = pa[x][c], vb = pb[x][c];
                if (std::isnan((double)va) || std::isnan((double)vb))
                {
                    EXPECT_TRUE(std::isnan((double)va) && std::isnan((double)vb))
                        << "NaN mismatch at (" << y << "," << x << ") c=" << c;
                }
                else
                {
                    EXPECT_EQ(0, memcmp(&va, &vb, sizeof(T)))
                        << "value mismatch at (" << y << "," << x << ") c=" << c
                        << ": " << va << " vs " << vb;
                }
            }
        }
    }
}

template<typename T>
static void runNoMaskCase(Size sz, int extra_flags)
{
    Mat_<T> depth(sz);
    randu(depth, T(0.1), T(10.0));

    if (extra_flags & 1) // zeros
        for (int i = 0; i < depth.rows * depth.cols; i += 7) depth(i / depth.cols, i % depth.cols) = T(0);
    if (extra_flags & 2) // negatives
        for (int i = 0; i < depth.rows * depth.cols; i += 5) depth(i / depth.cols, i % depth.cols) = T(-3.5);
    if (extra_flags & 4) // NaN
        for (int i = 0; i < depth.rows * depth.cols; i += 11) depth(i / depth.cols, i % depth.cols) = std::numeric_limits<T>::quiet_NaN();

    Matx<T, 3, 3> K(T(525.0), T(0.0), T(319.5),
                    T(0.0), T(525.0), T(239.5),
                    T(0.0), T(0.0), T(1.0));

    Mat points3d, ref;
    depthTo3d(depth, K, points3d);
    refDepthTo3d<T>(depth, K, ref);
    expectExact<T>(points3d, ref);
}

typedef testing::TestWithParam<std::tuple<int, int> > RGBD_DepthTo3d;

TEST_P(RGBD_DepthTo3d, noMaskBitExact)
{
    Size sz(get<0>(GetParam()), get<1>(GetParam()));
    runNoMaskCase<float>(sz, 0);
    runNoMaskCase<double>(sz, 0);
}

TEST_P(RGBD_DepthTo3d, noMaskEdgeValues)
{
    Size sz(get<0>(GetParam()), get<1>(GetParam()));
    runNoMaskCase<float>(sz, 7);
    runNoMaskCase<double>(sz, 7);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, RGBD_DepthTo3d,
    testing::Values(
        testing::make_tuple(127, 61),   // odd tail
        testing::make_tuple(13, 3),     // smaller than one vector
        testing::make_tuple(64, 1),
        testing::make_tuple(320, 240)
    )
);

TEST(RGBD_DepthTo3dMisc, sixteenUSemantics)
{
    // 16U: rescaled by 1/1000, zeros become NaN (rescaleDepth contract).
    // Note: output depth follows K->CV_32F (implementation behavior).
    Mat depth(4, 4, CV_16UC1, Scalar(0));
    depth.at<ushort>(0, 0) = 1000;   // -> 1.0 m
    depth.at<ushort>(1, 1) = 2500;   // -> 2.5 m
    Matx33d K(500., 0., 1.5, 0., 500., 1.5, 0., 0., 1.);

    Mat points3d;
    depthTo3d(depth, K, points3d);
    EXPECT_EQ(CV_32FC4, points3d.type());

    EXPECT_FLOAT_EQ(1.0f, points3d.at<Vec4f>(0, 0)[2]);
    EXPECT_FLOAT_EQ((0 - 1.5f) / 500.f * 1.0f, points3d.at<Vec4f>(0, 0)[0]);
    EXPECT_TRUE(std::isnan(points3d.at<Vec4f>(2, 2)[2]));   // 0 depth -> NaN
    EXPECT_EQ(0.0f, points3d.at<Vec4f>(0, 0)[3]);
}

TEST(RGBD_DepthTo3dMisc, roiInput)
{
    // Non-continuous input: results must not depend on step/padding.
    Mat big(50, 60, CV_32FC1);
    randu(big, 0.5f, 8.f);
    Mat roi = big(Rect(7, 3, 31, 17));

    Matx33f K(400.f, 0.f, 15.5f, 0.f, 400.f, 8.5f, 0.f, 0.f, 1.f);

    Mat pts_roi, pts_ref;
    depthTo3d(roi, K, pts_roi);
    refDepthTo3d<float>(roi, K, pts_ref);
    expectExact<float>(pts_roi, pts_ref);
}

TEST(RGBD_DepthTo3dMisc, maskPathSanity)
{
    // Mask path is untouched by the vectorization; sanity-check it still works.
    Mat depth(30, 40, CV_32FC1);
    randu(depth, 0.2f, 5.f);
    Mat mask(depth.size(), CV_8U, Scalar(0));
    mask(Rect(0, 0, 20, 30)).setTo(255);

    Matx33f K(300.f, 0.f, 19.f, 0.f, 300.f, 14.f, 0.f, 0.f, 1.f);

    Mat points3d;
    depthTo3d(depth, K, points3d, mask);
    EXPECT_EQ(1, points3d.rows);
    EXPECT_EQ(countNonZero(mask), points3d.cols);

    // spot check the first collected point against the formula
    float z = depth.at<float>(0, 0);
    Vec4f p = points3d.at<Vec4f>(0, 0);
    EXPECT_FLOAT_EQ((0 - 19.f) / 300.f * z, p[0]);
    EXPECT_FLOAT_EQ((0 - 14.f) / 300.f * z, p[1]);
    EXPECT_FLOAT_EQ(z, p[2]);
}

}} // namespace
