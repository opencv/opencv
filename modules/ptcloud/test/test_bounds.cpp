// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

using namespace cv;

TEST(Ptcloud_Bounds, aabb_known_box)
{
    // 8 corners of [1,4] x [2,5] x [-1,2]
    std::vector<Point3f> pts;
    for (float x : {1.f, 4.f}) for (float y : {2.f, 5.f}) for (float z : {-1.f, 2.f})
        pts.emplace_back(x, y, z);
    Mat cloud(pts);

    Mat lo, hi;
    getPointCloudBounds(cloud, lo, hi);
    ASSERT_EQ(lo.total(), (size_t)3);
    ASSERT_EQ(hi.total(), (size_t)3);
    EXPECT_NEAR(lo.at<float>(0), 1.f, 1e-5); EXPECT_NEAR(lo.at<float>(1), 2.f, 1e-5); EXPECT_NEAR(lo.at<float>(2), -1.f, 1e-5);
    EXPECT_NEAR(hi.at<float>(0), 4.f, 1e-5); EXPECT_NEAR(hi.at<float>(1), 5.f, 1e-5); EXPECT_NEAR(hi.at<float>(2),  2.f, 1e-5);
}

TEST(Ptcloud_Bounds, obb_recovers_rotated_box)
{
    // Grid filling a box of half-extents (2, 1, 0.5) centered at origin, then rotate+translate.
    const Vec3f he(2.f, 1.f, 0.5f);
    const Vec3f t(5.f, -3.f, 2.f);
    const float a = 0.5f, b = 0.3f;
    Matx33f Rz(std::cos(a), -std::sin(a), 0,  std::sin(a), std::cos(a), 0,  0, 0, 1);
    Matx33f Rx(1, 0, 0,  0, std::cos(b), -std::sin(b),  0, std::sin(b), std::cos(b));
    Matx33f R = Rz * Rx;

    std::vector<Point3f> pts;
    for (int i = -4; i <= 4; i++) for (int j = -3; j <= 3; j++) for (int k = -2; k <= 2; k++)
    {
        Vec3f p(he[0]*i/4.f, he[1]*j/3.f, he[2]*k/2.f);   // spans exactly [-he, he]
        Vec3f q = R * p + t;
        pts.emplace_back(q[0], q[1], q[2]);
    }
    Mat cloud(pts);

    Mat center, axes, half;
    getOrientedBoundingBox(cloud, center, axes, half);
    ASSERT_EQ(axes.rows, 3); ASSERT_EQ(axes.cols, 3);

    // Extents come back in descending order (PCA sorts by variance): 2, 1, 0.5.
    EXPECT_NEAR(half.at<float>(0), 2.0f, 1e-2);
    EXPECT_NEAR(half.at<float>(1), 1.0f, 1e-2);
    EXPECT_NEAR(half.at<float>(2), 0.5f, 1e-2);
    // Center is the box center after the same transform (box was origin-centered -> t).
    EXPECT_NEAR(center.at<float>(0), t[0], 1e-2);
    EXPECT_NEAR(center.at<float>(1), t[1], 1e-2);
    EXPECT_NEAR(center.at<float>(2), t[2], 1e-2);

    // Axes are orthonormal, and every point lies inside: |axes*(p-center)| <= half.
    Matx33f A; for (int r=0;r<3;r++) for (int c=0;c<3;c++) A(r,c)=axes.at<float>(r,c);
    Vec3f cen(center.at<float>(0), center.at<float>(1), center.at<float>(2));
    Vec3f hev(half.at<float>(0), half.at<float>(1), half.at<float>(2));
    Matx33f AAt = A * A.t();
    for (int r=0;r<3;r++) for (int c=0;c<3;c++)
        EXPECT_NEAR(AAt(r,c), (r==c)?1.f:0.f, 1e-3);   // orthonormal
    for (const Point3f& p : pts)
    {
        Vec3f proj = A * (Vec3f(p.x,p.y,p.z) - cen);
        for (int d = 0; d < 3; d++)
            EXPECT_LE(std::abs(proj[d]), hev[d] + 1e-3f);
    }
}

TEST(Ptcloud_Bounds, sphere_encloses_all)
{
    // Points on a sphere of radius 3 centered at (1,2,3).
    const Point3f ctr(1.f, 2.f, 3.f); const float R = 3.f;
    std::vector<Point3f> pts;
    const float ga = (float)(CV_PI * (3.0 - std::sqrt(5.0)));
    for (int i = 0; i < 1500; i++)
    {
        float z = 1.f - 2.f * (i + 0.5f) / 1500;
        float rr = std::sqrt(std::max(0.f, 1.f - z*z));
        float th = ga * i;
        pts.emplace_back(ctr.x + R*rr*std::cos(th), ctr.y + R*rr*std::sin(th), ctr.z + R*z);
    }
    Mat cloud(pts);

    Mat center;
    double radius = getBoundingSphere(cloud, center);
    Vec3f c(center.at<float>(0), center.at<float>(1), center.at<float>(2));

    EXPECT_LE(cv::norm(c - Vec3f(ctr.x, ctr.y, ctr.z)), 0.2);   // center recovered
    EXPECT_GE(radius, (double)R - 1e-2);                        // must enclose the true sphere
    EXPECT_LE(radius, (double)R * 1.10);                        // Ritter looseness bound (~10%)

    for (const Point3f& p : pts)                               // every point inside
        EXPECT_LE(cv::norm(Vec3f(p.x,p.y,p.z) - c), radius + 1e-3);
}

TEST(Ptcloud_Bounds, empty_and_tiny)
{
    Mat empty, a, b, cc;
    getPointCloudBounds(empty, a, b);
    EXPECT_TRUE(a.empty());
    EXPECT_EQ(getBoundingSphere(empty, cc), 0.0);
    EXPECT_TRUE(cc.empty());

    // one point: must not crash.
    Mat one = (Mat_<float>(1, 3) << 1, 2, 3), center, axes, half;
    EXPECT_NO_THROW(getOrientedBoundingBox(one, center, axes, half));
    EXPECT_NO_THROW(getBoundingSphere(one, center));
}

}} // namespace
