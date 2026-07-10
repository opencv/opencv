// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

using namespace cv;

// Planar grid on z = 0 (spacing 0.1). PCA normals must be parallel to +/-Z.
static Mat makePlane(int n = 30)
{
    std::vector<Point3f> pts;
    for (int x = 0; x < n; x++)
        for (int y = 0; y < n; y++)
            pts.emplace_back(x * 0.1f, y * 0.1f, 0.0f);
    return Mat(pts).clone();
}

// Evenly sampled unit sphere (Fibonacci). Normals must be radial.
static Mat makeSphere(int n = 2000)
{
    std::vector<Point3f> pts;
    const float ga = (float)(CV_PI * (3.0 - std::sqrt(5.0)));   // golden angle
    for (int i = 0; i < n; i++)
    {
        float z = 1.f - 2.f * (i + 0.5f) / n;
        float r = std::sqrt(std::max(0.f, 1.f - z * z));
        float t = ga * i;
        pts.emplace_back(r * std::cos(t), r * std::sin(t), z);
    }
    return Mat(pts).clone();
}

TEST(Ptcloud_Normals, estimate_on_plane_is_axis)
{
    Mat cloud = makePlane();
    Mat normals;
    estimateNormals(cloud, normals, 8);

    ASSERT_EQ((int)normals.total(), (int)cloud.total());
    ASSERT_EQ(normals.channels(), 3);
    for (int i = 0; i < (int)normals.total(); i++)
    {
        Vec3f n = normals.at<Vec3f>(i);
        EXPECT_NEAR(cv::norm(n), 1.0, 1e-2);              // unit length
        EXPECT_GT(std::abs(n[2]), 0.99f);                // parallel to Z (plane normal)
    }
}

TEST(Ptcloud_Normals, orient_to_viewpoint)
{
    Mat cloud = makePlane();
    Mat normals;
    estimateNormals(cloud, normals, 8);

    const Point3f viewpoint(1.5f, 1.5f, 5.0f);           // above the plane
    orientNormals(cloud, normals, viewpoint);

    for (int i = 0; i < (int)normals.total(); i++)
    {
        Point3f p = cloud.at<Point3f>(i);
        Vec3f n = normals.at<Vec3f>(i);
        Vec3f toView(viewpoint.x - p.x, viewpoint.y - p.y, viewpoint.z - p.z);
        EXPECT_GE(n.dot(toView), 0.f);                   // every normal faces the viewpoint
    }
}

TEST(Ptcloud_Normals, consistent_on_sphere_is_outward)
{
    Mat cloud = makeSphere(2000);
    Mat normals;
    estimateNormals(cloud, normals, 12);
    orientNormalsConsistent(cloud, normals, 12);

    // Sphere is centered at the origin, so a consistent orientation seeded outward
    // must leave (almost) every normal pointing away from the center.
    int outward = 0;
    for (int i = 0; i < (int)normals.total(); i++)
    {
        Point3f p = cloud.at<Point3f>(i);
        Vec3f n = normals.at<Vec3f>(i);
        if (n.dot(Vec3f(p.x, p.y, p.z)) > 0.f) outward++;
    }
    EXPECT_GE(outward, (int)(0.98 * normals.total()));
}

TEST(Ptcloud_Normals, empty_input)
{
    Mat empty, normals;
    estimateNormals(empty, normals, 8);
    EXPECT_TRUE(normals.empty());
    EXPECT_NO_THROW(orientNormals(empty, normals, Point3f(0, 0, 1)));
    EXPECT_NO_THROW(orientNormalsConsistent(empty, normals, 8));
}

}} // namespace
