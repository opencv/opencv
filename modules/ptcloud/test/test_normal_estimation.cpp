// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "opencv2/geometry/segment.hpp"   // cv::normalEstimate (produces the normals to orient)

namespace opencv_test { namespace {

using namespace cv;

// Normals via the geometry estimator (internal kNN), returned as Nx1 CV_32FC3.
static Mat computeNormals(const Mat& cloud, int k)
{
    Mat normals, curv;
    normalEstimate(normals, curv, cloud, noArray(), k);
    return normals.reshape(3, (int)cloud.total());
}

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

TEST(Ptcloud_Normals, orient_to_viewpoint)
{
    Mat cloud = makePlane();
    Mat normals = computeNormals(cloud, 8);

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
    Mat normals = computeNormals(cloud, 12);
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
    EXPECT_NO_THROW(orientNormals(empty, normals, Point3f(0, 0, 1)));
    EXPECT_NO_THROW(orientNormalsConsistent(empty, normals, 8));
}

}} // namespace
