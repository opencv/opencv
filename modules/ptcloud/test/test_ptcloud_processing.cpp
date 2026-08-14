// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "opencv2/geometry/segment.hpp"   // cv::normalEstimate

namespace opencv_test { namespace {

using namespace cv;

// ---------------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------------

// Evenly sampled unit sphere (Fibonacci lattice), returned as Nx1 CV_32FC3.
static Mat makeSphere(int n = 2000)
{
    std::vector<Point3f> pts;
    const float ga = (float)(CV_PI * (3.0 - std::sqrt(5.0)));
    for (int i = 0; i < n; i++)
    {
        float z = 1.f - 2.f * (i + 0.5f) / n;
        float r = std::sqrt(std::max(0.f, 1.f - z * z));
        float t = ga * i;
        pts.emplace_back(r * std::cos(t), r * std::sin(t), z);
    }
    return Mat(pts).clone();
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

// Dense unit-cube grid of inliers plus a handful of far-away isolated outliers.
static Mat makeCloud(int& numInliers, int& numOutliers)
{
    std::vector<Point3f> pts;
    for (int x = 0; x < 10; x++)
        for (int y = 0; y < 10; y++)
            for (int z = 0; z < 10; z++)
                pts.emplace_back(x * 0.1f, y * 0.1f, z * 0.1f);   // spacing 0.1
    numInliers = (int)pts.size();

    const Point3f outliers[] = {
        {5.f, 5.f, 5.f}, {-4.f, 2.f, 1.f}, {3.f, -6.f, 2.f}, {8.f, 8.f, -3.f}, {-5.f, -5.f, 7.f}
    };
    for (const Point3f& o : outliers)
        pts.push_back(o);
    numOutliers = (int)(sizeof(outliers) / sizeof(outliers[0]));

    return Mat(pts).clone();   // N x 1, CV_32FC3
}

// Normals via the geometry estimator (internal kNN), returned as Nx1 CV_32FC3.
static Mat computeNormals(const Mat& cloud, int k)
{
    Mat normals, curv;
    normalEstimate(normals, curv, cloud, noArray(), k);
    return normals.reshape(3, (int)cloud.total());
}

// Estimated normals oriented into a globally consistent field.
static Mat orientedNormals(const Mat& cloud, int k)
{
    Mat normals = computeNormals(cloud, k);
    orientNormalsConsistent(cloud, normals, k);
    return normals;
}

// ---------------------------------------------------------------------------------------------
// Ball-pivoting surface reconstruction
// ---------------------------------------------------------------------------------------------

TEST(Ptcloud_BPA, reconstructs_sphere)
{
    Mat cloud = makeSphere(2000);
    Mat normals = orientedNormals(cloud, 12);

    Mat vertices, triangles;
    createMeshBPA(cloud, normals, vertices, triangles);

    // Mesh is interpolating: vertices are the input points.
    ASSERT_EQ((int)vertices.total(), (int)cloud.total());
    ASSERT_FALSE(triangles.empty());
    ASSERT_EQ(triangles.cols, 3);
    ASSERT_EQ(triangles.type(), CV_32S);

    // A well-reconstructed closed surface has roughly 2N triangles (Euler). Allow a wide band.
    EXPECT_GT(triangles.rows, (int)(0.7 * cloud.total()));
    EXPECT_LT(triangles.rows, (int)(3.0 * cloud.total()));

    // Every index must be a valid vertex, and no degenerate triangle.
    for (int i = 0; i < triangles.rows; i++)
    {
        const int* t = triangles.ptr<int>(i);
        for (int j = 0; j < 3; j++)
        {
            EXPECT_GE(t[j], 0);
            EXPECT_LT(t[j], (int)cloud.total());
        }
        EXPECT_TRUE(t[0] != t[1] && t[1] != t[2] && t[0] != t[2]);
    }
}

TEST(Ptcloud_BPA, explicit_radii)
{
    Mat cloud = makeSphere(1500);
    Mat normals = orientedNormals(cloud, 12);

    float s = estimateMedianSpacing(cloud);
    ASSERT_GT(s, 0.f);
    std::vector<double> radii = { 1.5 * s, 3.0 * s };

    Mat vertices, triangles;
    createMeshBPA(cloud, normals, vertices, triangles, radii);
    EXPECT_FALSE(triangles.empty());
}

TEST(Ptcloud_BPA, mean_spacing_grid)
{
    // Grid with a known 0.1 spacing -> nearest neighbor distance is 0.1.
    std::vector<Point3f> pts;
    for (int x = 0; x < 20; x++)
        for (int y = 0; y < 20; y++)
            pts.emplace_back(x * 0.1f, y * 0.1f, 0.f);
    Mat cloud(pts);
    EXPECT_NEAR(estimateMedianSpacing(cloud), 0.1f, 1e-3f);
}

// Non-uniform sampling + surface noise: the input that actually exercises the exact
// empty-ball / candidate searches (a perfectly uniform sphere hides approximate-search bugs).
// The result must stay manifold: every undirected edge is shared by at most two triangles.
TEST(Ptcloud_BPA, manifold_on_noisy_sphere)
{
    RNG rng(7);
    std::vector<Point3f> pts;
    const float ga = (float)(CV_PI * (3.0 - std::sqrt(5.0)));
    for (int i = 0; i < 3000; i++)
    {
        // random (non-stratified) sampling + radial/positional jitter
        float u = (float)rng.uniform(0.0, 1.0);
        float z = 1.f - 2.f * u;
        float r = std::sqrt(std::max(0.f, 1.f - z * z));
        float t = ga * i + (float)rng.uniform(-0.1, 0.1);
        float rad = 1.f + (float)rng.gaussian(0.01);
        pts.emplace_back(rad * r * std::cos(t) + (float)rng.gaussian(0.005),
                         rad * r * std::sin(t) + (float)rng.gaussian(0.005),
                         rad * z + (float)rng.gaussian(0.005));
    }
    Mat cloud(pts);
    Mat normals = orientedNormals(cloud, 15);

    Mat vertices, triangles;
    createMeshBPA(cloud, normals, vertices, triangles);
    ASSERT_FALSE(triangles.empty());

    std::map<std::pair<int,int>, int> edgeUse;
    for (int i = 0; i < triangles.rows; i++)
    {
        const int* t = triangles.ptr<int>(i);
        for (int e = 0; e < 3; e++)
        {
            int a = t[e], b = t[(e + 1) % 3];
            edgeUse[{std::min(a,b), std::max(a,b)}]++;
        }
    }
    int nonManifold = 0;
    for (const auto& kv : edgeUse)
        if (kv.second > 2) nonManifold++;
    EXPECT_EQ(nonManifold, 0) << nonManifold << " edges shared by >2 triangles";
}

TEST(Ptcloud_BPA, empty_and_tiny)
{
    Mat empty, v, t;
    createMeshBPA(empty, empty, v, t);
    EXPECT_TRUE(v.empty());
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(estimateMedianSpacing(empty), 0.f);

    Mat two = (Mat_<float>(2, 3) << 0, 0, 0, 1, 0, 0);
    Mat n = (Mat_<float>(2, 3) << 0, 0, 1, 0, 0, 1), vv, tt;
    createMeshBPA(two, n, vv, tt);      // < 3 points -> no mesh, no crash
    EXPECT_TRUE(vv.empty());
}

// ---------------------------------------------------------------------------------------------
// Bounding volumes
// ---------------------------------------------------------------------------------------------

TEST(Ptcloud_Bounds, aabb_known_box)
{
    // 8 corners of [1,4] x [2,5] x [-1,2]
    std::vector<Point3f> pts;
    for (float x : {1.f, 4.f}) for (float y : {2.f, 5.f}) for (float z : {-1.f, 2.f})
        pts.emplace_back(x, y, z);
    Mat cloud(pts);

    Mat lo, hi;
    boundingBox3D(cloud, lo, hi);
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
    orientedBoundingBox3D(cloud, center, axes, half);
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
    double radius = approxEnclosingSphere3D(cloud, center);
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
    boundingBox3D(empty, a, b);
    EXPECT_TRUE(a.empty());
    EXPECT_EQ(approxEnclosingSphere3D(empty, cc), 0.0);
    EXPECT_TRUE(cc.empty());

    // one point: must not crash.
    Mat one = (Mat_<float>(1, 3) << 1, 2, 3), center, axes, half;
    EXPECT_NO_THROW(orientedBoundingBox3D(one, center, axes, half));
    EXPECT_NO_THROW(approxEnclosingSphere3D(one, center));
}

// ---------------------------------------------------------------------------------------------
// Normal orientation
// ---------------------------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------------------------
// Outlier removal
// ---------------------------------------------------------------------------------------------

TEST(Ptcloud_OutlierRemoval, statistical_drops_isolated_points)
{
    int nIn = 0, nOut = 0;
    Mat cloud = makeCloud(nIn, nOut);

    Mat filtered, kept;
    removeStatisticalOutliers(cloud, filtered, 20, 1.0, kept);

    // Every one of the far outliers must be gone; the vast majority of inliers must survive.
    EXPECT_LE(filtered.total(), (size_t)nIn);
    EXPECT_GE(filtered.total(), (size_t)(nIn * 0.9));
    EXPECT_EQ(kept.total(), filtered.total());

    // No surviving point may be one of the planted outliers.
    for (int i = 0; i < (int)filtered.total(); i++)
    {
        Point3f p = filtered.at<Point3f>(i);
        EXPECT_LT(cv::norm(Vec3f(p.x, p.y, p.z)), 3.0);
    }
}

TEST(Ptcloud_OutlierRemoval, radius_drops_isolated_points)
{
    int nIn = 0, nOut = 0;
    Mat cloud = makeCloud(nIn, nOut);

    // With spacing 0.1, a radius of 0.15 sees the 6 axis neighbors for interior points;
    // the isolated outliers see nobody.
    Mat filtered, kept;
    removeRadiusOutliers(cloud, filtered, 0.15, 3, kept);

    EXPECT_GT(filtered.total(), (size_t)0);
    EXPECT_LE(filtered.total(), (size_t)nIn);
    EXPECT_EQ(kept.total(), filtered.total());

    for (int i = 0; i < (int)filtered.total(); i++)
    {
        Point3f p = filtered.at<Point3f>(i);
        EXPECT_LT(cv::norm(Vec3f(p.x, p.y, p.z)), 3.0);
    }
}

TEST(Ptcloud_OutlierRemoval, empty_input)
{
    Mat empty, out;
    removeStatisticalOutliers(empty, out);
    EXPECT_TRUE(out.empty());
    removeRadiusOutliers(empty, out, 1.0);
    EXPECT_TRUE(out.empty());
}

TEST(Ptcloud_OutlierRemoval, accepts_nx3_layout)
{
    int nIn = 0, nOut = 0;
    Mat cloud3c = makeCloud(nIn, nOut);
    Mat cloudNx3 = cloud3c.reshape(1, (int)cloud3c.total());   // Nx3 CV_32F

    Mat a, b;
    removeStatisticalOutliers(cloud3c, a, 20, 1.0);
    removeStatisticalOutliers(cloudNx3, b, 20, 1.0);
    EXPECT_EQ(a.total(), b.total());
}

}} // namespace
