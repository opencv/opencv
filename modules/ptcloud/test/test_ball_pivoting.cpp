// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/geometry/segment.hpp"   // cv::normalEstimate

namespace opencv_test { namespace {

using namespace cv;

static Mat makeSphere(int n)
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

static Mat orientedNormals(const Mat& cloud, int k)
{
    Mat normals, curv;
    normalEstimate(normals, curv, cloud, noArray(), k);
    normals = normals.reshape(3, (int)cloud.total());
    orientNormalsConsistent(cloud, normals, k);
    return normals;
}

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

    float s = estimateMeanSpacing(cloud);
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
    EXPECT_NEAR(estimateMeanSpacing(cloud), 0.1f, 1e-3f);
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
    EXPECT_EQ(estimateMeanSpacing(empty), 0.f);

    Mat two = (Mat_<float>(2, 3) << 0, 0, 0, 1, 0, 0);
    Mat n = (Mat_<float>(2, 3) << 0, 0, 1, 0, 0, 1), vv, tt;
    createMeshBPA(two, n, vv, tt);      // < 3 points -> no mesh, no crash
    EXPECT_TRUE(vv.empty());
}

}} // namespace
