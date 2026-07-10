// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

using namespace cv;

// A dense unit-cube grid of inliers plus a handful of far-away isolated points.
// Both filters must keep (most of) the grid and drop the scattered outliers.
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
