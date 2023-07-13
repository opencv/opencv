// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

using namespace cv;

class OctreeTest: public testing::Test
{
protected:
    void SetUp() override
    {
        pointCloudSize = 1000;
        resolution=0.0001;
        int scale;
        Point3i pmin, pmax;
        scale = 1<<20;
        pmin = Point3i(-scale, -scale, -scale);
        pmax = Point3i(scale, scale, scale);

        RNG& rng_Point = theRNG(); // set random seed for fixing output 3D point.

        // Generate 3D PointCloud
        for(size_t i = 0; i < pointCloudSize; i++)
        {
            float _x = 10 * (float)rng_Point.uniform(pmin.x, pmax.x)/scale;
            float _y = 10 * (float)rng_Point.uniform(pmin.y, pmax.y)/scale;
            float _z = 10 * (float)rng_Point.uniform(pmin.z, pmax.z)/scale;
            pointCloud.push_back(Point3f(_x, _y, _z));
        }

        // Generate Octree From PointCloud.
        treeTest.create(pointCloud, resolution);

        // Randomly generate another 3D point.
        float _x = 10 * (float)rng_Point.uniform(pmin.x, pmax.x)/scale;
        float _y = 10 * (float)rng_Point.uniform(pmin.y, pmax.y)/scale;
        float _z = 10 * (float)rng_Point.uniform(pmin.z, pmax.z)/scale;
        restPoint = Point3f(_x, _y, _z);
    }

public:
    //Origin point cloud data
    std::vector<Point3f> pointCloud;

    //Point cloud data from octree
    std::vector<Point3f> restorePointCloudData;

    //Color attribute of pointCloud from octree
    std::vector<Point3f> restorePointCloudColor;

    size_t pointCloudSize;
    Point3f restPoint;
    Octree treeTest;
    double resolution;
};

TEST_F(OctreeTest, BasicFunctionTest)
{
    // Check if the point in Bound.
    EXPECT_TRUE(treeTest.isPointInBound(restPoint));
    EXPECT_FALSE(treeTest.isPointInBound(restPoint + Point3f(60, 60, 60)));

    // insert, delete Test.
    EXPECT_FALSE(treeTest.deletePoint(restPoint));

    EXPECT_FALSE(treeTest.insertPoint(restPoint + Point3f(60, 60, 60)));
    EXPECT_TRUE(treeTest.insertPoint(restPoint));

    EXPECT_TRUE(treeTest.deletePoint(restPoint));

    EXPECT_FALSE(treeTest.empty());
    EXPECT_NO_THROW(treeTest.clear());
    EXPECT_TRUE(treeTest.empty());
}

TEST_F(OctreeTest, RadiusSearchTest)
{
    float radius = 2.0f;
    std::vector<Point3f> outputPoints;
    std::vector<float> outputSquareDist;
    EXPECT_NO_THROW(treeTest.radiusNNSearch(restPoint, radius, outputPoints, outputSquareDist));
    EXPECT_EQ(outputPoints.size(),(unsigned int)5);

    //The outputPoints should be unordered, so in some resolution, this test will fail because it specified the order of output.
    EXPECT_FLOAT_EQ(outputPoints[0].x, -8.88461112976f);
    EXPECT_FLOAT_EQ(outputPoints[0].y, -1.881799697875f);
    EXPECT_FLOAT_EQ(outputPoints[1].x, -8.405818939208f);
    EXPECT_FLOAT_EQ(outputPoints[1].y, -2.991247177124f);
    EXPECT_FLOAT_EQ(outputPoints[2].x, -8.1184864044189f);
    EXPECT_FLOAT_EQ(outputPoints[2].y, -0.528564453125f);
    EXPECT_FLOAT_EQ(outputPoints[3].x, -6.551313400268f);
    EXPECT_FLOAT_EQ(outputPoints[3].y, -0.708484649658f);
}

TEST_F(OctreeTest, KNNSearchTest)
{
    int K = 10;
    std::vector<Point3f> outputPoints;
    std::vector<float> outputSquareDist;
    EXPECT_NO_THROW(treeTest.KNNSearch(restPoint, K, outputPoints, outputSquareDist));

    EXPECT_FLOAT_EQ(outputPoints[0].x, -8.118486404418f);
    EXPECT_FLOAT_EQ(outputPoints[0].y, -0.528564453125f);
    EXPECT_FLOAT_EQ(outputPoints[1].x, -8.405818939208f);
    EXPECT_FLOAT_EQ(outputPoints[1].y, -2.991247177124f);
    EXPECT_FLOAT_EQ(outputPoints[2].x, -8.88461112976f);
    EXPECT_FLOAT_EQ(outputPoints[2].y, -1.881799697875f);
    EXPECT_FLOAT_EQ(outputPoints[3].x, -6.551313400268f);
    EXPECT_FLOAT_EQ(outputPoints[3].y, -0.708484649658f);
}

TEST_F(OctreeTest, restoreTest) {
    //restore the pointCloud data from octree.
    EXPECT_NO_THROW(treeTest.getPointCloudByOctree(restorePointCloudData,restorePointCloudColor));

    //The points in same leaf node will be seen as the same point. So if the resolution is small,
    //it will work as a downSampling function.
    EXPECT_LE(restorePointCloudData.size(),pointCloudSize);

    //The distance between the restore point cloud data and origin data should less than resolution.
    std::vector<Point3f> outputPoints;
    std::vector<float> outputSquareDist;
    EXPECT_NO_THROW(treeTest.getPointCloudByOctree(restorePointCloudData,restorePointCloudColor));
    EXPECT_NO_THROW(treeTest.KNNSearch(restorePointCloudData[0], 1, outputPoints, outputSquareDist));
    EXPECT_LE(abs(outputPoints[0].x-restorePointCloudData[0].x),resolution);
    EXPECT_LE(abs(outputPoints[0].y-restorePointCloudData[0].y),resolution);
    EXPECT_LE(abs(outputPoints[0].z-restorePointCloudData[0].z),resolution);
}

} // namespace
} // opencv_test