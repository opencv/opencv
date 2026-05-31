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
        resolution = 0.0001;
        int scale;
        Point3i pmin, pmax;
        scale = 1 << 20;
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

        // Generate Octree From PointCloud
        treeTest = Octree::createWithResolution(resolution, pointCloud);

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
    Ptr<Octree> treeTest;
    double resolution;
};

TEST_F(OctreeTest, BasicFunctionTest)
{
    // Check if the point in Bound.
    EXPECT_TRUE(treeTest->isPointInBound(restPoint));
    EXPECT_FALSE(treeTest->isPointInBound(restPoint + Point3f(60, 60, 60)));

    // insert, delete Test.
    EXPECT_FALSE(treeTest->deletePoint(restPoint));
    EXPECT_FALSE(treeTest->insertPoint(restPoint + Point3f(60, 60, 60)));
    EXPECT_TRUE(treeTest->insertPoint(restPoint));
    EXPECT_TRUE(treeTest->deletePoint(restPoint));

    EXPECT_FALSE(treeTest->empty());
    EXPECT_NO_THROW(treeTest->clear());
    EXPECT_TRUE(treeTest->empty());
}

TEST_F(OctreeTest, RadiusSearchTest)
{
    float radius = 2.0f;
    std::vector<Point3f> outputPoints;
    std::vector<float> outputSquareDist;
    EXPECT_NO_THROW(treeTest->radiusNNSearch(restPoint, radius, outputPoints, outputSquareDist));
    EXPECT_EQ(outputPoints.size(),(unsigned int)5);

    // The output is unsorted, so let's sort it before checking
    std::map<float, Point3f> sortResults;
    for (int i = 0; i < (int)outputPoints.size(); i++)
    {
        sortResults[outputSquareDist[i]] = outputPoints[i];
    }

    std::vector<Point3f> goldVals = {
        {-8.1184864044189f, -0.528564453125f, 0.f},
        {-8.405818939208f,  -2.991247177124f, 0.f},
        {-8.88461112976f,   -1.881799697875f, 0.f},
        {-6.551313400268f,  -0.708484649658f, 0.f}
    };

    auto it = sortResults.begin();
    for (int i = 0; i < (int)goldVals.size(); i++, it++)
    {
        Point3f p = it->second;
        EXPECT_FLOAT_EQ(goldVals[i].x, p.x);
        EXPECT_FLOAT_EQ(goldVals[i].y, p.y);
    }
}

TEST_F(OctreeTest, KNNSearchTest)
{
    int K = 10;
    std::vector<Point3f> outputPoints;
    std::vector<float> outputSquareDist;
    EXPECT_NO_THROW(treeTest->KNNSearch(restPoint, K, outputPoints, outputSquareDist));

    // The output is unsorted, so let's sort it before checking
    std::map<float, Point3f> sortResults;
    for (int i = 0; i < (int)outputPoints.size(); i++)
    {
        sortResults[outputSquareDist[i]] = outputPoints[i];
    }

    std::vector<Point3f> goldVals = {
        { -8.118486404418f, -0.528564453125f, 0.f },
        { -8.405818939208f, -2.991247177124f, 0.f },
        { -8.884611129760f, -1.881799697875f, 0.f },
        { -6.551313400268f, -0.708484649658f, 0.f }
    };

    auto it = sortResults.begin();
    for (int i = 0; i < (int)goldVals.size(); i++, it++)
    {
        Point3f p = it->second;
        EXPECT_FLOAT_EQ(goldVals[i].x, p.x);
        EXPECT_FLOAT_EQ(goldVals[i].y, p.y);
    }
}

TEST_F(OctreeTest, restoreTest) {
    //restore the pointCloud data from octree.
    EXPECT_NO_THROW(treeTest->getPointCloudByOctree(restorePointCloudData,restorePointCloudColor));

    //The points in same leaf node will be seen as the same point. So if the resolution is small,
    //it will work as a downSampling function.
    EXPECT_LE(restorePointCloudData.size(),pointCloudSize);

    //The distance between the restore point cloud data and origin data should less than resolution.
    std::vector<Point3f> outputPoints;
    std::vector<float> outputSquareDist;
    EXPECT_NO_THROW(treeTest->getPointCloudByOctree(restorePointCloudData,restorePointCloudColor));
    EXPECT_NO_THROW(treeTest->KNNSearch(restorePointCloudData[0], 1, outputPoints, outputSquareDist));
    EXPECT_LE(abs(outputPoints[0].x - restorePointCloudData[0].x), resolution);
    EXPECT_LE(abs(outputPoints[0].y - restorePointCloudData[0].y), resolution);
    EXPECT_LE(abs(outputPoints[0].z - restorePointCloudData[0].z), resolution);
}

} // namespace
} // opencv_test