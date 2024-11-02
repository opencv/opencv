#include "perf_feature2d.hpp"

namespace opencv_test
{
using namespace perf;

PERF_TEST_P(features, detect, testing::Combine(Feature2DType::all(), TEST_IMAGES))
{
    Ptr<Feature2D> detector = getFeature2D(get<0>(GetParam()));
    std::string filename = getDataPath(get<1>(GetParam()));
    Mat img = imread(filename, IMREAD_GRAYSCALE);

    ASSERT_FALSE(img.empty());
    ASSERT_TRUE(detector);

    declare.in(img);
    Mat mask;
    vector<KeyPoint> points;

    TEST_CYCLE() detector->detect(img, points, mask);

    EXPECT_GT(points.size(), 20u);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(features, extract, testing::Combine(testing::Values(DETECTORS_EXTRACTORS), TEST_IMAGES))
{
    Ptr<Feature2D> detector = ORB::create();
    Ptr<Feature2D> extractor = getFeature2D(get<0>(GetParam()));
    std::string filename = getDataPath(get<1>(GetParam()));
    Mat img = imread(filename, IMREAD_GRAYSCALE);

    ASSERT_FALSE(img.empty());
    ASSERT_TRUE(extractor);

    declare.in(img);
    Mat mask;
    vector<KeyPoint> points;
    detector->detect(img, points, mask);

    EXPECT_GT(points.size(), 20u);

    Mat descriptors;

    TEST_CYCLE() extractor->compute(img, points, descriptors);

    EXPECT_EQ((size_t)descriptors.rows, points.size());
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(features, detectAndExtract, testing::Combine(testing::Values(DETECTORS_EXTRACTORS), TEST_IMAGES))
{
    Ptr<Feature2D> detector = getFeature2D(get<0>(GetParam()));
    std::string filename = getDataPath(get<1>(GetParam()));
    Mat img = imread(filename, IMREAD_GRAYSCALE);

    ASSERT_FALSE(img.empty());
    ASSERT_TRUE(detector);

    declare.in(img);
    Mat mask;
    vector<KeyPoint> points;
    Mat descriptors;

    TEST_CYCLE() detector->detectAndCompute(img, mask, points, descriptors, false);

    EXPECT_GT(points.size(), 20u);
    EXPECT_EQ((size_t)descriptors.rows, points.size());
    SANITY_CHECK_NOTHING();
}

} // namespace
