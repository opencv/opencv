#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"
#include "../perf_feature2d.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

OCL_PERF_TEST_P(features, detect, testing::Combine(Feature2DType::all(), TEST_IMAGES))
{
    Ptr<Feature2D> detector = getFeature2D(get<0>(GetParam()));
    std::string filename = getDataPath(get<1>(GetParam()));
    Mat mimg = imread(filename, IMREAD_GRAYSCALE);

    ASSERT_FALSE(mimg.empty());
    ASSERT_TRUE(detector);

    UMat img, mask;
    mimg.copyTo(img);
    declare.in(img);
    vector<KeyPoint> points;

    OCL_TEST_CYCLE() detector->detect(img, points, mask);

    EXPECT_GT(points.size(), 20u);
    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST_P(features, extract, testing::Combine(testing::Values(DETECTORS_EXTRACTORS), TEST_IMAGES))
{
    Ptr<Feature2D> detector = ORB::create();
    Ptr<Feature2D> extractor = getFeature2D(get<0>(GetParam()));
    std::string filename = getDataPath(get<1>(GetParam()));
    Mat mimg = imread(filename, IMREAD_GRAYSCALE);

    ASSERT_FALSE(mimg.empty());
    ASSERT_TRUE(extractor);

    UMat img, mask;
    mimg.copyTo(img);
    declare.in(img);
    vector<KeyPoint> points;
    detector->detect(img, points, mask);

    EXPECT_GT(points.size(), 20u);

    UMat descriptors;

    OCL_TEST_CYCLE() extractor->compute(img, points, descriptors);

    EXPECT_EQ((size_t)descriptors.rows, points.size());
    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST_P(features, detectAndExtract, testing::Combine(testing::Values(DETECTORS_EXTRACTORS), TEST_IMAGES))
{
    Ptr<Feature2D> detector = getFeature2D(get<0>(GetParam()));
    std::string filename = getDataPath(get<1>(GetParam()));
    Mat mimg = imread(filename, IMREAD_GRAYSCALE);

    ASSERT_FALSE(mimg.empty());
    ASSERT_TRUE(detector);

    UMat img, mask;
    mimg.copyTo(img);
    declare.in(img);
    vector<KeyPoint> points;
    UMat descriptors;

    OCL_TEST_CYCLE() detector->detectAndCompute(img, mask, points, descriptors, false);

    EXPECT_GT(points.size(), 20u);
    EXPECT_EQ((size_t)descriptors.rows, points.size());
    SANITY_CHECK_NOTHING();
}

} // ocl
} // cvtest

#endif // HAVE_OPENCL
