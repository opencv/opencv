#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

typedef ::perf::TestBaseWithParam<std::string> ORBFixture;

#define ORB_IMAGES OCL_PERF_ENUM("cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png", "stitching/a3.png")

OCL_PERF_TEST_P(ORBFixture, ORB_Detect, ORB_IMAGES)
{
    string filename = getDataPath(GetParam());
    Mat mframe = imread(filename, IMREAD_GRAYSCALE);

    if (mframe.empty())
        FAIL() << "Unable to load source image " << filename;

    UMat frame, mask;
    mframe.copyTo(frame);

    declare.in(frame);
    Ptr<ORB> detector = ORB::create(1500, 1.3f, 1);
    vector<KeyPoint> points;

    OCL_TEST_CYCLE() detector->detect(frame, points, mask);

    EXPECT_GT(points.size(), 20u);
    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST_P(ORBFixture, ORB_Extract, ORB_IMAGES)
{
    string filename = getDataPath(GetParam());
    Mat mframe = imread(filename, IMREAD_GRAYSCALE);

    if (mframe.empty())
        FAIL() << "Unable to load source image " << filename;

    UMat mask, frame;
    mframe.copyTo(frame);

    declare.in(frame);

    Ptr<ORB> detector = ORB::create(1500, 1.3f, 1);
    vector<KeyPoint> points;
    detector->detect(frame, points, mask);
    EXPECT_GT(points.size(), 20u);

    UMat descriptors;

    OCL_TEST_CYCLE() detector->compute(frame, points, descriptors);

    EXPECT_EQ((size_t)descriptors.rows, points.size());
    SANITY_CHECK_NOTHING();
}

OCL_PERF_TEST_P(ORBFixture, ORB_Full, ORB_IMAGES)
{
    string filename = getDataPath(GetParam());
    Mat mframe = imread(filename, IMREAD_GRAYSCALE);

    if (mframe.empty())
        FAIL() << "Unable to load source image " << filename;

    UMat mask, frame;
    mframe.copyTo(frame);

    declare.in(frame);
    Ptr<ORB> detector = ORB::create(1500, 1.3f, 1);

    vector<KeyPoint> points;
    UMat descriptors;

    OCL_TEST_CYCLE() detector->detectAndCompute(frame, mask, points, descriptors, false);

    EXPECT_GT(points.size(), 20u);
    EXPECT_EQ((size_t)descriptors.rows, points.size());
    SANITY_CHECK_NOTHING();
}

} // ocl
} // cvtest

#endif // HAVE_OPENCL
