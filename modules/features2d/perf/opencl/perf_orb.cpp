#include "perf_precomp.hpp"
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
    ORB detector(1500, 1.3f, 1);
    vector<KeyPoint> points;

    OCL_TEST_CYCLE() detector(frame, mask, points);

    std::sort(points.begin(), points.end(), comparators::KeypointGreater());
    SANITY_CHECK_KEYPOINTS(points, 1e-5);
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

    ORB detector(1500, 1.3f, 1);
    vector<KeyPoint> points;
    detector(frame, mask, points);
    std::sort(points.begin(), points.end(), comparators::KeypointGreater());

    UMat descriptors;

    OCL_TEST_CYCLE() detector(frame, mask, points, descriptors, true);

    SANITY_CHECK(descriptors);
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
    ORB detector(1500, 1.3f, 1);

    vector<KeyPoint> points;
    UMat descriptors;

    OCL_TEST_CYCLE() detector(frame, mask, points, descriptors, false);

    ::perf::sort(points, descriptors);
    SANITY_CHECK_KEYPOINTS(points, 1e-5);
    SANITY_CHECK(descriptors);
}

} // ocl
} // cvtest

#endif // HAVE_OPENCL
