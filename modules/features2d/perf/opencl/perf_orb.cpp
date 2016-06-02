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

    Ptr<ORB> detector = ORB::create(1500, 1.3f, 1);
    vector<KeyPoint> points;
    detector->detect(frame, points, mask);
    std::sort(points.begin(), points.end(), comparators::KeypointGreater());

    UMat descriptors;

    OCL_TEST_CYCLE() detector->compute(frame, points, descriptors);

    SANITY_CHECK(descriptors);
}

OCL_PERF_TEST_P(ORBFixture, ORB_Full, ORB_IMAGES)
{
    string filename = getDataPath(GetParam());
    Mat mframe = imread(filename, IMREAD_GRAYSCALE);

    double desc_eps = 1e-6;
#ifdef ANDROID
    if (cv::ocl::Device::getDefault().isNVidia())
        desc_eps = 2;
#endif

    if (mframe.empty())
        FAIL() << "Unable to load source image " << filename;

    UMat mask, frame;
    mframe.copyTo(frame);

    declare.in(frame);
    Ptr<ORB> detector = ORB::create(1500, 1.3f, 1);

    vector<KeyPoint> points;
    UMat descriptors;

    OCL_TEST_CYCLE() detector->detectAndCompute(frame, mask, points, descriptors, false);

    ::perf::sort(points, descriptors);
    SANITY_CHECK_KEYPOINTS(points, 1e-5);
    SANITY_CHECK(descriptors, desc_eps);
}

} // ocl
} // cvtest

#endif // HAVE_OPENCL
