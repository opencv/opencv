#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef perf::TestBaseWithParam<std::string> surf;

#define SURF_IMAGES \
    "cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png",\
    "stitching/a3.png"

#ifdef HAVE_OPENCV_OCL
#define OCL_TEST_CYCLE() for( ; startTimer(), next(); cv::ocl::finish(), stopTimer())
#endif

PERF_TEST_P(surf, detect, testing::Values(SURF_IMAGES))
{
    String filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame.empty()) << "Unable to load source image " << filename;

    declare.in(frame);

    Mat mask;
    vector<KeyPoint> points;
    Ptr<Feature2D> detector;

    if (getSelectedImpl() == "plain")
    {
        detector = new SURF;
        TEST_CYCLE() detector->operator()(frame, mask, points, noArray());
    }
#ifdef HAVE_OPENCV_OCL
    else if (getSelectedImpl() == "ocl")
    {
        detector = new ocl::SURF_OCL;
        OCL_TEST_CYCLE() detector->operator()(frame, mask, points, noArray());
    }
#endif
    else CV_TEST_FAIL_NO_IMPL();

    SANITY_CHECK_KEYPOINTS(points, 1e-3);
}

PERF_TEST_P(surf, extract, testing::Values(SURF_IMAGES))
{
    String filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame.empty()) << "Unable to load source image " << filename;

    declare.in(frame);

    Mat mask;
    Ptr<Feature2D> detector;
    vector<KeyPoint> points;
    vector<float> descriptors;

    if (getSelectedImpl() == "plain")
    {
        detector = new SURF;
        detector->operator()(frame, mask, points, noArray());
        TEST_CYCLE() detector->operator()(frame, mask, points, descriptors, true);
    }
#ifdef HAVE_OPENCV_OCL
    else if (getSelectedImpl() == "ocl")
    {
        detector = new ocl::SURF_OCL;
        detector->operator()(frame, mask, points, noArray());
        OCL_TEST_CYCLE() detector->operator()(frame, mask, points, descriptors, true);
    }
#endif
    else CV_TEST_FAIL_NO_IMPL();

    SANITY_CHECK(descriptors, 1e-4);
}

PERF_TEST_P(surf, full, testing::Values(SURF_IMAGES))
{
    String filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame.empty()) << "Unable to load source image " << filename;

    declare.in(frame).time(90);

    Mat mask;
    Ptr<Feature2D> detector;
    vector<KeyPoint> points;
    vector<float> descriptors;

    if (getSelectedImpl() == "plain")
    {
        detector = new SURF;
        TEST_CYCLE() detector->operator()(frame, mask, points, descriptors, false);
    }
#ifdef HAVE_OPENCV_OCL
    else if (getSelectedImpl() == "ocl")
    {
        detector = new ocl::SURF_OCL;
        detector->operator()(frame, mask, points, noArray());
        OCL_TEST_CYCLE() detector->operator()(frame, mask, points, descriptors, false);
    }
#endif
    else CV_TEST_FAIL_NO_IMPL();

    SANITY_CHECK_KEYPOINTS(points, 1e-3);
    SANITY_CHECK(descriptors, 1e-4);
}
