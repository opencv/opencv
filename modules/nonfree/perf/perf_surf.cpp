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
static Ptr<Feature2D> getSURF()
{
    ocl::PlatformsInfo p;
    if(ocl::getOpenCLPlatforms(p) > 0)
        return new ocl::SURF_OCL;
    else
        return new SURF;
}
#else
static Ptr<Feature2D> getSURF()
{
    return new SURF;
}
#endif

PERF_TEST_P(surf, detect, testing::Values(SURF_IMAGES))
{
    String filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);

    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame).time(90);
    Ptr<Feature2D> detector = getSURF();

    vector<KeyPoint> points;

    TEST_CYCLE() detector->operator()(frame, mask, points, noArray());

    SANITY_CHECK_KEYPOINTS(points, 1e-3);
}

PERF_TEST_P(surf, extract, testing::Values(SURF_IMAGES))
{
    String filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);

    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame).time(90);

    Ptr<Feature2D> detector = getSURF();
    vector<KeyPoint> points;
    vector<float> descriptors;
    detector->operator()(frame, mask, points, noArray());

    TEST_CYCLE() detector->operator()(frame, mask, points, descriptors, true);

    SANITY_CHECK(descriptors, 1e-4);
}

PERF_TEST_P(surf, full, testing::Values(SURF_IMAGES))
{
    String filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);

    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame).time(90);
    Ptr<Feature2D> detector = getSURF();
    vector<KeyPoint> points;
    vector<float> descriptors;

    TEST_CYCLE() detector->operator()(frame, mask, points, descriptors, false);

    SANITY_CHECK_KEYPOINTS(points, 1e-3);
    SANITY_CHECK(descriptors, 1e-4);
}
