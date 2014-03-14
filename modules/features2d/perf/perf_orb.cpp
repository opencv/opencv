#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef perf::TestBaseWithParam<std::string> orb;

#define ORB_IMAGES \
    "cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png",\
    "stitching/a3.png"

PERF_TEST_P(orb, detect, testing::Values(ORB_IMAGES))
{
    string filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);

    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame);
    ORB detector(1500, 1.3f, 1);
    vector<KeyPoint> points;

    TEST_CYCLE() detector(frame, mask, points);

    sort(points.begin(), points.end(), comparators::KeypointGreater());
    SANITY_CHECK_KEYPOINTS(points, 1e-5);
}

PERF_TEST_P(orb, extract, testing::Values(ORB_IMAGES))
{
    string filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);

    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame);

    ORB detector(1500, 1.3f, 1);
    vector<KeyPoint> points;
    detector(frame, mask, points);
    sort(points.begin(), points.end(), comparators::KeypointGreater());

    Mat descriptors;

    TEST_CYCLE() detector(frame, mask, points, descriptors, true);

    SANITY_CHECK(descriptors);
}

PERF_TEST_P(orb, full, testing::Values(ORB_IMAGES))
{
    string filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);

    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame);
    ORB detector(1500, 1.3f, 1);

    vector<KeyPoint> points;
    Mat descriptors;

    TEST_CYCLE() detector(frame, mask, points, descriptors, false);

    perf::sort(points, descriptors);
    SANITY_CHECK_KEYPOINTS(points, 1e-5);
    SANITY_CHECK(descriptors);
}
