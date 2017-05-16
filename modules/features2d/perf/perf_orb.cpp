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
    Ptr<ORB> detector = ORB::create(1500, 1.3f, 1);
    vector<KeyPoint> points;

    TEST_CYCLE() detector->detect(frame, points, mask);

    EXPECT_GT(points.size(), 20u);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(orb, extract, testing::Values(ORB_IMAGES))
{
    string filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);

    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame);

    Ptr<ORB> detector = ORB::create(1500, 1.3f, 1);
    vector<KeyPoint> points;
    detector->detect(frame, points, mask);

    EXPECT_GT(points.size(), 20u);

    Mat descriptors;

    TEST_CYCLE() detector->compute(frame, points, descriptors);

    EXPECT_EQ((size_t)descriptors.rows, points.size());
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(orb, full, testing::Values(ORB_IMAGES))
{
    string filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);

    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame);
    Ptr<ORB> detector = ORB::create(1500, 1.3f, 1);

    vector<KeyPoint> points;
    Mat descriptors;

    TEST_CYCLE() detector->detectAndCompute(frame, mask, points, descriptors, false);

    EXPECT_GT(points.size(), 20u);
    EXPECT_EQ((size_t)descriptors.rows, points.size());
    SANITY_CHECK_NOTHING();
}
