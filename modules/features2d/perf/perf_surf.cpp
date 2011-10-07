#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;


typedef perf::TestBaseWithParam<std::string> surf;

#define SURF_IMAGES \
    "cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png",\
    "stitching/a3.jpg"

PERF_TEST_P( surf, detect, testing::Values(SURF_IMAGES) )
{
    String filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);

    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame).time(90);
    SURF detector;
    vector<KeyPoint> points;

    TEST_CYCLE(100)
    {
        detector(frame, mask, points);
    }
}

PERF_TEST_P( surf, extract, testing::Values(SURF_IMAGES) )
{
    String filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);

    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame).time(90);

    SURF detector;
    vector<KeyPoint> points;
    vector<float> descriptors;
    detector(frame, mask, points);

    TEST_CYCLE(100)
    {
        detector(frame, mask, points, descriptors, true);
    }
}

PERF_TEST_P( surf, full, testing::Values(SURF_IMAGES) )
{
    String filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);

    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame).time(90);
    SURF detector;
    vector<KeyPoint> points;
    vector<float> descriptors;

    TEST_CYCLE(100)
    {
        detector(frame, mask, points, descriptors, false);
    }
}
