#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;


typedef perf::TestBaseWithParam<std::string> fast;

#define FAST_IMAGES \
    "cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png",\
    "stitching/a3.jpg"

PERF_TEST_P( fast, detectForORB, testing::Values(FAST_IMAGES) )
{
    String filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);

    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame);

    FastFeatureDetector fd(20, true);
    vector<KeyPoint> points;

    TEST_CYCLE(100)
    {
        fd.detect(frame, points);
    }
}

