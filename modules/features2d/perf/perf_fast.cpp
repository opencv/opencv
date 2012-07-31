#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef perf::TestBaseWithParam<std::string> fast;

#define FAST_IMAGES \
    "cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png",\
    "stitching/a3.jpg"

PERF_TEST_P(fast, detectForORB, testing::Values(FAST_IMAGES))
{
    String filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);

    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;

    declare.in(frame);

    FastFeatureDetector fd(20, true, FastFeatureDetector::TYPE_5_8);
    vector<KeyPoint> points;

    TEST_CYCLE() fd.detect(frame, points);
    fd = FastFeatureDetector(20, true, FastFeatureDetector::TYPE_7_12);
    TEST_CYCLE() fd.detect(frame, points);
    fd = FastFeatureDetector(20, true, FastFeatureDetector::TYPE_9_16);
    TEST_CYCLE() fd.detect(frame, points);
}

