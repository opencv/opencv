// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef perf::TestBaseWithParam<std::string> SIFT_detect;
typedef perf::TestBaseWithParam<std::string> SIFT_extract;
typedef perf::TestBaseWithParam<std::string> SIFT_full;

#define SIFT_IMAGES \
    "cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png",\
    "stitching/a3.png"

PERF_TEST_P_(SIFT_detect, SIFT)
{
    string filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame.empty()) << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame).time(90);
    Ptr<SIFT> detector = SIFT::create();
    vector<KeyPoint> points;

    PERF_SAMPLE_BEGIN();
        detector->detect(frame, points, mask);
    PERF_SAMPLE_END();

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(SIFT_extract, SIFT)
{
    string filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame.empty()) << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame).time(90);

    Ptr<SIFT> detector = SIFT::create();
    vector<KeyPoint> points;
    Mat descriptors;
    detector->detect(frame, points, mask);

    PERF_SAMPLE_BEGIN();
        detector->compute(frame, points, descriptors);
    PERF_SAMPLE_END();

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(SIFT_full, SIFT)
{
    string filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame.empty()) << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame).time(90);
    Ptr<SIFT> detector = SIFT::create();
    vector<KeyPoint> points;
    Mat descriptors;

    PERF_SAMPLE_BEGIN();
        detector->detectAndCompute(frame, mask, points, descriptors, false);
    PERF_SAMPLE_END();

    SANITY_CHECK_NOTHING();
}


INSTANTIATE_TEST_CASE_P(/*nothing*/, SIFT_detect,
    testing::Values(SIFT_IMAGES)
);
INSTANTIATE_TEST_CASE_P(/*nothing*/, SIFT_extract,
    testing::Values(SIFT_IMAGES)
);
INSTANTIATE_TEST_CASE_P(/*nothing*/, SIFT_full,
    testing::Values(SIFT_IMAGES)
);

}} // namespace
