#ifndef __OPENCV_PERF_FEATURE2D_HPP__
#define __OPENCV_PERF_FEATURE2D_HPP__

#include "perf_precomp.hpp"

namespace opencv_test
{

/* configuration for tests of detectors/descriptors. shared between ocl and cpu tests. */

// detectors/descriptors configurations to test
#define DETECTORS_ONLY                                                                  \
    FAST_DEFAULT, FAST_20_TRUE_TYPE5_8, FAST_20_TRUE_TYPE7_12, FAST_20_TRUE_TYPE9_16,   \
    FAST_20_FALSE_TYPE5_8, FAST_20_FALSE_TYPE7_12, FAST_20_FALSE_TYPE9_16,              \
                                                                                        \
    MSER_DEFAULT

#define DETECTORS_EXTRACTORS                                                            \
    ORB_DEFAULT, ORB_1500_13_1,                                                         \
    SIFT_DEFAULT

#define CV_ENUM_EXPAND(name, ...) CV_ENUM(name, __VA_ARGS__)

enum Feature2DVals { DETECTORS_ONLY, DETECTORS_EXTRACTORS };
CV_ENUM_EXPAND(Feature2DType, DETECTORS_ONLY, DETECTORS_EXTRACTORS)

typedef tuple<Feature2DType, string> Feature2DType_String_t;
typedef perf::TestBaseWithParam<Feature2DType_String_t> feature2d;

#define TEST_IMAGES testing::Values(\
    "cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png",\
    "stitching/a3.png", \
    "stitching/s2.jpg")

static inline Ptr<Feature2D> getFeature2D(Feature2DType type)
{
    switch(type) {
    case ORB_DEFAULT:
        return ORB::create();
    case ORB_1500_13_1:
        return ORB::create(1500, 1.3f, 1);
    case FAST_DEFAULT:
        return FastFeatureDetector::create();
    case FAST_20_TRUE_TYPE5_8:
        return FastFeatureDetector::create(20, true, FastFeatureDetector::TYPE_5_8);
    case FAST_20_TRUE_TYPE7_12:
        return FastFeatureDetector::create(20, true, FastFeatureDetector::TYPE_7_12);
    case FAST_20_TRUE_TYPE9_16:
        return FastFeatureDetector::create(20, true, FastFeatureDetector::TYPE_9_16);
    case FAST_20_FALSE_TYPE5_8:
        return FastFeatureDetector::create(20, false, FastFeatureDetector::TYPE_5_8);
    case FAST_20_FALSE_TYPE7_12:
        return FastFeatureDetector::create(20, false, FastFeatureDetector::TYPE_7_12);
    case FAST_20_FALSE_TYPE9_16:
        return FastFeatureDetector::create(20, false, FastFeatureDetector::TYPE_9_16);
    case MSER_DEFAULT:
        return MSER::create();
    case SIFT_DEFAULT:
        return SIFT::create();
    default:
        return Ptr<Feature2D>();
    }
}

} // namespace

#endif // __OPENCV_PERF_FEATURE2D_HPP__
