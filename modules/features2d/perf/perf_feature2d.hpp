#ifndef __OPENCV_PERF_FEATURE2D_HPP__
#define __OPENCV_PERF_FEATURE2D_HPP__

#include "perf_precomp.hpp"

/* cofiguration for tests of detectors/descriptors. shared between ocl and cpu tests. */

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;
// detectors/descriptors configurations to test
#define DETECTORS_ONLY                                                                  \
    FAST_DEFAULT, FAST_20_TRUE_TYPE5_8, FAST_20_TRUE_TYPE7_12, FAST_20_TRUE_TYPE9_16,   \
    FAST_20_FALSE_TYPE5_8, FAST_20_FALSE_TYPE7_12, FAST_20_FALSE_TYPE9_16,              \
                                                                                        \
    AGAST_DEFAULT, AGAST_5_8, AGAST_7_12d, AGAST_7_12s, AGAST_OAST_9_16,                \
                                                                                        \
    MSER_DEFAULT

#define DETECTORS_EXTRACTORS                                                            \
    ORB_DEFAULT, ORB_1500_13_1,                                                         \
    AKAZE_DEFAULT, AKAZE_DESCRIPTOR_KAZE,                                               \
    BRISK_DEFAULT,                                                                      \
    KAZE_DEFAULT

#define CV_ENUM_EXPAND(name, ...) CV_ENUM(name, __VA_ARGS__)

enum Feature2DVals { DETECTORS_ONLY, DETECTORS_EXTRACTORS };
CV_ENUM_EXPAND(Feature2DType, DETECTORS_ONLY, DETECTORS_EXTRACTORS)

typedef std::tr1::tuple<Feature2DType, string> Feature2DType_String_t;
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
    case AGAST_DEFAULT:
        return AgastFeatureDetector::create();
    case AGAST_5_8:
        return AgastFeatureDetector::create(70, true, AgastFeatureDetector::AGAST_5_8);
    case AGAST_7_12d:
        return AgastFeatureDetector::create(70, true, AgastFeatureDetector::AGAST_7_12d);
    case AGAST_7_12s:
        return AgastFeatureDetector::create(70, true, AgastFeatureDetector::AGAST_7_12s);
    case AGAST_OAST_9_16:
        return AgastFeatureDetector::create(70, true, AgastFeatureDetector::OAST_9_16);
    case AKAZE_DEFAULT:
        return AKAZE::create();
    case AKAZE_DESCRIPTOR_KAZE:
        return AKAZE::create(AKAZE::DESCRIPTOR_KAZE);
    case BRISK_DEFAULT:
        return BRISK::create();
    case KAZE_DEFAULT:
        return KAZE::create();
    case MSER_DEFAULT:
        return MSER::create();
    default:
        return Ptr<Feature2D>();
    }
}

#endif // __OPENCV_PERF_FEATURE2D_HPP__
