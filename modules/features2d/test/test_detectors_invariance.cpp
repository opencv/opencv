// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"
#include "test_invariance_utils.hpp"

#include "test_detectors_invariance.impl.hpp"

namespace opencv_test { namespace {

const static std::string IMAGE_TSUKUBA = "features2d/tsukuba.png";
const static std::string IMAGE_BIKES = "detectors_descriptors_evaluation/images_datasets/bikes/img1.png";
#define Value(...) Values(make_tuple(__VA_ARGS__))

/*
 * Detector's rotation invariance check
 */

INSTANTIATE_TEST_CASE_P(SIFT, DetectorRotationInvariance,
                        Value(IMAGE_TSUKUBA, []() { return SIFT::create(); }, 0.45f, 0.70f));

INSTANTIATE_TEST_CASE_P(BRISK, DetectorRotationInvariance,
                        Value(IMAGE_TSUKUBA, []() { return BRISK::create(); }, 0.45f, 0.76f));

INSTANTIATE_TEST_CASE_P(ORB, DetectorRotationInvariance,
                        Value(IMAGE_TSUKUBA, []() { return ORB::create(); }, 0.5f, 0.76f));

INSTANTIATE_TEST_CASE_P(AKAZE, DetectorRotationInvariance,
                        Value(IMAGE_TSUKUBA, []() { return AKAZE::create(); }, 0.5f, 0.71f));

INSTANTIATE_TEST_CASE_P(AKAZE_DESCRIPTOR_KAZE, DetectorRotationInvariance,
                        Value(IMAGE_TSUKUBA, []() { return AKAZE::create(AKAZE::DESCRIPTOR_KAZE); }, 0.5f, 0.71f));

/*
 * Detector's scale invariance check
 */

INSTANTIATE_TEST_CASE_P(SIFT, DetectorScaleInvariance,
                        Value(IMAGE_BIKES, []() { return SIFT::create(0, 3, 0.09); }, 0.60f, 0.98f));

INSTANTIATE_TEST_CASE_P(BRISK, DetectorScaleInvariance,
                        Value(IMAGE_BIKES, []() { return BRISK::create(); }, 0.08f, 0.49f));

INSTANTIATE_TEST_CASE_P(ORB, DetectorScaleInvariance,
                        Value(IMAGE_BIKES, []() { return ORB::create(); }, 0.08f, 0.49f));

INSTANTIATE_TEST_CASE_P(KAZE, DetectorScaleInvariance,
                        Value(IMAGE_BIKES, []() { return KAZE::create(); }, 0.08f, 0.49f));

INSTANTIATE_TEST_CASE_P(AKAZE, DetectorScaleInvariance,
                        Value(IMAGE_BIKES, []() { return AKAZE::create(); }, 0.08f, 0.49f));

INSTANTIATE_TEST_CASE_P(AKAZE_DESCRIPTOR_KAZE, DetectorScaleInvariance,
                        Value(IMAGE_BIKES, []() { return AKAZE::create(AKAZE::DESCRIPTOR_KAZE); }, 0.08f, 0.49f));

}} // namespace
