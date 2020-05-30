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
                        Value(IMAGE_TSUKUBA, SIFT::create(), 0.45f, 0.70f));

INSTANTIATE_TEST_CASE_P(BRISK, DetectorRotationInvariance,
                        Value(IMAGE_TSUKUBA, BRISK::create(), 0.45f, 0.76f));

INSTANTIATE_TEST_CASE_P(ORB, DetectorRotationInvariance,
                        Value(IMAGE_TSUKUBA, ORB::create(), 0.5f, 0.76f));

INSTANTIATE_TEST_CASE_P(AKAZE, DetectorRotationInvariance,
                        Value(IMAGE_TSUKUBA, AKAZE::create(), 0.5f, 0.71f));

INSTANTIATE_TEST_CASE_P(AKAZE_DESCRIPTOR_KAZE, DetectorRotationInvariance,
                        Value(IMAGE_TSUKUBA, AKAZE::create(AKAZE::DESCRIPTOR_KAZE), 0.5f, 0.71f));

/*
 * Detector's scale invariance check
 */

// TODO: Expected: (keyPointMatchesRatio) >= (minKeyPointMatchesRatio), actual: 0.596752 vs 0.69
INSTANTIATE_TEST_CASE_P(DISABLED_SIFT, DetectorScaleInvariance,
                        Value(IMAGE_BIKES, SIFT::create(), 0.69f, 0.98f));

INSTANTIATE_TEST_CASE_P(BRISK, DetectorScaleInvariance,
                        Value(IMAGE_BIKES, BRISK::create(), 0.08f, 0.49f));

INSTANTIATE_TEST_CASE_P(ORB, DetectorScaleInvariance,
                        Value(IMAGE_BIKES, ORB::create(), 0.08f, 0.49f));

INSTANTIATE_TEST_CASE_P(KAZE, DetectorScaleInvariance,
                        Value(IMAGE_BIKES, KAZE::create(), 0.08f, 0.49f));

INSTANTIATE_TEST_CASE_P(AKAZE, DetectorScaleInvariance,
                        Value(IMAGE_BIKES, AKAZE::create(), 0.08f, 0.49f));

INSTANTIATE_TEST_CASE_P(AKAZE_DESCRIPTOR_KAZE, DetectorScaleInvariance,
                        Value(IMAGE_BIKES, AKAZE::create(AKAZE::DESCRIPTOR_KAZE), 0.08f, 0.49f));

}} // namespace
