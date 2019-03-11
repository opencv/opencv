// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"
#include "test_invariance_utils.hpp"

#include "test_descriptors_invariance.impl.hpp"

namespace opencv_test { namespace {

const static std::string IMAGE_TSUKUBA = "features2d/tsukuba.png";
const static std::string IMAGE_BIKES = "detectors_descriptors_evaluation/images_datasets/bikes/img1.png";
#define Value(...) Values(make_tuple(__VA_ARGS__))

/*
 * Descriptors's rotation invariance check
 */

INSTANTIATE_TEST_CASE_P(BRISK, DescriptorRotationInvariance,
                        Value(IMAGE_TSUKUBA, BRISK::create(), BRISK::create(), 0.99f));

INSTANTIATE_TEST_CASE_P(ORB, DescriptorRotationInvariance,
                        Value(IMAGE_TSUKUBA, ORB::create(), ORB::create(), 0.99f));

INSTANTIATE_TEST_CASE_P(AKAZE, DescriptorRotationInvariance,
                        Value(IMAGE_TSUKUBA, AKAZE::create(), AKAZE::create(), 0.99f));

INSTANTIATE_TEST_CASE_P(AKAZE_DESCRIPTOR_KAZE, DescriptorRotationInvariance,
                        Value(IMAGE_TSUKUBA, AKAZE::create(AKAZE::DESCRIPTOR_KAZE), AKAZE::create(AKAZE::DESCRIPTOR_KAZE), 0.99f));

/*
 * Descriptor's scale invariance check
 */

INSTANTIATE_TEST_CASE_P(AKAZE, DescriptorScaleInvariance,
                        Value(IMAGE_BIKES, AKAZE::create(), AKAZE::create(), 0.6f));

INSTANTIATE_TEST_CASE_P(AKAZE_DESCRIPTOR_KAZE, DescriptorScaleInvariance,
                        Value(IMAGE_BIKES, AKAZE::create(AKAZE::DESCRIPTOR_KAZE), AKAZE::create(AKAZE::DESCRIPTOR_KAZE), 0.55f));

}} // namespace
