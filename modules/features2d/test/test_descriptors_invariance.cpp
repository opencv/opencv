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

INSTANTIATE_TEST_CASE_P(SIFT, DescriptorRotationInvariance,
                        Value(IMAGE_TSUKUBA, []() { return SIFT::create(); }, []() { return SIFT::create(); }, 0.98f));

INSTANTIATE_TEST_CASE_P(BRISK, DescriptorRotationInvariance,
                        Value(IMAGE_TSUKUBA, []() { return BRISK::create(); }, []() { return BRISK::create(); }, 0.99f));

INSTANTIATE_TEST_CASE_P(ORB, DescriptorRotationInvariance,
                        Value(IMAGE_TSUKUBA, []() { return ORB::create(); }, []() { return ORB::create(); }, 0.99f));

INSTANTIATE_TEST_CASE_P(AKAZE, DescriptorRotationInvariance,
                        Value(IMAGE_TSUKUBA, []() { return AKAZE::create(); }, []() { return AKAZE::create(); }, 0.99f));

INSTANTIATE_TEST_CASE_P(AKAZE_DESCRIPTOR_KAZE, DescriptorRotationInvariance,
                        Value(IMAGE_TSUKUBA, []() { return AKAZE::create(AKAZE::DESCRIPTOR_KAZE); }, []() { return AKAZE::create(AKAZE::DESCRIPTOR_KAZE); }, 0.99f));

/*
 * Descriptor's scale invariance check
 */

INSTANTIATE_TEST_CASE_P(SIFT, DescriptorScaleInvariance,
                        Value(IMAGE_BIKES, []() { return SIFT::create(0, 3, 0.09); }, []() { return SIFT::create(0, 3, 0.09); }, 0.78f));

INSTANTIATE_TEST_CASE_P(AKAZE, DescriptorScaleInvariance,
                        Value(IMAGE_BIKES, []() { return AKAZE::create(); }, []() { return AKAZE::create(); }, 0.6f));

INSTANTIATE_TEST_CASE_P(AKAZE_DESCRIPTOR_KAZE, DescriptorScaleInvariance,
                        Value(IMAGE_BIKES, []() { return AKAZE::create(AKAZE::DESCRIPTOR_KAZE); }, []() { return AKAZE::create(AKAZE::DESCRIPTOR_KAZE); }, 0.55f));

}} // namespace
