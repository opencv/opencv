// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation


#ifndef OPENCV_GAPI_STEREO_TESTS_HPP
#define OPENCV_GAPI_STEREO_TESTS_HPP


#include <opencv2/gapi/stereo.hpp> // fore cv::gapi::StereoOutputFormat

#include "gapi_tests_common.hpp"
#include "gapi_parsers_tests_common.hpp"

namespace opencv_test
{

GAPI_TEST_FIXTURE(TestGAPIStereo, initMatsRandU, FIXTURE_API(cv::gapi::StereoOutputFormat, int, int, double, double, CompareMats), 6,
                                                             oF, numDisparities, blockSize, baseline,
                                                             focus, cmpF)

} // namespace opencv_test

#endif // OPENCV_GAPI_STEREO_TESTS_HPP
