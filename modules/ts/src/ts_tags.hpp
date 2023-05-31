// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_TS_SRC_TAGS_HPP
#define OPENCV_TS_SRC_TAGS_HPP

// [all | test_tag] - (test_tag_skip - test_tag_enable) + test_tag_force

#define CV_TEST_TAGS_PARAMS \
    "{ test_tag           |         |run tests with specified 'tag' markers only (comma ',' separated list) }" \
    "{ test_tag_skip      |         |skip tests with 'tag' markers (comma ',' separated list) }" \
    "{ test_tag_enable    |         |don't skip tests with 'tag' markers (comma ',' separated list) }" \
    "{ test_tag_force     |         |force running of tests with 'tag' markers (comma ',' separated list) }" \
    "{ test_tag_print     | false   |print assigned tags for each test }" \

// TODO
//  "{ test_tag_file      |         |read test tags assignment }" \

namespace cvtest {

void activateTestTags(const cv::CommandLineParser& parser);

void testTagIncreaseSkipCount(const std::string& tag, bool isMain = true, bool appendSkipTests = false);

} // namespace

#endif // OPENCV_TS_SRC_TAGS_HPP
