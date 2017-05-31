# CMake generated Testfile for 
# Source directory: /Users/chihiro/Programs/opencv/opencv_contrib/modules/line_descriptor
# Build directory: /Users/chihiro/Programs/opencv/opencv/release/modules/line_descriptor
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_line_descriptor "/Users/chihiro/Programs/opencv/opencv/release/bin/opencv_test_line_descriptor" "--gtest_output=xml:opencv_test_line_descriptor.xml")
set_tests_properties(opencv_test_line_descriptor PROPERTIES  LABELS "Extra;opencv_line_descriptor;Accuracy" WORKING_DIRECTORY "/Users/chihiro/Programs/opencv/opencv/release/test-reports/accuracy")
add_test(opencv_perf_line_descriptor "/Users/chihiro/Programs/opencv/opencv/release/bin/opencv_perf_line_descriptor" "--gtest_output=xml:opencv_perf_line_descriptor.xml")
set_tests_properties(opencv_perf_line_descriptor PROPERTIES  LABELS "Extra;opencv_line_descriptor;Performance" WORKING_DIRECTORY "/Users/chihiro/Programs/opencv/opencv/release/test-reports/performance")
add_test(opencv_sanity_line_descriptor "/Users/chihiro/Programs/opencv/opencv/release/bin/opencv_perf_line_descriptor" "--gtest_output=xml:opencv_perf_line_descriptor.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_line_descriptor PROPERTIES  LABELS "Extra;opencv_line_descriptor;Sanity" WORKING_DIRECTORY "/Users/chihiro/Programs/opencv/opencv/release/test-reports/sanity")
