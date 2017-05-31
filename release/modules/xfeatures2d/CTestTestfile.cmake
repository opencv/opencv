# CMake generated Testfile for 
# Source directory: /Users/chihiro/Programs/opencv/opencv_contrib/modules/xfeatures2d
# Build directory: /Users/chihiro/Programs/opencv/opencv/release/modules/xfeatures2d
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_xfeatures2d "/Users/chihiro/Programs/opencv/opencv/release/bin/opencv_test_xfeatures2d" "--gtest_output=xml:opencv_test_xfeatures2d.xml")
set_tests_properties(opencv_test_xfeatures2d PROPERTIES  LABELS "Extra;opencv_xfeatures2d;Accuracy" WORKING_DIRECTORY "/Users/chihiro/Programs/opencv/opencv/release/test-reports/accuracy")
add_test(opencv_perf_xfeatures2d "/Users/chihiro/Programs/opencv/opencv/release/bin/opencv_perf_xfeatures2d" "--gtest_output=xml:opencv_perf_xfeatures2d.xml")
set_tests_properties(opencv_perf_xfeatures2d PROPERTIES  LABELS "Extra;opencv_xfeatures2d;Performance" WORKING_DIRECTORY "/Users/chihiro/Programs/opencv/opencv/release/test-reports/performance")
add_test(opencv_sanity_xfeatures2d "/Users/chihiro/Programs/opencv/opencv/release/bin/opencv_perf_xfeatures2d" "--gtest_output=xml:opencv_perf_xfeatures2d.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_xfeatures2d PROPERTIES  LABELS "Extra;opencv_xfeatures2d;Sanity" WORKING_DIRECTORY "/Users/chihiro/Programs/opencv/opencv/release/test-reports/sanity")
