# CMake generated Testfile for 
# Source directory: /Users/chihiro/Programs/opencv/opencv_contrib/modules/optflow
# Build directory: /Users/chihiro/Programs/opencv/opencv/release/modules/optflow
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_optflow "/Users/chihiro/Programs/opencv/opencv/release/bin/opencv_test_optflow" "--gtest_output=xml:opencv_test_optflow.xml")
set_tests_properties(opencv_test_optflow PROPERTIES  LABELS "Extra;opencv_optflow;Accuracy" WORKING_DIRECTORY "/Users/chihiro/Programs/opencv/opencv/release/test-reports/accuracy")
add_test(opencv_perf_optflow "/Users/chihiro/Programs/opencv/opencv/release/bin/opencv_perf_optflow" "--gtest_output=xml:opencv_perf_optflow.xml")
set_tests_properties(opencv_perf_optflow PROPERTIES  LABELS "Extra;opencv_optflow;Performance" WORKING_DIRECTORY "/Users/chihiro/Programs/opencv/opencv/release/test-reports/performance")
add_test(opencv_sanity_optflow "/Users/chihiro/Programs/opencv/opencv/release/bin/opencv_perf_optflow" "--gtest_output=xml:opencv_perf_optflow.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_optflow PROPERTIES  LABELS "Extra;opencv_optflow;Sanity" WORKING_DIRECTORY "/Users/chihiro/Programs/opencv/opencv/release/test-reports/sanity")
