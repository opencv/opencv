# CMake generated Testfile for 
# Source directory: /Users/chihiro/Programs/opencv/opencv_contrib/modules/dnn
# Build directory: /Users/chihiro/Programs/opencv/opencv/release/modules/dnn
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_dnn "/Users/chihiro/Programs/opencv/opencv/release/bin/opencv_test_dnn" "--gtest_output=xml:opencv_test_dnn.xml")
set_tests_properties(opencv_test_dnn PROPERTIES  LABELS "Extra;opencv_dnn;Accuracy" WORKING_DIRECTORY "/Users/chihiro/Programs/opencv/opencv/release/test-reports/accuracy")
add_test(opencv_perf_dnn "/Users/chihiro/Programs/opencv/opencv/release/bin/opencv_perf_dnn" "--gtest_output=xml:opencv_perf_dnn.xml")
set_tests_properties(opencv_perf_dnn PROPERTIES  LABELS "Extra;opencv_dnn;Performance" WORKING_DIRECTORY "/Users/chihiro/Programs/opencv/opencv/release/test-reports/performance")
add_test(opencv_sanity_dnn "/Users/chihiro/Programs/opencv/opencv/release/bin/opencv_perf_dnn" "--gtest_output=xml:opencv_perf_dnn.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_dnn PROPERTIES  LABELS "Extra;opencv_dnn;Sanity" WORKING_DIRECTORY "/Users/chihiro/Programs/opencv/opencv/release/test-reports/sanity")
subdirs("../../3rdparty/protobuf")
