# CMake generated Testfile for 
# Source directory: /home/boney/opencvprog/opencv-source/modules/features2d
# Build directory: /home/boney/opencvprog/opencv-source/modules/features2d
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_features2d "/home/boney/opencvprog/opencv-source/bin/opencv_test_features2d" "--gtest_output=xml:opencv_test_features2d.xml")
set_tests_properties(opencv_test_features2d PROPERTIES  LABELS "Main;opencv_features2d;Accuracy" WORKING_DIRECTORY "/home/boney/opencvprog/opencv-source/test-reports/accuracy")
add_test(opencv_perf_features2d "/home/boney/opencvprog/opencv-source/bin/opencv_perf_features2d" "--gtest_output=xml:opencv_perf_features2d.xml")
set_tests_properties(opencv_perf_features2d PROPERTIES  LABELS "Main;opencv_features2d;Performance" WORKING_DIRECTORY "/home/boney/opencvprog/opencv-source/test-reports/performance")
add_test(opencv_sanity_features2d "/home/boney/opencvprog/opencv-source/bin/opencv_perf_features2d" "--gtest_output=xml:opencv_perf_features2d.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_features2d PROPERTIES  LABELS "Main;opencv_features2d;Sanity" WORKING_DIRECTORY "/home/boney/opencvprog/opencv-source/test-reports/sanity")
