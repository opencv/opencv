# CMake generated Testfile for 
# Source directory: /home/boney/opencvprog/opencv-source/modules/videoio
# Build directory: /home/boney/opencvprog/opencv-source/modules/videoio
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_videoio "/home/boney/opencvprog/opencv-source/bin/opencv_test_videoio" "--gtest_output=xml:opencv_test_videoio.xml")
set_tests_properties(opencv_test_videoio PROPERTIES  LABELS "Main;opencv_videoio;Accuracy" WORKING_DIRECTORY "/home/boney/opencvprog/opencv-source/test-reports/accuracy")
add_test(opencv_perf_videoio "/home/boney/opencvprog/opencv-source/bin/opencv_perf_videoio" "--gtest_output=xml:opencv_perf_videoio.xml")
set_tests_properties(opencv_perf_videoio PROPERTIES  LABELS "Main;opencv_videoio;Performance" WORKING_DIRECTORY "/home/boney/opencvprog/opencv-source/test-reports/performance")
add_test(opencv_sanity_videoio "/home/boney/opencvprog/opencv-source/bin/opencv_perf_videoio" "--gtest_output=xml:opencv_perf_videoio.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_videoio PROPERTIES  LABELS "Main;opencv_videoio;Sanity" WORKING_DIRECTORY "/home/boney/opencvprog/opencv-source/test-reports/sanity")
