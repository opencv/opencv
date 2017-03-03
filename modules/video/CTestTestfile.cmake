# CMake generated Testfile for 
# Source directory: /home/boney/opencvprog/opencv-source/modules/video
# Build directory: /home/boney/opencvprog/opencv-source/modules/video
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_video "/home/boney/opencvprog/opencv-source/bin/opencv_test_video" "--gtest_output=xml:opencv_test_video.xml")
set_tests_properties(opencv_test_video PROPERTIES  LABELS "Main;opencv_video;Accuracy" WORKING_DIRECTORY "/home/boney/opencvprog/opencv-source/test-reports/accuracy")
add_test(opencv_perf_video "/home/boney/opencvprog/opencv-source/bin/opencv_perf_video" "--gtest_output=xml:opencv_perf_video.xml")
set_tests_properties(opencv_perf_video PROPERTIES  LABELS "Main;opencv_video;Performance" WORKING_DIRECTORY "/home/boney/opencvprog/opencv-source/test-reports/performance")
add_test(opencv_sanity_video "/home/boney/opencvprog/opencv-source/bin/opencv_perf_video" "--gtest_output=xml:opencv_perf_video.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_video PROPERTIES  LABELS "Main;opencv_video;Sanity" WORKING_DIRECTORY "/home/boney/opencvprog/opencv-source/test-reports/sanity")
