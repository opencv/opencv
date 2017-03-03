# CMake generated Testfile for 
# Source directory: /home/boney/opencvprog/opencv-source/modules/flann
# Build directory: /home/boney/opencvprog/opencv-source/modules/flann
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_flann "/home/boney/opencvprog/opencv-source/bin/opencv_test_flann" "--gtest_output=xml:opencv_test_flann.xml")
set_tests_properties(opencv_test_flann PROPERTIES  LABELS "Main;opencv_flann;Accuracy" WORKING_DIRECTORY "/home/boney/opencvprog/opencv-source/test-reports/accuracy")
