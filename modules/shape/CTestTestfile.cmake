# CMake generated Testfile for 
# Source directory: /home/boney/opencvprog/opencv-source/modules/shape
# Build directory: /home/boney/opencvprog/opencv-source/modules/shape
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_shape "/home/boney/opencvprog/opencv-source/bin/opencv_test_shape" "--gtest_output=xml:opencv_test_shape.xml")
set_tests_properties(opencv_test_shape PROPERTIES  LABELS "Main;opencv_shape;Accuracy" WORKING_DIRECTORY "/home/boney/opencvprog/opencv-source/test-reports/accuracy")
