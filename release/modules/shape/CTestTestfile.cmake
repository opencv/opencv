# CMake generated Testfile for 
# Source directory: /Users/chihiro/Programs/opencv/opencv/modules/shape
# Build directory: /Users/chihiro/Programs/opencv/opencv/release/modules/shape
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_shape "/Users/chihiro/Programs/opencv/opencv/release/bin/opencv_test_shape" "--gtest_output=xml:opencv_test_shape.xml")
set_tests_properties(opencv_test_shape PROPERTIES  LABELS "Main;opencv_shape;Accuracy" WORKING_DIRECTORY "/Users/chihiro/Programs/opencv/opencv/release/test-reports/accuracy")
