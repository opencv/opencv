# CMake generated Testfile for 
# Source directory: /Users/chihiro/Programs/opencv/opencv_contrib/modules/structured_light
# Build directory: /Users/chihiro/Programs/opencv/opencv/release/modules/structured_light
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_structured_light "/Users/chihiro/Programs/opencv/opencv/release/bin/opencv_test_structured_light" "--gtest_output=xml:opencv_test_structured_light.xml")
set_tests_properties(opencv_test_structured_light PROPERTIES  LABELS "Extra;opencv_structured_light;Accuracy" WORKING_DIRECTORY "/Users/chihiro/Programs/opencv/opencv/release/test-reports/accuracy")
