# CMake generated Testfile for 
# Source directory: /Users/chihiro/Programs/opencv/opencv_contrib/modules/text
# Build directory: /Users/chihiro/Programs/opencv/opencv/release/modules/text
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_text "/Users/chihiro/Programs/opencv/opencv/release/bin/opencv_test_text" "--gtest_output=xml:opencv_test_text.xml")
set_tests_properties(opencv_test_text PROPERTIES  LABELS "Extra;opencv_text;Accuracy" WORKING_DIRECTORY "/Users/chihiro/Programs/opencv/opencv/release/test-reports/accuracy")
