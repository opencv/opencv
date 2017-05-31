# CMake generated Testfile for 
# Source directory: /Users/chihiro/Programs/opencv/opencv_contrib/modules/face
# Build directory: /Users/chihiro/Programs/opencv/opencv/release/modules/face
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_face "/Users/chihiro/Programs/opencv/opencv/release/bin/opencv_test_face" "--gtest_output=xml:opencv_test_face.xml")
set_tests_properties(opencv_test_face PROPERTIES  LABELS "Extra;opencv_face;Accuracy" WORKING_DIRECTORY "/Users/chihiro/Programs/opencv/opencv/release/test-reports/accuracy")
