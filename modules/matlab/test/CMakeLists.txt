# run the matlab test suite
add_test(opencv_matlab_test
    COMMAND ${MATLAB_BIN} "-nodisplay" "-r" "testsuite.m"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
