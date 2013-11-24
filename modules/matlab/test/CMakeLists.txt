set(TEST_PROXY ${CMAKE_CURRENT_BINARY_DIR}/test.proxy)
file(REMOVE ${TEST_PROXY})

# generate
# call the python executable to generate the Matlab gateways
add_custom_command(
    OUTPUT ${TEST_PROXY}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/OpenCVTest.m ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/testsuite.m ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${TEST_PROXY}
    COMMENT "Building Matlab tests"
)

# targets
# opencv_matlab_sources --> opencv_matlab
add_custom_target(opencv_test_matlab ALL DEPENDS ${TEST_PROXY})
add_dependencies(opencv_test_matlab ${the_module})

# run the matlab test suite
add_test(opencv_test_matlab
    COMMAND ${MATLAB_BIN} "-nodisplay" "-r" "testsuite.m"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
