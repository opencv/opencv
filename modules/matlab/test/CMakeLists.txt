# compile the test sources
file(GLOB SOURCE_FILES "*.cpp")
add_custom_target(opencv_test_matlab_sources ALL)
foreach(SOURCE_FILE ${SOURCE_FILES})
  get_filename_component(FILENAME ${SOURCE_FILE} NAME_WE)
  # compile the source file using mex
  add_custom_command(TARGET opencv_test_matlab_sources PRE_BUILD
      COMMAND echo ${MATLAB_MEX_SCRIPT} ${MEX_INCLUDES}
              ${SOURCE_FILE}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endforeach()

# copy the test files into the build dir
file(GLOB TEST_FILES "*.m")
foreach(TEST_FILE ${TEST_FILES})
  add_custom_command(TARGET opencv_test_matlab_sources PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E
      copy ${TEST_FILE} ${CMAKE_CURRENT_BINARY_DIR}
  )
endforeach()
    

# run the matlab test suite
add_test(opencv_test_matlab
    COMMAND ${MATLAB_BIN} "-nodisplay" "-r" "testsuite.m"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
