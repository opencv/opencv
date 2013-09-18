set(test_deps opencv_cudev opencv_core opencv_imgproc opencv_highgui opencv_ts ${OPENCV_MODULE_opencv_ts_DEPS})

ocv_check_dependencies(${test_deps})

if(OCV_DEPENDENCIES_FOUND)
  set(the_target "opencv_test_${name}")

  ocv_module_include_directories("${test_deps}" "${the_module}")

  file(GLOB test_srcs "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
  file(GLOB test_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
  source_group("Src" FILES ${test_srcs})
  source_group("Include" FILES ${test_hdrs})
  set(OPENCV_TEST_${the_module}_SOURCES ${test_srcs} ${test_hdrs})

  foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    # we remove /EHa as it generates warnings under windows
    string(REPLACE "/EHa" "" ${var} "${${var}}")

    # we remove -ggdb3 flag as it leads to preprocessor errors when compiling CUDA files (CUDA 4.1)
    string(REPLACE "-ggdb3" "" ${var} "${${var}}")

    # we remove -Wsign-promo as it generates warnings under linux
    string(REPLACE "-Wsign-promo" "" ${var} "${${var}}")

    # we remove -fvisibility-inlines-hidden because it's used for C++ compiler
    # but NVCC uses C compiler by default
    string(REPLACE "-fvisibility-inlines-hidden" "" ${var} "${${var}}")
  endforeach()

  CUDA_ADD_EXECUTABLE(${the_target} ${OPENCV_TEST_${the_module}_SOURCES})
  target_link_libraries(${the_target} ${test_deps} ${OPENCV_LINKER_LIBS} ${CUDA_LIBRARIES})
  add_dependencies(opencv_tests ${the_target})

  # Additional target properties
  set_target_properties(${the_target} PROPERTIES
    DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
    RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}"
  )

  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(${the_target} PROPERTIES FOLDER "tests accuracy")
  endif()

  enable_testing()
  get_target_property(LOC ${the_target} LOCATION)
  add_test(${the_target} "${LOC}")
endif()
