# ----------------------------------------------------------------------------
#   Uninstall target, for "make uninstall"
# ----------------------------------------------------------------------------
CONFIGURE_FILE(
  "${OpenCV_SOURCE_DIR}/cmake/templates/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)

ADD_CUSTOM_TARGET(uninstall "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")
if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(uninstall PROPERTIES FOLDER "CMakeTargets")
endif()


# ----------------------------------------------------------------------------
# target building all OpenCV modules
# ----------------------------------------------------------------------------
add_custom_target(opencv_modules)
if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(opencv_modules PROPERTIES FOLDER "extra")
endif()


# ----------------------------------------------------------------------------
# targets building all tests
# ----------------------------------------------------------------------------
if(BUILD_TESTS)
  add_custom_target(opencv_tests)
  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(opencv_tests PROPERTIES FOLDER "extra")
  endif()
endif()
if(BUILD_PERF_TESTS)
  add_custom_target(opencv_perf_tests)
  if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(opencv_perf_tests PROPERTIES FOLDER "extra")
  endif()
endif()
