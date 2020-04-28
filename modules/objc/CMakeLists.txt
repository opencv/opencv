if(OPENCV_INITIAL_PASS AND APPLE_FRAMEWORK AND NOT (BUILD_opencv_objc STREQUAL "OFF"))
  # generator for Objective-C source code and documentation signatures
  add_subdirectory(generator)
endif()

#include(${CMAKE_CURRENT_SOURCE_DIR}/common.cmake)
