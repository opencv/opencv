if(NOT WITH_FASTCV OR NOT FASTCV_DIR)
  message(STATUS "FastCV is not available, disabling related HAL and stuff")
  return()
endif()

if(NOT ANDROID AND NOT UNIX)
  message(FATAL_ERROR "FastCV HAL supports Android and UNIX only!")
endif()

set(OPENCV_3P_FASTCV_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(hal)
