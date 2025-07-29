if(NOT HAVE_OPENVX)
  message(STATUS "OpenVX is not available, disabling openvx-related HAL and stuff")
  return()
endif()

set(OPENCV_3P_OPENVX_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(hal)