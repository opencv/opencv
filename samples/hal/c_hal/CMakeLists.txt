cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)

add_library(c_hal impl.c)
set_target_properties(c_hal PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set(OPENCV_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
target_include_directories(c_hal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OPENCV_SRC_DIR}/modules/core/include)

configure_file("impl.h" "${CMAKE_BINARY_DIR}/impl.h" COPYONLY)
configure_file("config.cmake" "${CMAKE_BINARY_DIR}/OpenCV_HALConfig.cmake" COPYONLY)
