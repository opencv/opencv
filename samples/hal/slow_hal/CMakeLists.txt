cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)

add_library(slow_hal impl.cpp)
set_target_properties(slow_hal PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set(OPENCV_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
target_include_directories(slow_hal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OPENCV_SRC_DIR}/modules/core/include)

configure_file("impl.hpp" "${CMAKE_BINARY_DIR}/impl.hpp" COPYONLY)
configure_file("config.cmake" "${CMAKE_BINARY_DIR}/OpenCV_HALConfig.cmake" COPYONLY)
