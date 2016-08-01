cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)

set(PROJECT_NAME "slow_hal")
set(HAL_LIB_NAME "slow_hal")

add_library(${HAL_LIB_NAME} impl.cpp)
set_target_properties(${HAL_LIB_NAME} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set(OPENCV_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
target_include_directories(${HAL_LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OPENCV_SRC_DIR}/modules/core/include)

set(OpenCV_HAL_FOUND TRUE)
set(OpenCV_HAL_VERSION 0.0.1)
set(OpenCV_HAL_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/lib${HAL_LIB_NAME}.a)
set(OpenCV_HAL_HEADERS "impl.hpp")
set(OpenCV_HAL_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR})

configure_file("impl.hpp" "${CMAKE_BINARY_DIR}/impl.hpp" COPYONLY)
configure_file("config.cmake" "${CMAKE_BINARY_DIR}/OpenCV_HALConfig.cmake")
