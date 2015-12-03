cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  endif()
endif()

add_library(simple_hal simple.cpp)
set(OPENCV_HAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
target_include_directories(simple_hal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OPENCV_HAL_DIR}/include)
