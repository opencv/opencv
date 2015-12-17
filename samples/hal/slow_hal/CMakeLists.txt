cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  endif()
endif()

add_library(slow_hal impl.cpp)
set(OPENCV_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
target_include_directories(slow_hal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OPENCV_SRC_DIR}/modules/core/include)
