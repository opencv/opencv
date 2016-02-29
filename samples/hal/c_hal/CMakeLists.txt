cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUC OR CV_ICC)
    set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  endif()
endif()

add_library(c_hal impl.c)
set(OPENCV_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
target_include_directories(c_hal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OPENCV_SRC_DIR}/modules/core/include)
