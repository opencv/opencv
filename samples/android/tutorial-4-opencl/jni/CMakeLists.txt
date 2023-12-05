cmake_minimum_required(VERSION 3.6)

set(target JNIpart)
project(${target} CXX)

if (OPENCV_FROM_SDK)
  message(STATUS "Using OpenCV from local SDK")
  set(ANDROID_OPENCV_COMPONENTS "opencv_java" CACHE STRING "")
else()
  message(STATUS "Using OpenCV from AAR (Maven repo)")
  set(ANDROID_OPENCV_COMPONENTS "OpenCV::opencv_java${OPENCV_VERSION_MAJOR}" CACHE STRING "")
endif()

message(STATUS "ANDROID_ABI=${ANDROID_ABI}")
find_package(OpenCV REQUIRED COMPONENTS ${ANDROID_OPENCV_COMPONENTS})
find_package(OpenCL QUIET)

file(GLOB srcs *.cpp *.c)
file(GLOB hdrs *.hpp *.h)

include_directories("${CMAKE_CURRENT_LIST_DIR}")
add_library(${target} SHARED ${srcs} ${hdrs})

target_link_libraries(${target} ${ANDROID_OPENCV_COMPONENTS} -lGLESv2 -lEGL -llog)

if(OpenCL_FOUND)
  include_directories(${OpenCL_INCLUDE_DIRS})
  target_link_libraries(${OpenCL_LIBRARIES})
  add_definitions("-DOPENCL_FOUND")
elseif(DEFINED ANDROID_OPENCL_SDK)
  include_directories(${ANDROID_OPENCL_SDK}/include)
  link_directories(${ANDROID_OPENCL_SDK}/lib/${ANDROID_NDK_ABI_NAME})
  target_link_libraries(-lOpenCL)
  add_definitions("-DOPENCL_FOUND")
endif()
