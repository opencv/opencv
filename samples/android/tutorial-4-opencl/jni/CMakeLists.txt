cmake_minimum_required(VERSION 3.6)

set(target JNIpart)
project(${target} CXX)

if (OPENCV_FROM_SDK)
  set(ANDROID_OPENCV_COMPONENTS "opencv_java" CACHE STRING "")
else()
  set(ANDROID_OPENCV_COMPONENTS "OpenCV::opencv_java${OPENCV_VERSION_MAJOR}" CACHE STRING "")
endif()

message(STATUS "ANDROID_ABI=${ANDROID_ABI}")
find_package(OpenCV REQUIRED COMPONENTS ${ANDROID_OPENCV_COMPONENTS})

file(GLOB srcs *.cpp *.c)
file(GLOB hdrs *.hpp *.h)

include_directories("${CMAKE_CURRENT_LIST_DIR}")
add_library(${target} SHARED ${srcs} ${hdrs})
target_link_libraries(${target} ${ANDROID_OPENCV_COMPONENTS} -lGLESv2 -lEGL -lOpenCL)
