# Detect if we want to build samples with library binaries or not
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_LIST_DIR)
#
# BUILD CASE 1: Build samples with library sources
#


# ----------------------------------------------------------------------------
#  CMake file for samples. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------

add_subdirectory(c)
add_subdirectory(cpp)
add_subdirectory(gpu)
add_subdirectory(ocl)

if(ANDROID AND BUILD_ANDROID_EXAMPLES)
  add_subdirectory(android)
endif()


#
# END OF BUILD CASE 1: Build samples with library sources
#
else()
#
# BUILD CASE 2: Build samples with library binaries
#
cmake_minimum_required(VERSION 2.8)

project(samples C CXX)
option(BUILD_EXAMPLES "Build samples" ON)

find_package(OpenCV REQUIRED)

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

add_subdirectory(c)
add_subdirectory(cpp)
add_subdirectory(ocl)
add_subdirectory(gpu)

#
# END OF BUILD CASE 2: Build samples with library binaries
#
endif()