cmake_minimum_required(VERSION 3.5)
project(opencv_core_parallel_openmp CXX)

get_filename_component(OpenCV_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
include("${OpenCV_SOURCE_DIR}/cmake/OpenCVPluginStandalone.cmake")

# scan dependencies
set(WITH_OPENMP ON)
include("${OpenCV_SOURCE_DIR}/modules/core/cmake/parallel/init.cmake")

message(STATUS "OpenMP: ${OpenMP_CXX_VERSION}")
ocv_create_plugin(core "opencv_core_parallel_openmp" "ocv.3rdparty.openmp" "OPENMP" "src/parallel/parallel_openmp.cpp")
