cmake_minimum_required(VERSION 3.5)
project(opencv_core_parallel_tbb CXX)

get_filename_component(OpenCV_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
include("${OpenCV_SOURCE_DIR}/cmake/OpenCVPluginStandalone.cmake")

# scan dependencies
set(WITH_TBB ON)
include("${OpenCV_SOURCE_DIR}/modules/core/cmake/parallel/init.cmake")

message(STATUS "TBB: ver ${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR} interface ${TBB_INTERFACE_VERSION}")
ocv_create_plugin(core "opencv_core_parallel_tbb" "ocv.3rdparty.tbb" "TBB" "src/parallel/parallel_tbb.cpp")
