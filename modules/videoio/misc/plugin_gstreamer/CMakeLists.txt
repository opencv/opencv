cmake_minimum_required(VERSION 3.5)

set(OpenCV_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../..")
set(WITH_GSTREAMER ON)
include("${OpenCV_SOURCE_DIR}/modules/videoio/cmake/plugin_standalone.cmake")
ocv_create_videoio_plugin("opencv_videoio_gstreamer" "ocv.3rdparty.gstreamer" "GStreamer" "cap_gstreamer.cpp")

message(STATUS "Using GStreamer: ${GSTREAMER_VERSION}")
