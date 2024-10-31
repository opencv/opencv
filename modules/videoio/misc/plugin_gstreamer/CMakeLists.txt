cmake_minimum_required(VERSION 3.13)

get_filename_component(OpenCV_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
include("${OpenCV_SOURCE_DIR}/cmake/OpenCVPluginStandalone.cmake")

# scan dependencies
set(WITH_GSTREAMER ON)
include("${OpenCV_SOURCE_DIR}/modules/videoio/cmake/init.cmake")

set(OPENCV_PLUGIN_DEPS core imgproc imgcodecs)
ocv_create_plugin(videoio "opencv_videoio_gstreamer" "ocv.3rdparty.gstreamer" "GStreamer" "src/cap_gstreamer.cpp")

message(STATUS "Using GStreamer: ${GSTREAMER_VERSION}")
