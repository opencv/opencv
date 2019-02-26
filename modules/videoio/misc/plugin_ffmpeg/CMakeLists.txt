cmake_minimum_required(VERSION 3.5)

set(OpenCV_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../..")
set(WITH_FFMPEG ON)
set(OPENCV_FFMPEG_SKIP_BUILD_CHECK ON)
include("${OpenCV_SOURCE_DIR}/modules/videoio/cmake/plugin.cmake")
ocv_create_videoio_plugin("opencv_videoio_ffmpeg" "ocv.3rdparty.ffmpeg" "FFmpeg" "cap_ffmpeg.cpp")

message(STATUS "FFMPEG_libavcodec_VERSION=${FFMPEG_libavcodec_VERSION}")
message(STATUS "FFMPEG_libavformat_VERSION=${FFMPEG_libavformat_VERSION}")
message(STATUS "FFMPEG_libavutil_VERSION=${FFMPEG_libavutil_VERSION}")
message(STATUS "FFMPEG_libswscale_VERSION=${FFMPEG_libswscale_VERSION}")
message(STATUS "FFMPEG_libavresample_VERSION=${FFMPEG_libavresample_VERSION}")
