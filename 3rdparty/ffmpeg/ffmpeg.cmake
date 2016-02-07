# Binary branch name: ffmpeg/master_20160207
# Binaries were created for OpenCV: e04f3720833502f776fd6aea630f0bfe826b79e6
set(FFMPEG_BINARIES_COMMIT "44876e8bc67764e723cd93cccf5429a36071d596")
set(FFMPEG_FILE_HASH_BIN32 "dba0016f4a1a83c4e03ea19de35201ea")
set(FFMPEG_FILE_HASH_BIN64 "0c1cd92e89a266a4a8882ea91720a037")
set(FFMPEG_FILE_HASH_CMAKE "8606f947a780071f8fcce8cbf39ceef5")

set(FFMPEG_DOWNLOAD_URL ${OPENCV_FFMPEG_URL};$ENV{OPENCV_FFMPEG_URL};https://raw.githubusercontent.com/Itseez/opencv_3rdparty/${FFMPEG_BINARIES_COMMIT}/ffmpeg/)

ocv_download(PACKAGE opencv_ffmpeg.dll
             HASH ${FFMPEG_FILE_HASH_BIN32}
             URL ${FFMPEG_DOWNLOAD_URL}
             DESTINATION_DIR ${CMAKE_CURRENT_LIST_DIR})

ocv_download(PACKAGE opencv_ffmpeg_64.dll
             HASH ${FFMPEG_FILE_HASH_BIN64}
             URL ${FFMPEG_DOWNLOAD_URL}
             DESTINATION_DIR ${CMAKE_CURRENT_LIST_DIR})

ocv_download(PACKAGE ffmpeg_version.cmake
             HASH ${FFMPEG_FILE_HASH_CMAKE}
             URL ${FFMPEG_DOWNLOAD_URL}
             DESTINATION_DIR ${CMAKE_CURRENT_LIST_DIR})

include(${CMAKE_CURRENT_LIST_DIR}/ffmpeg_version.cmake)
