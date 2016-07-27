# Binary branch name: ffmpeg/master_20160715
# Binaries were created for OpenCV: 0e6aa189cb9a9642b0ae7983d301693516faad5d
set(FFMPEG_BINARIES_COMMIT "7eef9080d3271c7547d303fa839a62e1124ff1e6")
set(FFMPEG_FILE_HASH_BIN32 "3bb2a8388af90adf6c762210e696400d")
set(FFMPEG_FILE_HASH_BIN64 "ebcfc963f0a94f7e83d58d60eaf23849")
set(FFMPEG_FILE_HASH_CMAKE "f99941d10c1e87bf16b9055e8fc91ab2")

set(FFMPEG_DOWNLOAD_URL ${OPENCV_FFMPEG_URL};$ENV{OPENCV_FFMPEG_URL};https://raw.githubusercontent.com/opencv/opencv_3rdparty/${FFMPEG_BINARIES_COMMIT}/ffmpeg/)

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
