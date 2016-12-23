# Binary branch name: ffmpeg/master_20161202
# Binaries were created for OpenCV: 594c136d1fcbb5816c57516e50f9cbeffbd90835
set(FFMPEG_BINARIES_COMMIT "2a19d0006415955c79431116e4634f04d5eb5a74")
set(FFMPEG_FILE_HASH_BIN32 "f081abd9d6ca7e425d340ce586f9c090")
set(FFMPEG_FILE_HASH_BIN64 "a423363a6eb76d362ca6c406c96c8db6")
set(FFMPEG_FILE_HASH_CMAKE "5346ae1854fc7aa569a722e85af480ec")

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
