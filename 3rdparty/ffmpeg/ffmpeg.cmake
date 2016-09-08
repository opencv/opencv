# Binary branch name: ffmpeg/master_20160908
# Binaries were created for OpenCV: 11a65475d8d460a01c8818c5a2d0544ec49d7d68
set(FFMPEG_BINARIES_COMMIT "03835134465888981e066434dc95009e8328d4ea")
set(FFMPEG_FILE_HASH_BIN32 "32ba7790b0ac7a6dc66be91603637a7d")
set(FFMPEG_FILE_HASH_BIN64 "068ecaa459a5571e7909cff90999a420")
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
