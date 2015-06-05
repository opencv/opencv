# Binary branch name: ffmpeg/master_20150703
# Binaries were created for OpenCV: e379ea6ed60b0caad4d4e3eea096e9d850cb8c86
set(FFMPEG_BINARIES_COMMIT "8aeefc4efe3215de89d8c7e114ae6f7a6091b8eb")
set(FFMPEG_FILE_HASH_BIN32 "89c783eee1c47bfc733f08334ec2e31c")
set(FFMPEG_FILE_HASH_BIN64 "35fe6ccdda6d7a04e9056b0d73b98e76")
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
