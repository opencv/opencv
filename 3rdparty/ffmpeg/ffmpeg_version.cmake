set(FFMPEG_DOWNLOAD_URL ${OPENCV_FFMPEG_URL};$ENV{OPENCV_FFMPEG_URL};http://build.opencv.org/buildbot/storage/exports/ffmpeg_20150622/)

ocv_download(PACKAGE opencv_ffmpeg.dll
             HASH "02a6637ed4b26b830bc58c18e21ea672"
             URL ${FFMPEG_DOWNLOAD_URL}
             DESTINATION_DIR ${CMAKE_CURRENT_LIST_DIR})

ocv_download(PACKAGE opencv_ffmpeg_64.dll
             HASH "ea83e3d6409899f8590de3edbdccc60d"
             URL ${FFMPEG_DOWNLOAD_URL}
             DESTINATION_DIR ${CMAKE_CURRENT_LIST_DIR})

set(HAVE_FFMPEG 1)
set(HAVE_FFMPEG_CODEC 1)
set(HAVE_FFMPEG_FORMAT 1)
set(HAVE_FFMPEG_UTIL 1)
set(HAVE_FFMPEG_SWSCALE 1)
set(HAVE_FFMPEG_RESAMPLE 0)
set(HAVE_GENTOO_FFMPEG 1)

set(ALIASOF_libavcodec_VERSION 56.26.100)
set(ALIASOF_libavformat_VERSION 56.25.101)
set(ALIASOF_libavutil_VERSION 54.20.100)
set(ALIASOF_libswscale_VERSION 3.1.101)
set(ALIASOF_libavresample_VERSION 2.1.0)
