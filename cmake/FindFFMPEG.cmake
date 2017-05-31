set(HAVE_FFMPEG 1)
set(HAVE_FFMPEG_CODEC 1)
set(HAVE_FFMPEG_FORMAT 1)
set(HAVE_FFMPEG_UTIL 1)
set(HAVE_FFMPEG_SWSCALE 1)
set(HAVE_GENTOO_FFMPEG 1)

if(FFMPEG_LIBRARIES AND FFMPEG_INCLUDE_DIR)
  set(FFMPEG_FOUND TRUE)
else(FFMPEG_LIBRARIES AND FFMPEG_INCLUDE_DIR)
  find_package(PkgConfig)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(_FFMPEG_AVCODEC libavcodec)
    pkg_check_modules(_FFMPEG_AVFORMAT libavformat)
    pkg_check_modules(_FFMPEG_AVUTIL libavutil)
  endif(PKG_CONFIG_FOUND)
  find_path(FFMPEG_AVCODEC_INCLUDE_DIR
    NAMES libavcodec/avcodec.h
    PATHS ${_FFMPEG_AVCODEC_INCLUDE_DIRS} /usr/include/ /usr/local/include/ /opt/local/include
    PATH_SUFFIXES ffmpeg libav)
  find_library(FFMPEG_LIBAVFORMAT
    NAMES avformat
    PATHS ${_FFMPEG_AVFORMAT_LIBRARY_DIRS} /usr/lib /usr/local/lib /opt/local/lib 
    )
  find_library(FFMPEG_LIBAVCODEC
    NAMES avcodec
    PATHS ${_FFMPEG_AVCODEC_LIBRARY_DIRS} /usr/lib /usr/local/lib /opt/local/lib 
    )
  find_library(FFMPEG_LIBAVUTIL
    NAMES avutil
    PATHS ${_FFMPEG_avutil_LIBRARY_DIRS} /usr/lib /usr/local/lib /opt/local/lib 
    )
  if (FFMPEG_LIBAVCODEC AND FFMPEG_LIBAVFORMAT)
    set(FFMPEG_FOUND TRUE)
  endif()
  if (FFMPEG_FOUND)
    set(FFMPEG_INCLUDE_DIR ${FFMPEG_AVCODEC_INCLUDE_DIR})
    set(FFMPEG_LIBRARIES
      ${FFMPEG_LIBAVCODEC}
      ${FFMPEG_LIBAVFORMAT}
      ${FFMPEG_LIBAVUTIL}
      )
  endif(FFMPEG_FOUND)
  if (FFMPEG_FOUND)
    if (NOT FFMPEG_FIND_QUIETLY)
      message(STATUS "found ffmpeg or libav ${FFMPEG_LIBRARIES}, ${FFMPEG_INCLUDE_DIR}")
    endif (NOT FFMPEG_FIND_QUIETLY)
  else(FFMPEG_FOUND)
    if(FFMPEG_FIND_REQUIED)
      message(FATAL_ERROR "could not find ffmpeg or libav")
    endif(FFMPEG_FIND_REQUIRED)
  endif(FFMEPG_FOUND)
endif(FFMPEG_LIBRARIES AND FFMEPG_INCLUDE_DIR)

  
    # set(FFMPEG_FOUND 
# set(FFMPEG_INCLUDE_DIRS "/usr/include/libavcodec"
# /"
# set(ALIASOF_libavcodec_VERSION 55.18.102)
# set(ALIASOF_libavformat_VERSION 55.12.100)
# set(ALIASOF_libavutil_VERSION 52.38.100)
# set(ALIASOF_libswscale_VERSION 2.3.100)
