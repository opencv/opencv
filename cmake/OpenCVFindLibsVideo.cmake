# ----------------------------------------------------------------------------
#  Detect 3rd-party video IO libraries
# ----------------------------------------------------------------------------

# --- GStreamer ---
ocv_clear_vars(HAVE_GSTREAMER)
if(WITH_GSTREAMER)
  CHECK_MODULE(gstreamer-base-0.10 HAVE_GSTREAMER)
  if(HAVE_GSTREAMER)
    CHECK_MODULE(gstreamer-app-0.10 HAVE_GSTREAMER)
  endif()
  if(HAVE_GSTREAMER)
    CHECK_MODULE(gstreamer-video-0.10 HAVE_GSTREAMER)
  endif()
endif(WITH_GSTREAMER)

# --- unicap ---
ocv_clear_vars(HAVE_UNICAP)
if(WITH_UNICAP)
  CHECK_MODULE(libunicap HAVE_UNICAP_)
  CHECK_MODULE(libucil HAVE_UNICAP_UCIL)
  if(HAVE_UNICAP_ AND HAVE_UNICAP_UCIL)
    set(HAVE_UNICAP TRUE)
  endif()
endif(WITH_UNICAP)

# --- PvApi ---
ocv_clear_vars(HAVE_PVAPI)
if(WITH_PVAPI)
  find_path(PVAPI_INCLUDE_PATH "PvApi.h"
            PATHS /usr/local /opt /usr ENV ProgramFiles ENV ProgramW6432
            PATH_SUFFIXES include "Allied Vision Technologies/GigESDK/inc-pc" "AVT GigE SDK/inc-pc" "GigESDK/inc-pc"
            DOC "The path to PvAPI header")

  if(PVAPI_INCLUDE_PATH)
    if(X86 AND NOT WIN32)
      set(PVAPI_SDK_SUBDIR x86)
    elseif(X86_64)
      set(PVAPI_SDK_SUBDIR x64)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES arm)
      set(PVAPI_SDK_SUBDIR arm)
    endif()

    get_filename_component(_PVAPI_LIBRARY "${PVAPI_INCLUDE_PATH}/../lib-pc" ABSOLUTE)
    if(PVAPI_SDK_SUBDIR)
      set(_PVAPI_LIBRARY "${_PVAPI_LIBRARY}/${PVAPI_SDK_SUBDIR}")
    endif()
    if(NOT WIN32 AND CMAKE_COMPILER_IS_GNUCXX)
      set(_PVAPI_LIBRARY "${_PVAPI_LIBRARY}/${CMAKE_OPENCV_GCC_VERSION_MAJOR}.${CMAKE_OPENCV_GCC_VERSION_MINOR}")
    endif()

    set(PVAPI_LIBRARY "${_PVAPI_LIBRARY}/${CMAKE_STATIC_LIBRARY_PREFIX}PvAPI${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE PATH "The PvAPI library")
    if(EXISTS "${PVAPI_LIBRARY}")
      set(HAVE_PVAPI TRUE)
    endif()
  endif(PVAPI_INCLUDE_PATH)
endif(WITH_PVAPI)

# --- Dc1394 ---
ocv_clear_vars(HAVE_DC1394 HAVE_DC1394_2)
if(WITH_1394)
  CHECK_MODULE(libdc1394-2 HAVE_DC1394_2)
  if(NOT HAVE_DC1394_2)
    CHECK_MODULE(libdc1394 HAVE_DC1394)
  endif()
endif(WITH_1394)

# --- xine ---
ocv_clear_vars(HAVE_XINE)
if(WITH_XINE)
  CHECK_MODULE(libxine HAVE_XINE)
endif(WITH_XINE)

# --- V4L ---
ocv_clear_vars(HAVE_LIBV4L HAVE_CAMV4L HAVE_CAMV4L2)
if(WITH_V4L)
  CHECK_MODULE(libv4l1 HAVE_LIBV4L)
  CHECK_INCLUDE_FILE(linux/videodev.h HAVE_CAMV4L)
  CHECK_INCLUDE_FILE(linux/videodev2.h HAVE_CAMV4L2)
endif(WITH_V4L)

# --- OpenNI ---
ocv_clear_vars(HAVE_OPENNI HAVE_OPENNI_PRIME_SENSOR_MODULE)
if(WITH_OPENNI)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVFindOpenNI.cmake")
endif(WITH_OPENNI)

# --- XIMEA ---
ocv_clear_vars(HAVE_XIMEA)
if(WITH_XIMEA)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVFindXimea.cmake")
  if(XIMEA_FOUND)
    set(HAVE_XIMEA TRUE)
  endif()
endif(WITH_XIMEA)

# --- FFMPEG ---
ocv_clear_vars(HAVE_FFMPEG HAVE_FFMPEG_CODEC HAVE_FFMPEG_FORMAT HAVE_FFMPEG_UTIL HAVE_FFMPEG_SWSCALE HAVE_GENTOO_FFMPEG HAVE_FFMPEG_FFMPEG)
if(WITH_FFMPEG)
  if(WIN32)
    include("${OpenCV_SOURCE_DIR}/3rdparty/ffmpeg/ffmpeg_version.cmake" REQUIRED)
  elseif(UNIX)
    CHECK_MODULE(libavcodec HAVE_FFMPEG_CODEC)
    CHECK_MODULE(libavformat HAVE_FFMPEG_FORMAT)
    CHECK_MODULE(libavutil HAVE_FFMPEG_UTIL)
    CHECK_MODULE(libswscale HAVE_FFMPEG_SWSCALE)

    CHECK_INCLUDE_FILE(libavformat/avformat.h HAVE_GENTOO_FFMPEG)
    CHECK_INCLUDE_FILE(ffmpeg/avformat.h HAVE_FFMPEG_FFMPEG)
    if(NOT HAVE_GENTOO_FFMPEG AND NOT HAVE_FFMPEG_FFMPEG)
      if(EXISTS /usr/include/ffmpeg/libavformat/avformat.h OR HAVE_FFMPEG_SWSCALE)
        set(HAVE_GENTOO_FFMPEG TRUE)
      endif()
    endif()
    if(HAVE_FFMPEG_CODEC AND HAVE_FFMPEG_FORMAT AND HAVE_FFMPEG_UTIL AND HAVE_FFMPEG_SWSCALE)
      set(HAVE_FFMPEG TRUE)
    endif()

    if(HAVE_FFMPEG)
      # Find the bzip2 library because it is required on some systems
      FIND_LIBRARY(BZIP2_LIBRARIES NAMES bz2 bzip2)
      if(NOT BZIP2_LIBRARIES)
        # Do an other trial
        FIND_FILE(BZIP2_LIBRARIES NAMES libbz2.so.1 PATHS /lib)
      endif()
    endif(HAVE_FFMPEG)
  endif()

  if(APPLE)
    find_path(FFMPEG_INCLUDE_DIR "libavformat/avformat.h"
              PATHS /usr/local /usr /opt
              PATH_SUFFIXES include
              DOC "The path to FFMPEG headers")
    if(FFMPEG_INCLUDE_DIR)
      set(HAVE_GENTOO_FFMPEG TRUE)
      set(FFMPEG_LIB_DIR "${FFMPEG_INCLUDE_DIR}/../lib" CACHE PATH "Full path of FFMPEG library directory")
      if(EXISTS "${FFMPEG_LIB_DIR}/libavcodec.a")
        set(HAVE_FFMPEG_CODEC 1)
        set(ALIASOF_libavcodec_VERSION "Unknown")
        if(EXISTS "${FFMPEG_LIB_DIR}/libavformat.a")
          set(HAVE_FFMPEG_FORMAT 1)
          set(ALIASOF_libavformat_VERSION "Unknown")
          if(EXISTS "${FFMPEG_LIB_DIR}/libavutil.a")
            set(HAVE_FFMPEG_UTIL 1)
            set(ALIASOF_libavutil_VERSION "Unknown")
            if(EXISTS "${FFMPEG_LIB_DIR}/libswscale.a")
              set(HAVE_FFMPEG_SWSCALE 1)
              set(ALIASOF_libswscale_VERSION "Unknown")
              set(HAVE_FFMPEG 1)
            endif()
          endif()
        endif()
      endif()
    endif(FFMPEG_INCLUDE_DIR)
    if(HAVE_FFMPEG)
      set(HIGHGUI_LIBRARIES ${HIGHGUI_LIBRARIES} "${FFMPEG_LIB_DIR}/libavcodec.a"
          "${FFMPEG_LIB_DIR}/libavformat.a" "${FFMPEG_LIB_DIR}/libavutil.a"
          "${FFMPEG_LIB_DIR}/libswscale.a")
      ocv_include_directories(${FFMPEG_INCLUDE_DIR})
    endif()
  endif(APPLE)
endif(WITH_FFMPEG)

# --- VideoInput ---
if(WITH_VIDEOINPUT)
  # always have VideoInput on Windows
  set(HAVE_VIDEOINPUT 1)
endif(WITH_VIDEOINPUT)

# --- Extra HighGUI libs on Windows ---
if(WIN32)
  list(APPEND HIGHGUI_LIBRARIES comctl32 gdi32 ole32)
  if(MSVC)
    list(APPEND HIGHGUI_LIBRARIES vfw32)
  elseif(MINGW64)
    list(APPEND HIGHGUI_LIBRARIES msvfw32 avifil32 avicap32 winmm)
  elseif(MINGW)
    list(APPEND HIGHGUI_LIBRARIES vfw32 winmm)
  endif()
endif(WIN32)
