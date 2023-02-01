# ----------------------------------------------------------------------------
#  Detect 3rd-party video IO libraries
# ----------------------------------------------------------------------------

ocv_clear_vars(HAVE_VFW)
if(WITH_VFW)
  try_compile(HAVE_VFW
    "${OpenCV_BINARY_DIR}"
    "${OpenCV_SOURCE_DIR}/cmake/checks/vfwtest.cpp"
    CMAKE_FLAGS "-DLINK_LIBRARIES:STRING=vfw32")
endif(WITH_VFW)

# --- GStreamer ---
ocv_clear_vars(HAVE_GSTREAMER)
# try to find gstreamer 1.x first if 0.10 was not requested
if(WITH_GSTREAMER AND NOT WITH_GSTREAMER_0_10)
  if(WIN32)
    SET(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_LIST_DIR}")
    FIND_PACKAGE(GstreamerWindows)
    IF(GSTREAMER_gstbase_LIBRARY AND GSTREAMER_gstvideo_LIBRARY AND GSTREAMER_gstapp_LIBRARY AND GSTREAMER_gstpbutils_LIBRARY AND GSTREAMER_gstriff_LIBRARY)
      set(HAVE_GSTREAMER TRUE)
      set(GSTREAMER_BASE_VERSION 1.0)
      set(GSTREAMER_VIDEO_VERSION 1.0)
      set(GSTREAMER_APP_VERSION 1.0)
      set(GSTREAMER_RIFF_VERSION 1.0)
      set(GSTREAMER_PBUTILS_VERSION 1.0)
    ENDIF(GSTREAMER_gstbase_LIBRARY AND GSTREAMER_gstvideo_LIBRARY AND GSTREAMER_gstapp_LIBRARY AND GSTREAMER_gstpbutils_LIBRARY AND GSTREAMER_gstriff_LIBRARY)

  else(WIN32)
    ocv_check_modules(GSTREAMER-1.0 gstreamer-base-1.0 gstreamer-video-1.0 gstreamer-app-1.0 gstreamer-riff-1.0 gstreamer-pbutils-1.0)
    if(HAVE_GSTREAMER-1.0)
      set(HAVE_GSTREAMER TRUE)
      ocv_append_build_options(VIDEOIO GSTREAMER-1.0)
      set(GSTREAMER_BASE_VERSION ${GSTREAMER-1.0_gstreamer-base-1.0_VERSION})
      set(GSTREAMER_VIDEO_VERSION ${GSTREAMER-1.0_gstreamer-video-1.0_VERSION})
      set(GSTREAMER_APP_VERSION ${GSTREAMER-1.0_gstreamer-app-1.0_VERSION})
      set(GSTREAMER_RIFF_VERSION ${GSTREAMER-1.0_gstreamer-riff-1.0_VERSION})
      set(GSTREAMER_PBUTILS_VERSION ${GSTREAMER-1.0_gstreamer-pbutils-1.0_VERSION})
    endif()
  endif(WIN32)
endif()

# if gstreamer 1.x was not found, or we specified we wanted 0.10, try to find it
if(WITH_GSTREAMER AND NOT HAVE_GSTREAMER OR WITH_GSTREAMER_0_10)
  ocv_check_modules(GSTREAMER-0.10 gstreamer-base-0.10 gstreamer-video-0.10 gstreamer-app-0.10 gstreamer-riff-0.10 gstreamer-pbutils-0.10)
  if(HAVE_GSTREAMER-0.10)
    set(HAVE_GSTREAMER TRUE)
    ocv_append_build_options(VIDEOIO GSTREAMER-0.10)
    set(GSTREAMER_BASE_VERSION ${GSTREAMER-0.10_gstreamer-base-0.10_VERSION})
    set(GSTREAMER_VIDEO_VERSION ${GSTREAMER-0.10_gstreamer-video-0.10_VERSION})
    set(GSTREAMER_APP_VERSION ${GSTREAMER-0.10_gstreamer-app-0.10_VERSION})
    set(GSTREAMER_RIFF_VERSION ${GSTREAMER-0.10_gstreamer-riff-0.10_VERSION})
    set(GSTREAMER_PBUTILS_VERSION ${GSTREAMER-0.10_gstreamer-pbutils-0.10_VERSION})
  endif()
endif()

# --- unicap ---
ocv_clear_vars(HAVE_UNICAP)
if(WITH_UNICAP)
  ocv_check_modules(HAVE_UNICAP libunicap libucil)
  if(HAVE_UNICAP)
    ocv_append_build_options(VIDEOIO UNICAP)
  endif()
endif()

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
    elseif(ARM)
      set(PVAPI_SDK_SUBDIR arm)
    endif()

    get_filename_component(_PVAPI_LIBRARY_HINT "${PVAPI_INCLUDE_PATH}/../lib-pc" ABSOLUTE)

    find_library(PVAPI_LIBRARY NAMES "PvAPI" PATHS "${_PVAPI_LIBRARY_HINT}")

    if(PVAPI_LIBRARY)
      if(WIN32)
        if(MINGW)
          set(PVAPI_DEFINITIONS "-DPVDECL=__stdcall")
        endif(MINGW)
      endif()
      set(HAVE_PVAPI TRUE)
    endif()
  endif(PVAPI_INCLUDE_PATH)
endif(WITH_PVAPI)

# --- GigEVisionSDK ---
ocv_clear_vars(HAVE_GIGE_API)
if(WITH_GIGEAPI)
  find_path(GIGEAPI_INCLUDE_PATH "GigEVisionSDK.h"
            PATHS /usr/local /var /opt /usr ENV ProgramFiles ENV ProgramW6432
            PATH_SUFFIXES include "Smartek Vision Technologies/GigEVisionSDK/gige_cpp" "GigEVisionSDK/gige_cpp" "GigEVisionSDK/gige_c"
            DOC "The path to Smartek GigEVisionSDK header")
  FIND_LIBRARY(GIGEAPI_LIBRARIES NAMES GigEVisionSDK)
  if(GIGEAPI_LIBRARIES AND GIGEAPI_INCLUDE_PATH)
    set(HAVE_GIGE_API TRUE)
  endif()
endif(WITH_GIGEAPI)

# --- Aravis SDK ---
ocv_clear_vars(HAVE_ARAVIS_API)
if(WITH_ARAVIS)
  ocv_check_modules(ARAVIS_GLIB glib-2.0)
  if(HAVE_ARAVIS_GLIB)
    find_path(ARAVIS_INCLUDE_PATH "arv.h"
              PATHS /usr/local /var /opt /usr ENV ProgramFiles ENV ProgramW6432
              PATH_SUFFIXES include "aravis-0.6" "aravis-0.4"
              DOC "The path to Aravis SDK headers")
    find_library(ARAVIS_LIBRARIES NAMES "aravis-0.6" "aravis-0.4" )
    if(ARAVIS_LIBRARIES AND ARAVIS_INCLUDE_PATH)
      ocv_append_build_options(VIDEOIO ARAVIS_GLIB)
      set(HAVE_ARAVIS_API TRUE)
    endif()
  else()
    message("Can not build Aravis support without glib2")
  endif()
endif(WITH_ARAVIS)

# --- Dc1394 ---
ocv_clear_vars(HAVE_DC1394 HAVE_DC1394_2)
if(WITH_1394)
  if(WIN32 AND MINGW)
      # TODO remove this, use pkgconfig from MinGW instead
      find_path(CMU1394_INCLUDE_PATH "/1394common.h"
                PATH_SUFFIXES include
                DOC "The path to cmu1394 headers")
      find_path(DC1394_2_INCLUDE_PATH "/dc1394/dc1394.h"
                PATH_SUFFIXES include
                DOC "The path to DC1394 2.x headers")
      if(CMU1394_INCLUDE_PATH AND DC1394_2_INCLUDE_PATH)
        set(CMU1394_LIB_DIR  "${CMU1394_INCLUDE_PATH}/../lib"  CACHE PATH "Full path of CMU1394 library directory")
        set(DC1394_2_LIB_DIR "${DC1394_2_INCLUDE_PATH}/../lib" CACHE PATH "Full path of DC1394 2.x library directory")
        if(EXISTS "${CMU1394_LIB_DIR}/lib1394camera.a" AND EXISTS "${DC1394_2_LIB_DIR}/libdc1394.a")
          set(HAVE_DC1394_2 TRUE)
        endif()
      endif()
      if(HAVE_DC1394_2)
        ocv_parse_pkg(DC1394_2_VERSION "libdc1394-2" "${DC1394_2_LIB_DIR}/pkgconfig")
        ocv_include_directories(${DC1394_2_INCLUDE_PATH})
        set(VIDEOIO_LIBRARIES ${VIDEOIO_LIBRARIES}
            "${DC1394_2_LIB_DIR}/libdc1394.a"
            "${CMU1394_LIB_DIR}/lib1394camera.a")
      endif()
  endif()
  if(NOT HAVE_DC1394_2)
    ocv_check_modules(DC1394_2 libdc1394-2)
    if(HAVE_DC1394_2)
      ocv_append_build_options(VIDEOIO DC1394_2)
    else()
      ocv_check_modules(DC1394 libdc1394)
      if(HAVE_DC1394)
        ocv_append_build_options(VIDEOIO DC1394)
      endif()
    endif()
  endif()
endif()

# --- xine ---
ocv_clear_vars(HAVE_XINE)
if(WITH_XINE)
  ocv_check_modules(XINE libxine)
  if(HAVE_XINE)
    ocv_append_build_options(VIDEOIO XINE)
  endif()
endif(WITH_XINE)

# --- V4L ---
ocv_clear_vars(HAVE_LIBV4L HAVE_CAMV4L HAVE_CAMV4L2 HAVE_VIDEOIO)
if(WITH_V4L)
  if(WITH_LIBV4L)
    ocv_check_modules(LIBV4L libv4l1 libv4l2)
    if(HAVE_LIBV4L)
      ocv_append_build_options(VIDEOIO LIBV4L)
    endif()
  endif()
  CHECK_INCLUDE_FILE(linux/videodev.h HAVE_CAMV4L)
  CHECK_INCLUDE_FILE(linux/videodev2.h HAVE_CAMV4L2)
  CHECK_INCLUDE_FILE(sys/videoio.h HAVE_VIDEOIO)
endif(WITH_V4L)

# --- OpenNI ---
ocv_clear_vars(HAVE_OPENNI HAVE_OPENNI_PRIME_SENSOR_MODULE)
if(WITH_OPENNI)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVFindOpenNI.cmake")
endif(WITH_OPENNI)

ocv_clear_vars(HAVE_OPENNI2)
if(WITH_OPENNI2)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVFindOpenNI2.cmake")
endif(WITH_OPENNI2)

# --- XIMEA ---
ocv_clear_vars(HAVE_XIMEA)
if(WITH_XIMEA)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVFindXimea.cmake")
  if(XIMEA_FOUND)
    set(HAVE_XIMEA TRUE)
  endif()
endif(WITH_XIMEA)

# --- FFMPEG ---
ocv_clear_vars(HAVE_FFMPEG)
if(WITH_FFMPEG)  # try FFmpeg autodetection
  if(OPENCV_FFMPEG_USE_FIND_PACKAGE)
    if(OPENCV_FFMPEG_USE_FIND_PACKAGE STREQUAL "1" OR OPENCV_FFMPEG_USE_FIND_PACKAGE STREQUAL "ON")
      set(OPENCV_FFMPEG_USE_FIND_PACKAGE "FFMPEG")
    endif()
    find_package(${OPENCV_FFMPEG_USE_FIND_PACKAGE}) # Required components: AVCODEC AVFORMAT AVUTIL SWSCALE
    if(FFMPEG_FOUND OR FFmpeg_FOUND)
      set(HAVE_FFMPEG TRUE)
    else()
      message(STATUS "Can't find FFmpeg via find_package(${OPENCV_FFMPEG_USE_FIND_PACKAGE})")
    endif()
  elseif(WIN32 AND NOT ARM AND NOT OPENCV_FFMPEG_SKIP_DOWNLOAD)
    include("${OpenCV_SOURCE_DIR}/3rdparty/ffmpeg/ffmpeg.cmake")
    download_win_ffmpeg(FFMPEG_CMAKE_SCRIPT)
    if(FFMPEG_CMAKE_SCRIPT)
      set(HAVE_FFMPEG TRUE)
      set(HAVE_FFMPEG_WRAPPER 1)
      include("${FFMPEG_CMAKE_SCRIPT}")
    endif()
  elseif(PKG_CONFIG_FOUND)
    ocv_check_modules(FFMPEG libavcodec libavformat libavutil libswscale)
    ocv_check_modules(FFMPEG_libavresample libavresample)
    if(FFMPEG_libavresample_FOUND)
      ocv_append_build_options(FFMPEG FFMPEG_libavresample)
    endif()
  else()
    message(STATUS "Can't find ffmpeg - 'pkg-config' utility is missing")
  endif()
endif()
if(HAVE_FFMPEG
    AND NOT HAVE_FFMPEG_WRAPPER
)
  try_compile(__VALID_FFMPEG
      "${OpenCV_BINARY_DIR}"
      "${OpenCV_SOURCE_DIR}/cmake/checks/ffmpeg_test.cpp"
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${FFMPEG_INCLUDE_DIRS}"
                  "-DLINK_DIRECTORIES:STRING=${FFMPEG_LIBRARY_DIRS}"
                  "-DLINK_LIBRARIES:STRING=${FFMPEG_LIBRARIES}"
      OUTPUT_VARIABLE TRY_OUT
  )
  if(NOT __VALID_FFMPEG)
    #message(FATAL_ERROR "FFMPEG: test check build log:\n${TRY_OUT}")
    message(STATUS "WARNING: Can't build ffmpeg test code")
    set(HAVE_FFMPEG FALSE)
  else()
    ocv_append_build_options(VIDEOIO FFMPEG)
  endif()
endif()

# --- VideoInput/DirectShow ---
if(WITH_DSHOW)
  if(MSVC_VERSION GREATER 1499)
    set(HAVE_DSHOW 1)
  elseif(NOT HAVE_DSHOW)
    check_include_file(DShow.h HAVE_DSHOW)
  endif()
endif(WITH_DSHOW)

# --- VideoInput/Microsoft Media Foundation ---
ocv_clear_vars(HAVE_MSMF)
if(WITH_MSMF)
  check_include_file(Mfapi.h HAVE_MSMF)
  set(HAVE_MSMF_DXVA "")
  if(WITH_MSMF_DXVA)
    check_include_file(D3D11.h D3D11_found)
    check_include_file(D3d11_4.h D3D11_4_found)
    if(D3D11_found AND D3D11_4_found)
      set(HAVE_MSMF_DXVA YES)
    endif()
  endif()
endif()

# --- Extra HighGUI and VideoIO libs on Windows ---
if(WIN32)
  list(APPEND HIGHGUI_LIBRARIES comctl32 gdi32 ole32 setupapi ws2_32)
  if(HAVE_VFW)
    list(APPEND VIDEOIO_LIBRARIES vfw32)
  endif()
  if(MINGW AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    list(APPEND VIDEOIO_LIBRARIES avifil32 avicap32 winmm msvfw32)
    list(REMOVE_ITEM VIDEOIO_LIBRARIES vfw32)
  elseif(MINGW)
    list(APPEND VIDEOIO_LIBRARIES winmm)
  endif()
endif(WIN32)

if(APPLE)
  if(WITH_AVFOUNDATION)
    set(HAVE_AVFOUNDATION YES)
  endif()
  if(NOT IOS)
    if(WITH_QUICKTIME)
      set(HAVE_QUICKTIME YES)
    elseif(WITH_QTKIT)
      set(HAVE_QTKIT YES)
    endif()
  endif()
endif(APPLE)

# --- Intel Perceptual Computing SDK ---
if(WITH_INTELPERC)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVFindIntelPerCSDK.cmake")
endif(WITH_INTELPERC)

if(WITH_MFX)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVDetectMediaSDK.cmake")
endif()

# --- gPhoto2 ---
ocv_clear_vars(HAVE_GPHOTO2)
if(WITH_GPHOTO2)
  ocv_check_modules(GPHOTO2 libgphoto2)
  if(HAVE_GPHOTO2)
    ocv_append_build_options(VIDEOIO GPHOTO2)
  endif()
endif()

# --- VA & VA_INTEL ---
if(WITH_VA_INTEL)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVFindVA_INTEL.cmake")
  if(VA_INTEL_IOCL_INCLUDE_DIR)
    ocv_include_directories(${VA_INTEL_IOCL_INCLUDE_DIR})
  endif()
  set(WITH_VA YES)
endif(WITH_VA_INTEL)

if(WITH_VA)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVFindVA.cmake")
  if(VA_INCLUDE_DIR)
    ocv_include_directories(${VA_INCLUDE_DIR})
  endif()
endif(WITH_VA)
