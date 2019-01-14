# --- FFMPEG ---
if(NOT HAVE_FFMPEG AND OPENCV_FFMPEG_USE_FIND_PACKAGE)
  if(OPENCV_FFMPEG_USE_FIND_PACKAGE STREQUAL "1" OR OPENCV_FFMPEG_USE_FIND_PACKAGE STREQUAL "ON")
    set(OPENCV_FFMPEG_USE_FIND_PACKAGE "FFMPEG")
  endif()
  find_package(${OPENCV_FFMPEG_USE_FIND_PACKAGE}) # Required components: AVCODEC AVFORMAT AVUTIL SWSCALE
  if(FFMPEG_FOUND OR FFmpeg_FOUND)
    set(HAVE_FFMPEG TRUE)
  endif()
endif()

if(NOT HAVE_FFMPEG AND WIN32 AND NOT ARM AND NOT OPENCV_FFMPEG_SKIP_DOWNLOAD)
  include("${OpenCV_SOURCE_DIR}/3rdparty/ffmpeg/ffmpeg.cmake")
  download_win_ffmpeg(FFMPEG_CMAKE_SCRIPT)
  if(FFMPEG_CMAKE_SCRIPT)
    include("${FFMPEG_CMAKE_SCRIPT}")
    set(FFMPEG_libavcodec_VERSION ${FFMPEG_libavcodec_VERSION} PARENT_SCOPE) # info
    set(FFMPEG_libavformat_VERSION ${FFMPEG_libavformat_VERSION} PARENT_SCOPE) # info
    set(FFMPEG_libavutil_VERSION ${FFMPEG_libavutil_VERSION} PARENT_SCOPE) # info
    set(FFMPEG_libswscale_VERSION ${FFMPEG_libswscale_VERSION} PARENT_SCOPE) # info
    set(FFMPEG_libavresample_VERSION ${FFMPEG_libavresample_VERSION} PARENT_SCOPE) # info
    set(HAVE_FFMPEG TRUE)
    set(HAVE_FFMPEG_WRAPPER TRUE)
  endif()
endif()

if(NOT HAVE_FFMPEG AND PKG_CONFIG_FOUND)
  pkg_check_modules(FFMPEG libavcodec libavformat libavutil libswscale QUIET)
  pkg_check_modules(FFMPEG_libavresample libavresample QUIET) # optional
  if(FFMPEG_FOUND)
    if(FFMPEG_libavresample_FOUND)
      list(APPEND FFMPEG_LIBRARIES ${FFMPEG_libavresample_LIBRARIES})
    endif()
    # rewrite libraries to absolute paths
    foreach(lib ${FFMPEG_LIBRARIES})
      find_library(FFMPEG_ABSOLUTE_${lib} "${lib}" PATHS "${FFMPEG_lib${lib}_LIBDIR}" "${FFMPEG_LIBRARY_DIRS}" NO_DEFAULT_PATH)
      if(FFMPEG_ABSOLUTE_${lib})
        list(APPEND ffmpeg_abs_libs "${FFMPEG_ABSOLUTE_${lib}}")
      else()
        list(APPEND ffmpeg_abs_libs "${lib}")
      endif()
    endforeach()
    set(FFMPEG_LIBRARIES "${ffmpeg_abs_libs}" CACHE INTERNAL "" FORCE)

    set(HAVE_FFMPEG TRUE)
  endif()
endif()

#==================================

if(HAVE_FFMPEG AND NOT HAVE_FFMPEG_WRAPPER AND NOT OPENCV_FFMPEG_SKIP_BUILD_CHECK)
  try_compile(__VALID_FFMPEG
      "${OpenCV_BINARY_DIR}"
      "${OpenCV_SOURCE_DIR}/cmake/checks/ffmpeg_test.cpp"
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${FFMPEG_INCLUDE_DIRS}"
                  "-DLINK_LIBRARIES:STRING=${FFMPEG_LIBRARIES}"
      OUTPUT_VARIABLE TRY_OUT
  )
  if(NOT __VALID_FFMPEG)
    # message(FATAL_ERROR "FFMPEG: test check build log:\n${TRY_OUT}")
    message(STATUS "WARNING: Can't build ffmpeg test code")
    set(HAVE_FFMPEG FALSE)
  endif()
endif()

#==================================

if(HAVE_FFMPEG)
  set(defs "HAVE_FFMPEG")
  if(HAVE_FFMPEG_WRAPPER)
    list(APPEND defs "HAVE_FFMPEG_WRAPPER")
  endif()
  ocv_add_external_target(ffmpeg "${FFMPEG_INCLUDE_DIRS}" "${FFMPEG_LIBRARIES}" "${defs}")
endif()

set(HAVE_FFMPEG ${HAVE_FFMPEG} PARENT_SCOPE)
