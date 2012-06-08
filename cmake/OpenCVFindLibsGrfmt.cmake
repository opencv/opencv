# ----------------------------------------------------------------------------
#  Detect 3rd-party image IO libraries
# ----------------------------------------------------------------------------

# --- zlib (required) ---
if(BUILD_ZLIB)
  ocv_clear_vars(ZLIB_FOUND)
else()
  include(FindZLIB)
  if(ZLIB_FOUND)
    if(ZLIB_VERSION_STRING)
      #cmake 2.8.2 bug - it fails to determine zlib version
      unset(ZLIB_VERSION_STRING CACHE)
    endif()
    ocv_parse_header2(ZLIB "${ZLIB_INCLUDE_DIR}/zlib.h" ZLIB_VERSION "")
  endif()
  if(ZLIB_FOUND AND ANDROID)
    if(ZLIB_LIBRARY STREQUAL "${ANDROID_SYSROOT}/usr/lib/libz.so")
      set(ZLIB_LIBRARY z)
      set(ZLIB_LIBRARIES z)
    endif()
  endif()
endif()

if(NOT ZLIB_FOUND)
  ocv_clear_vars(ZLIB_LIBRARY ZLIB_LIBRARIES ZLIB_INCLUDE_DIR)

  set(ZLIB_LIBRARY zlib)
  set(ZLIB_LIBRARIES ${ZLIB_LIBRARY})
  add_subdirectory("${OpenCV_SOURCE_DIR}/3rdparty/zlib")
  set(ZLIB_INCLUDE_DIR "${${ZLIB_LIBRARY}_SOURCE_DIR}" "${${ZLIB_LIBRARY}_BINARY_DIR}")
endif()

# --- libtiff (optional, should be searched after zlib) ---
if(WITH_TIFF)
  if(BUILD_TIFF)
    ocv_clear_vars(TIFF_FOUND)
  else()
    include(FindTIFF)
    if(TIFF_FOUND)
      ocv_parse_header("${TIFF_INCLUDE_DIR}/tiff.h" TIFF_VERSION_LINES TIFF_VERSION_CLASSIC TIFF_VERSION_BIG TIFF_VERSION TIFF_BIGTIFF_VERSION)
    endif()
  endif()
endif()

if(WITH_TIFF AND NOT TIFF_FOUND)
  ocv_clear_vars(TIFF_LIBRARY TIFF_LIBRARIES TIFF_INCLUDE_DIR)

  set(TIFF_LIBRARY libtiff)
  set(TIFF_LIBRARIES ${TIFF_LIBRARY})
  add_subdirectory("${OpenCV_SOURCE_DIR}/3rdparty/libtiff")
  set(TIFF_INCLUDE_DIR "${${TIFF_LIBRARY}_SOURCE_DIR}" "${${TIFF_LIBRARY}_BINARY_DIR}")
  ocv_parse_header("${${TIFF_LIBRARY}_SOURCE_DIR}/tiff.h" TIFF_VERSION_LINES TIFF_VERSION_CLASSIC TIFF_VERSION_BIG TIFF_VERSION TIFF_BIGTIFF_VERSION)
endif()

if(TIFF_VERSION_CLASSIC AND NOT TIFF_VERSION)
  set(TIFF_VERSION ${TIFF_VERSION_CLASSIC})
endif()

if(TIFF_BIGTIFF_VERSION AND NOT TIFF_VERSION_BIG)
  set(TIFF_VERSION_BIG ${TIFF_BIGTIFF_VERSION})
endif()

if(NOT TIFF_VERSION_STRING AND TIFF_INCLUDE_DIR)
  list(GET TIFF_INCLUDE_DIR 0 _TIFF_INCLUDE_DIR)
  if(EXISTS "${_TIFF_INCLUDE_DIR}/tiffvers.h")
    file(STRINGS "${_TIFF_INCLUDE_DIR}/tiffvers.h" tiff_version_str REGEX "^#define[\t ]+TIFFLIB_VERSION_STR[\t ]+\"LIBTIFF, Version .*")
    string(REGEX REPLACE "^#define[\t ]+TIFFLIB_VERSION_STR[\t ]+\"LIBTIFF, Version +([^ \\n]*).*" "\\1" TIFF_VERSION_STRING "${tiff_version_str}")
    unset(tiff_version_str)
  endif()
  unset(_TIFF_INCLUDE_DIR)
endif()

# --- libjpeg (optional) ---
if(WITH_JPEG)
  if(BUILD_JPEG)
    ocv_clear_vars(JPEG_FOUND)
  else()
    include(FindJPEG)
  endif()
endif()

if(WITH_JPEG AND NOT JPEG_FOUND)
  ocv_clear_vars(JPEG_LIBRARY JPEG_LIBRARIES JPEG_INCLUDE_DIR)

  set(JPEG_LIBRARY libjpeg)
  set(JPEG_LIBRARIES ${JPEG_LIBRARY})
  add_subdirectory("${OpenCV_SOURCE_DIR}/3rdparty/libjpeg")
  set(JPEG_INCLUDE_DIR "${${JPEG_LIBRARY}_SOURCE_DIR}")
endif()

ocv_parse_header("${JPEG_INCLUDE_DIR}/jpeglib.h" JPEG_VERSION_LINES JPEG_LIB_VERSION)


# --- libjasper (optional, should be searched after libjpeg) ---
if(WITH_JASPER)
  if(BUILD_JASPER)
    ocv_clear_vars(JASPER_FOUND)
  else()
    include(FindJasper)
  endif()
endif()

if(WITH_JASPER AND NOT JASPER_FOUND)
  ocv_clear_vars(JASPER_LIBRARY JASPER_LIBRARIES JASPER_INCLUDE_DIR)

  set(JASPER_LIBRARY libjasper)
  set(JASPER_LIBRARIES ${JASPER_LIBRARY})
  add_subdirectory("${OpenCV_SOURCE_DIR}/3rdparty/libjasper")
  set(JASPER_INCLUDE_DIR "${${JASPER_LIBRARY}_SOURCE_DIR}")
endif()

if(NOT JASPER_VERSION_STRING)
  ocv_parse_header2(JASPER "${JASPER_INCLUDE_DIR}/jasper/jas_config.h" JAS_VERSION "")
endif()

# --- libpng (optional, should be searched after zlib) ---
if(WITH_PNG)
  if(BUILD_PNG)
    ocv_clear_vars(PNG_FOUND)
  else()
    include(FindPNG)
    if(PNG_FOUND)
      check_include_file("${PNG_PNG_INCLUDE_DIR}/png.h"        HAVE_PNG_H)
      check_include_file("${PNG_PNG_INCLUDE_DIR}/libpng/png.h" HAVE_LIBPNG_PNG_H)
      if(HAVE_PNG_H)
        ocv_parse_header("${PNG_PNG_INCLUDE_DIR}/png.h" PNG_VERSION_LINES PNG_LIBPNG_VER_MAJOR PNG_LIBPNG_VER_MINOR PNG_LIBPNG_VER_RELEASE)
      elseif(HAVE_LIBPNG_PNG_H)
        ocv_parse_header("${PNG_PNG_INCLUDE_DIR}/libpng/png.h" PNG_VERSION_LINES PNG_LIBPNG_VER_MAJOR PNG_LIBPNG_VER_MINOR PNG_LIBPNG_VER_RELEASE)
      endif()
    endif()
  endif()
endif()

if(WITH_PNG AND NOT PNG_FOUND)
  ocv_clear_vars(PNG_LIBRARY PNG_LIBRARIES PNG_INCLUDE_DIR PNG_PNG_INCLUDE_DIR HAVE_PNG_H HAVE_LIBPNG_PNG_H PNG_DEFINITIONS)

  set(PNG_LIBRARY libpng)
  set(PNG_LIBRARIES ${PNG_LIBRARY})
  add_subdirectory("${OpenCV_SOURCE_DIR}/3rdparty/libpng")
  set(PNG_INCLUDE_DIR "${${PNG_LIBRARY}_SOURCE_DIR}")
  set(PNG_DEFINITIONS "")
  ocv_parse_header("${PNG_INCLUDE_DIR}/png.h" PNG_VERSION_LINES PNG_LIBPNG_VER_MAJOR PNG_LIBPNG_VER_MINOR PNG_LIBPNG_VER_RELEASE)
endif()

set(PNG_VERSION "${PNG_LIBPNG_VER_MAJOR}.${PNG_LIBPNG_VER_MINOR}.${PNG_LIBPNG_VER_RELEASE}")

# --- OpenEXR (optional) ---
if(WITH_OPENEXR)
  include("${OpenCV_SOURCE_DIR}/cmake/OpenCVFindOpenEXR.cmake")
endif()
