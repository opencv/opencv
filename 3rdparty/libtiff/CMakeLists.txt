# ----------------------------------------------------------------------------
#  CMake file for libtiff. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
project(${TIFF_LIBRARY})

include(CheckCSourceCompiles)
include(CheckFunctionExists)
include(CheckIncludeFile)
include(CheckTypeSize)


# Find libm, if available
find_library(M_LIBRARY m)

check_include_file(assert.h    HAVE_ASSERT_H)
if(NOT MSVC)
  check_include_file(dlfcn.h     HAVE_DLFCN_H)
endif()
check_include_file(fcntl.h     HAVE_FCNTL_H)
check_include_file(inttypes.h  HAVE_INTTYPES_H)
check_include_file(io.h        HAVE_IO_H)
check_include_file(limits.h    HAVE_LIMITS_H)
check_include_file(malloc.h    HAVE_MALLOC_H)
check_include_file(memory.h    HAVE_MEMORY_H)
check_include_file(search.h    HAVE_SEARCH_H)
check_include_file(stdint.h    HAVE_STDINT_H)
check_include_file(string.h    HAVE_STRING_H)
if(NOT MSVC)
  check_include_file(strings.h   HAVE_STRINGS_H)
  check_include_file(sys/time.h  HAVE_SYS_TIME_H)
endif()
check_include_file(sys/types.h HAVE_SYS_TYPES_H)
if(NOT MSVC)
  check_include_file(unistd.h    HAVE_UNISTD_H)
endif()

# Inspired from /usr/share/autoconf/autoconf/c.m4
foreach(inline_keyword "inline" "__inline__" "__inline")
  if(NOT DEFINED C_INLINE)
    set(CMAKE_REQUIRED_DEFINITIONS_SAVE ${CMAKE_REQUIRED_DEFINITIONS})
    set(CMAKE_REQUIRED_DEFINITIONS ${CMAKE_REQUIRED_DEFINITIONS}
        "-Dinline=${inline_keyword}")
    check_c_source_compiles("
        typedef int foo_t;
        static inline foo_t static_foo() {return 0;}
        foo_t foo(){return 0;}
        int main(int argc, char *argv[]) {return 0;}"
      C_HAS_${inline_keyword})
    set(CMAKE_REQUIRED_DEFINITIONS ${CMAKE_REQUIRED_DEFINITIONS_SAVE})
    if(C_HAS_${inline_keyword})
      set(C_INLINE TRUE)
      set(INLINE_KEYWORD "${inline_keyword}")
    endif()
 endif()
endforeach()
if(NOT DEFINED C_INLINE)
  set(INLINE_KEYWORD)
endif()


# Check type sizes
# NOTE: Could be replaced with C99 <stdint.h>
check_type_size("signed short" SIZEOF_SIGNED_SHORT)
check_type_size("unsigned short" SIZEOF_UNSIGNED_SHORT)
check_type_size("signed int" SIZEOF_SIGNED_INT)
check_type_size("unsigned int" SIZEOF_UNSIGNED_INT)
check_type_size("signed long" SIZEOF_SIGNED_LONG)
check_type_size("unsigned long" SIZEOF_UNSIGNED_LONG)
check_type_size("signed long long" SIZEOF_SIGNED_LONG_LONG)
check_type_size("unsigned long long" SIZEOF_UNSIGNED_LONG_LONG)
check_type_size("unsigned char *" SIZEOF_UNSIGNED_CHAR_P)

set(CMAKE_EXTRA_INCLUDE_FILES_SAVE ${CMAKE_EXTRA_INCLUDE_FILES})
set(CMAKE_EXTRA_INCLUDE_FILES ${CMAKE_EXTRA_INCLUDE_FILES} "stddef.h")
check_type_size("size_t" SIZEOF_SIZE_T)
check_type_size("ptrdiff_t" SIZEOF_PTRDIFF_T)
set(CMAKE_EXTRA_INCLUDE_FILES ${CMAKE_EXTRA_INCLUDE_FILES_SAVE})

set(TIFF_INT8_T "signed char")
set(TIFF_UINT8_T "unsigned char")

set(TIFF_INT16_T "signed short")
set(TIFF_UINT16_T "unsigned short")

if(SIZEOF_SIGNED_INT EQUAL 4)
  set(TIFF_INT32_T "signed int")
  set(TIFF_INT32_FORMAT "%d")
elseif(SIZEOF_SIGNED_LONG EQUAL 4)
  set(TIFF_INT32_T "signed long")
  set(TIFF_INT32_FORMAT "%ld")
endif()

if(SIZEOF_UNSIGNED_INT EQUAL 4)
  set(TIFF_UINT32_T "unsigned int")
  set(TIFF_UINT32_FORMAT "%u")
elseif(SIZEOF_UNSIGNED_LONG EQUAL 4)
  set(TIFF_UINT32_T "unsigned long")
  set(TIFF_UINT32_FORMAT "%lu")
endif()

if(SIZEOF_SIGNED_LONG EQUAL 8)
  set(TIFF_INT64_T "signed long")
  set(TIFF_INT64_FORMAT "%ld")
elseif(SIZEOF_SIGNED_LONG_LONG EQUAL 8)
  set(TIFF_INT64_T "signed long long")
  if(MINGW)
    set(TIFF_INT64_FORMAT "%I64d")
  else()
    set(TIFF_INT64_FORMAT "%lld")
  endif()
endif()

if(SIZEOF_UNSIGNED_LONG EQUAL 8)
  set(TIFF_UINT64_T "unsigned long")
  set(TIFF_UINT64_FORMAT "%lu")
elseif(SIZEOF_UNSIGNED_LONG_LONG EQUAL 8)
  set(TIFF_UINT64_T "unsigned long long")
  if(MINGW)
    set(TIFF_UINT64_FORMAT "%I64u")
  else()
    set(TIFF_UINT64_FORMAT "%llu")
  endif()
endif()

if(SIZEOF_UNSIGNED_INT EQUAL SIZEOF_SIZE_T)
  set(TIFF_SIZE_T "unsigned int")
  set(TIFF_SIZE_FORMAT "%u")
elseif(SIZEOF_UNSIGNED_LONG EQUAL SIZEOF_SIZE_T)
  set(TIFF_SIZE_T "unsigned long")
  set(TIFF_SIZE_FORMAT "%lu")
elseif(SIZEOF_UNSIGNED_LONG_LONG EQUAL SIZEOF_SIZE_T)
  set(TIFF_SIZE_T "unsigned long")
  if(MINGW)
    set(TIFF_SIZE_FORMAT "%I64u")
  else()
    set(TIFF_SIZE_FORMAT "%llu")
  endif()
endif()

if(SIZEOF_SIGNED_INT EQUAL SIZEOF_UNSIGNED_CHAR_P)
  set(TIFF_SSIZE_T "signed int")
  set(TIFF_SSIZE_FORMAT "%d")
elseif(SIZEOF_SIGNED_LONG EQUAL SIZEOF_UNSIGNED_CHAR_P)
  set(TIFF_SSIZE_T "signed long")
  set(TIFF_SSIZE_FORMAT "%ld")
elseif(SIZEOF_SIGNED_LONG_LONG EQUAL SIZEOF_UNSIGNED_CHAR_P)
  set(TIFF_SSIZE_T "signed long long")
  if(MINGW)
    set(TIFF_SSIZE_FORMAT "%I64d")
  else()
    set(TIFF_SSIZE_FORMAT "%lld")
  endif()
endif()

if(NOT SIZEOF_PTRDIFF_T)
  set(TIFF_PTRDIFF_T "${TIFF_SSIZE_T}")
  set(TIFF_PTRDIFF_FORMAT "${SSIZE_FORMAT}")
else()
  set(TIFF_PTRDIFF_T "ptrdiff_t")
  set(TIFF_PTRDIFF_FORMAT "%ld")
endif()

# Nonstandard int types
if(NOT MSVC)
  check_type_size(INT8 int8)
  set(HAVE_INT8 ${INT8})
  check_type_size(INT16 int16)
  set(HAVE_INT16 ${INT16})
  check_type_size(INT32 int32)
  set(HAVE_INT32 ${INT32})
endif()

# Check functions
if(NOT MSVC)
   set(CMAKE_REQUIRED_LIBRARIES_SAVE ${CMAKE_REQUIRED_LIBRARIES})
  set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${M_LIBRARY})
  check_function_exists(floor HAVE_FLOOR)
  check_function_exists(pow   HAVE_POW)
  check_function_exists(sqrt  HAVE_SQRT)
  set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES_SAVE})
endif()

if(NOT MSVC)
  check_function_exists(isascii    HAVE_ISASCII)
  check_function_exists(memset     HAVE_MEMSET)
  check_function_exists(mmap       HAVE_MMAP)
  check_function_exists(getopt     HAVE_GETOPT)
endif()
check_function_exists(memmove    HAVE_MEMMOVE)
check_function_exists(setmode    HAVE_SETMODE)
check_function_exists(strcasecmp HAVE_STRCASECMP)
check_function_exists(strchr     HAVE_STRCHR)
check_function_exists(strrchr    HAVE_STRRCHR)
check_function_exists(strstr     HAVE_STRSTR)
check_function_exists(strtol     HAVE_STRTOL)
check_function_exists(strtol     HAVE_STRTOUL)
check_function_exists(strtoull   HAVE_STRTOULL)
check_function_exists(lfind      HAVE_LFIND)

# May be inlined, so check it compiles:
check_c_source_compiles("
#include <stdio.h>
int main(void) {
  char buf[10];
  snprintf(buf, 10, \"Test %d\", 1);
  return 0;
}"
  HAVE_SNPRINTF)

if(NOT HAVE_SNPRINTF)
  add_definitions(-DNEED_LIBPORT)
endif()

# CPU bit order
set(fillorder FILLORDER_MSB2LSB)
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "i.*86.*" OR
   CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "amd64.*" OR
   CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64.*")
  set(fillorder FILLORDER_LSB2MSB)
endif()
set(HOST_FILLORDER ${fillorder} CACHE STRING "Native CPU bit order")
mark_as_advanced(HOST_FILLORDER)

# CPU endianness
include(TestBigEndian)
test_big_endian(bigendian)
if(bigendian)
  set(bigendian ON)
else()
  set(bigendian OFF)
endif()
set(HOST_BIG_ENDIAN ${bigendian} CACHE STRING "Native CPU bit order")
mark_as_advanced(HOST_BIG_ENDIAN)
if(HOST_BIG_ENDIAN)
  set(HOST_BIG_ENDIAN 1)
else()
  set(HOST_BIG_ENDIAN 0)
endif()

# IEEE floating point
set(HAVE_IEEEFP 1 CACHE STRING "IEEE floating point is available")
mark_as_advanced(HAVE_IEEEFP)

# Large file support
if(UNIX OR MINGW)
  if(ANDROID AND (ANDROID_NATIVE_API_LEVEL LESS 21) AND CV_GCC)
    # Android NDK build problem: 'mmap' issue with GCC and API<21
  else()
    # This might not catch every possibility catered for by
    # AC_SYS_LARGEFILE.
    add_definitions(-D_FILE_OFFSET_BITS=64)
    set(FILE_OFFSET_BITS 64)
  endif()
endif()

# Documentation install directory (default to cmake project docdir)
set(LIBTIFF_DOCDIR "${CMAKE_INSTALL_FULL_DOCDIR}")

# Options to enable and disable internal codecs

option(ccitt "support for CCITT Group 3 & 4 algorithms" ON)
set(CCITT_SUPPORT ${ccitt})

option(packbits "support for Macintosh PackBits algorithm" ON)
set(PACKBITS_SUPPORT ${packbits})

option(lzw "support for LZW algorithm" ON)
set(LZW_SUPPORT ${lzw})

option(thunder "support for ThunderScan 4-bit RLE algorithm" ON)
set(THUNDER_SUPPORT ${thunder})

option(next "support for NeXT 2-bit RLE algorithm" ON)
set(NEXT_SUPPORT ${next})

option(logluv "support for LogLuv high dynamic range algorithm" ON)
set(LOGLUV_SUPPORT ${logluv})

# Option for Microsoft Document Imaging
option(mdi "support for Microsoft Document Imaging" ON)
set(MDI_SUPPORT ${mdi})

# ZLIB
set(ZLIB_SUPPORT 0)
if(ZLIB_LIBRARY)
  set(ZLIB_SUPPORT 1)
endif()
set(ZIP_SUPPORT ${ZLIB_SUPPORT})

set(PIXARLOG_SUPPORT FALSE)

# JPEG
set(JPEG_SUPPORT FALSE)
if(HAVE_JPEG)
  set(JPEG_SUPPORT TRUE)
  if(TARGET ${JPEG_LIBRARY} AND DEFINED ${JPEG_LIBRARY}_BINARY_DIR)
    include_directories("${${JPEG_LIBRARY}_BINARY_DIR}")
  endif()
  include_directories(${JPEG_INCLUDE_DIR})
endif()

option(old-jpeg "support for Old JPEG compression (read-only)" OFF)  # OpenCV: changed to OFF
set(OJPEG_SUPPORT FALSE)
if(JPEG_SUPPORT AND old-jpeg)
  set(OJPEG_SUPPORT TRUE)
endif()

# OpenCV: turned off
set(JBIG_SUPPORT 0)
set(LZMA_SUPPORT 0)  # OpenCV: turned off
set(JPEG12_FOUND FALSE)  # OpenCV: turned off
set(STRIPCHOP_DEFAULT)
set(STRIP_SIZE_DEFAULT 8192)

# Win32 IO
set(win32_io FALSE)
if(WIN32 AND NOT WINRT)
  set(win32_io TRUE)
endif()
set(USE_WIN32_FILEIO ${win32_io} CACHE BOOL "Use win32 IO system (Microsoft Windows only)")
if(USE_WIN32_FILEIO)
  set(USE_WIN32_FILEIO TRUE)
else()
  set(USE_WIN32_FILEIO FALSE)
endif()

# Orthogonal features

# OpenCV: turned ON
set(SUBIFD_SUPPORT 1)
set(DEFAULT_EXTRASAMPLE_AS_ALPHA 1)
set(CHECK_JPEG_YCBCR_SUBSAMPLING 1)

if(JPEG_INCLUDE_DIR)
  list(APPEND TIFF_INCLUDES ${JPEG_INCLUDE_DIR})
endif()
if(JPEG12_INCLUDE_DIR)
  list(APPEND TIFF_INCLUDES ${JPEG12_INCLUDE_DIR})
endif()
if(JBIG_INCLUDE_DIR)
  list(APPEND TIFF_INCLUDES ${JBIG_INCLUDE_DIR})
endif()
if(LIBLZMA_INCLUDE_DIRS)
  list(APPEND TIFF_INCLUDES ${LIBLZMA_INCLUDE_DIRS})
endif()

# Libraries required by libtiff
set(TIFF_LIBRARY_DEPS)
if(M_LIBRARY)
  list(APPEND TIFF_LIBRARY_DEPS ${M_LIBRARY})
endif()
if(ZLIB_LIBRARIES)
  list(APPEND TIFF_LIBRARY_DEPS ${ZLIB_LIBRARIES})
endif()
if(JPEG_LIBRARIES)
  list(APPEND TIFF_LIBRARY_DEPS ${JPEG_LIBRARIES})
endif()
if(JPEG12_LIBRARIES)
  list(APPEND TIFF_LIBRARY_DEPS ${JPEG12_LIBRARIES})
endif()
if(JBIG_LIBRARIES)
  list(APPEND TIFF_LIBRARY_DEPS ${JBIG_LIBRARIES})
endif()
if(LIBLZMA_LIBRARIES)
  list(APPEND TIFF_LIBRARY_DEPS ${LIBLZMA_LIBRARIES})
endif()



configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tif_config.h.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/tif_config.h"
               @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tiffconf.h.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/tiffconf.h"
               @ONLY)

ocv_include_directories("${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}" ${ZLIB_INCLUDE_DIRS})

set(lib_srcs
    tif_aux.c
    tif_close.c
    tif_codec.c
    tif_color.c
    tif_compress.c
    tif_dir.c
    tif_dirinfo.c
    tif_dirread.c
    tif_dirwrite.c
    tif_dumpmode.c
    tif_error.c
    tif_extension.c
    tif_fax3.c
    tif_fax3sm.c
    tif_flush.c
    tif_getimage.c
    tif_jbig.c
    tif_jpeg_12.c
    tif_jpeg.c
    tif_luv.c
    tif_lzma.c
    tif_lzw.c
    tif_next.c
    tif_ojpeg.c
    tif_open.c
    tif_packbits.c
    tif_pixarlog.c
    tif_predict.c
    tif_print.c
    tif_read.c
    tif_strip.c
    tif_swab.c
    tif_thunder.c
    tif_tile.c
    tif_version.c
    tif_warning.c
    tif_write.c
    tif_zip.c
    tif_stream.cxx
    snprintf.c
    t4.h
    tif_dir.h
    tif_fax3.h
    tiff.h
    tiffio.h
    tiffiop.h
    tiffvers.h
    tif_predict.h
    uvcode.h
    tiffio.hxx
    "${CMAKE_CURRENT_BINARY_DIR}/tif_config.h"
    "${CMAKE_CURRENT_BINARY_DIR}/tiffconf.h"
    )

if(WIN32 AND NOT WINRT)
  list(APPEND lib_srcs tif_win32.c)
else()
  list(APPEND lib_srcs tif_unix.c)
endif()

ocv_warnings_disable(CMAKE_C_FLAGS -Wno-unused-but-set-variable -Wmissing-prototypes -Wmissing-declarations -Wundef -Wunused -Wsign-compare
                                   -Wcast-align -Wshadow -Wno-maybe-uninitialized -Wno-pointer-to-int-cast -Wno-int-to-pointer-cast
                                   -Wmisleading-indentation
                                   -Wimplicit-fallthrough
)
ocv_warnings_disable(CMAKE_C_FLAGS -Wunused-parameter) # clang
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wmissing-declarations -Wunused-parameter
    -Wundef  # tiffiop.h: #if __clang_major__ >= 4
)
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4018 /wd4100 /wd4127 /wd4311 /wd4701 /wd4706) # vs2005
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4244) # vs2008
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4267 /wd4305 /wd4306) # vs2008 Win64
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4703) # vs2012
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4456 /wd4457 /wd4312) # vs2015

ocv_warnings_disable(CMAKE_C_FLAGS /wd4267 /wd4244 /wd4018 /wd4311 /wd4312)

add_library(${TIFF_LIBRARY} STATIC ${lib_srcs})
target_link_libraries(${TIFF_LIBRARY} ${ZLIB_LIBRARIES})

set_target_properties(${TIFF_LIBRARY}
    PROPERTIES
    OUTPUT_NAME "${TIFF_LIBRARY}"
    DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
    COMPILE_PDB_NAME ${TIFF_LIBRARY}
    COMPILE_PDB_NAME_DEBUG "${TIFF_LIBRARY}${OPENCV_DEBUG_POSTFIX}"
    ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
    )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${TIFF_LIBRARY} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${TIFF_LIBRARY} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

ocv_install_3rdparty_licenses(libtiff COPYRIGHT)
