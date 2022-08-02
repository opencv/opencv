project(${JPEG_LIBRARY} C)

ocv_warnings_disable(CMAKE_C_FLAGS -Wunused-parameter -Wsign-compare -Wshorten-64-to-32 -Wimplicit-fallthrough)

set(VERSION_MAJOR 2)
set(VERSION_MINOR 1)
set(VERSION_REVISION 2)
set(VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_REVISION})
set(LIBJPEG_TURBO_VERSION_NUMBER 2001002)

string(TIMESTAMP BUILD "opencv-${OPENCV_VERSION}-libjpeg-turbo")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(BUILD "${BUILD}-debug")
endif()

message(STATUS "libjpeg-turbo: VERSION = ${VERSION}, BUILD = ${BUILD}")

# Detect CPU type and whether we're building 64-bit or 32-bit code
math(EXPR BITS "${CMAKE_SIZEOF_VOID_P} * 8")
string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} CMAKE_SYSTEM_PROCESSOR_LC)
set(COUNT 1)
foreach(ARCH ${CMAKE_OSX_ARCHITECTURES})
  if(COUNT GREATER 1)
    message(FATAL_ERROR "The libjpeg-turbo build system does not support multiple values in CMAKE_OSX_ARCHITECTURES.")
  endif()
  math(EXPR COUNT "${COUNT}+1")
endforeach()
if(CMAKE_SYSTEM_PROCESSOR_LC MATCHES "x86_64" OR
        CMAKE_SYSTEM_PROCESSOR_LC MATCHES "amd64" OR
        CMAKE_SYSTEM_PROCESSOR_LC MATCHES "i[0-9]86" OR
        CMAKE_SYSTEM_PROCESSOR_LC MATCHES "x86" OR
        CMAKE_SYSTEM_PROCESSOR_LC MATCHES "ia32")
  if(BITS EQUAL 64 OR CMAKE_C_COMPILER_ABI MATCHES "ELF X32")
    set(CPU_TYPE x86_64)
  else()
    set(CPU_TYPE i386)
  endif()
  if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL ${CPU_TYPE})
    set(CMAKE_SYSTEM_PROCESSOR ${CPU_TYPE})
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR_LC STREQUAL "aarch64" OR
        CMAKE_SYSTEM_PROCESSOR_LC MATCHES "^arm")
  if(BITS EQUAL 64)
    set(CPU_TYPE arm64)
  else()
    set(CPU_TYPE arm)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR_LC MATCHES "^ppc" OR
        CMAKE_SYSTEM_PROCESSOR_LC MATCHES "^powerpc")
  set(CPU_TYPE powerpc)
else()
  set(CPU_TYPE ${CMAKE_SYSTEM_PROCESSOR_LC})
endif()
if(CMAKE_OSX_ARCHITECTURES MATCHES "x86_64" OR
        CMAKE_OSX_ARCHITECTURES MATCHES "arm64" OR
        CMAKE_OSX_ARCHITECTURES MATCHES "i386")
  set(CPU_TYPE ${CMAKE_OSX_ARCHITECTURES})
endif()
if(CMAKE_OSX_ARCHITECTURES MATCHES "ppc")
  set(CPU_TYPE powerpc)
endif()
if(MSVC_IDE AND CMAKE_GENERATOR_PLATFORM MATCHES "arm64")
  set(CPU_TYPE arm64)
endif()

message(STATUS "${BITS}-bit build (${CPU_TYPE})")

option(REQUIRE_SIMD "Generate a fatal error if SIMD extensions are not available for this platform (default is to fall back to a non-SIMD build)" FALSE)
option(WITH_ARITH_ENC "Include arithmetic encoding support when emulating the libjpeg v6b API/ABI" TRUE)
option(WITH_ARITH_DEC "Include arithmetic decoding support when emulating the libjpeg v6b API/ABI" TRUE)
option(WITH_SIMD "Include SIMD extensions, if available for this platform" TRUE)

include(CheckCSourceCompiles)
include(CheckIncludeFiles)
include(CheckTypeSize)

check_type_size("size_t" SIZEOF_SIZE_T)
check_type_size("unsigned long" SIZEOF_UNSIGNED_LONG)

if(SIZEOF_SIZE_T EQUAL SIZEOF_UNSIGNED_LONG)
  check_c_source_compiles("int main(int argc, char **argv) { unsigned long a = argc;  return __builtin_ctzl(a); }"
    HAVE_BUILTIN_CTZL)
endif()
if(MSVC)
  check_include_files("intrin.h" HAVE_INTRIN_H)
endif()

if(UNIX)
  # Check for headers
  check_include_files(locale.h HAVE_LOCALE_H)
  check_include_files(stddef.h HAVE_STDDEF_H)
  check_include_files(stdlib.h HAVE_STDLIB_H)
  check_include_files(sys/types.h NEED_SYS_TYPES_H)

  # Other predefines
  # undef NEED_BSD_STRINGS
  ocv_update(HAVE_UNSIGNED_CHAR 1)
  ocv_update(HAVE_UNSIGNED_SHORT 1)
  # undef INCOMPLETE_TYPES_BROKEN
  ocv_update(RIGHT_SHIFT_IS_UNSIGNED 0)
endif()


set(BITS_IN_JSAMPLE 8)

if(WITH_ARITH_ENC)
  set(C_ARITH_CODING_SUPPORTED 1)
endif()

if(WITH_ARITH_DEC)
  set(D_ARITH_CODING_SUPPORTED 1)
endif()

set(JPEG_LIB_VERSION 62)

# OpenCV
set(JPEG_LIB_VERSION "${VERSION}-${JPEG_LIB_VERSION}" PARENT_SCOPE)

set(THREAD_LOCAL "")  # WITH_TURBOJPEG is not used

if(MSVC)
  add_definitions(-W3 -wd4996 -wd4018)
endif()

if(WIN32)
  configure_file(jconfig.h.win.in jconfig.h)
else()
  configure_file(jconfig.h.in jconfig.h)
endif()
configure_file(jconfigint.h.in jconfigint.h)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(JPEG_SOURCES jcapimin.c jcapistd.c jccoefct.c jccolor.c jcdctmgr.c jchuff.c
        jcicc.c jcinit.c jcmainct.c jcmarker.c jcmaster.c jcomapi.c jcparam.c
        jcphuff.c jcprepct.c jcsample.c jctrans.c jdapimin.c jdapistd.c jdatadst.c
        jdatasrc.c jdcoefct.c jdcolor.c jddctmgr.c jdhuff.c jdicc.c jdinput.c
        jdmainct.c jdmarker.c jdmaster.c jdmerge.c jdphuff.c jdpostct.c jdsample.c
        jdtrans.c jerror.c jfdctflt.c jfdctfst.c jfdctint.c jidctflt.c jidctfst.c
        jidctint.c jidctred.c jquant1.c jquant2.c jutils.c jmemmgr.c jmemnobs.c)

if(WITH_ARITH_ENC OR WITH_ARITH_DEC)
  set(JPEG_SOURCES ${JPEG_SOURCES} jaricom.c)
endif()

if(WITH_ARITH_ENC)
  set(JPEG_SOURCES ${JPEG_SOURCES} jcarith.c)
endif()

if(WITH_ARITH_DEC)
  set(JPEG_SOURCES ${JPEG_SOURCES} jdarith.c)
endif()

if(WITH_SIMD)
  add_subdirectory(${PROJECT_SOURCE_DIR}/src/simd)
  if(NEON_INTRINSICS)
    add_definitions(-DNEON_INTRINSICS)
  endif()
elseif(NOT WITH_12BIT)
  message(STATUS "SIMD extensions: None (WITH_SIMD = ${WITH_SIMD})")
endif()
if(WITH_SIMD)
  message(STATUS "SIMD extensions: ${CPU_TYPE} (WITH_SIMD = ${WITH_SIMD})")
  if(MSVC_IDE OR XCODE)
    set_source_files_properties(${SIMD_OBJS} PROPERTIES GENERATED 1)
  endif()
else()
  add_library(simd OBJECT src/jsimd_none.c)
  if(NOT WIN32 AND (CMAKE_POSITION_INDEPENDENT_CODE OR ENABLE_SHARED))
    set_target_properties(simd PROPERTIES POSITION_INDEPENDENT_CODE 1)
  endif()
endif()

set(JPEG_SOURCES ${JPEG_SOURCES} ${SIMD_OBJS})

# No SIMD
# set(JPEG_SOURCES ${JPEG_SOURCES} jsimd_none.c)

ocv_list_add_prefix(JPEG_SOURCES src/)

add_library(${JPEG_LIBRARY} STATIC ${OPENCV_3RDPARTY_EXCLUDE_FROM_ALL} ${JPEG_SOURCES} $<TARGET_OBJECTS:simd>
        ${SIMD_OBJS})

set_target_properties(${JPEG_LIBRARY}
  PROPERTIES OUTPUT_NAME ${JPEG_LIBRARY}
  DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
  COMPILE_PDB_NAME ${JPEG_LIBRARY}
  COMPILE_PDB_NAME_DEBUG "${JPEG_LIBRARY}${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
  )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${JPEG_LIBRARY} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${JPEG_LIBRARY} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev OPTIONAL)
endif()

ocv_install_3rdparty_licenses(libjpeg-turbo README.md LICENSE.md README.ijg)
