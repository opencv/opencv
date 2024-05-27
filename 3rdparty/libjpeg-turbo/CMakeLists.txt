project(${JPEG_LIBRARY} C)

macro(boolean_number var)
  if(${var})
    set(${var} 1 ${ARGN})
  else()
    set(${var} 0 ${ARGN})
  endif()
endmacro()

ocv_warnings_disable(CMAKE_C_FLAGS -Wunused-parameter -Wsign-compare -Wshorten-64-to-32 -Wimplicit-fallthrough)
if(APPLE)
  ocv_warnings_disable(CMAKE_C_FLAGS -Wunused-variable) # NEON flags are not used on Mac
endif()

if(CV_GCC AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
  # src/jchuff.c:1042:22: warning: writing 1 byte into a region of size 0 [-Wstringop-overflow=]
  ocv_warnings_disable(CMAKE_C_FLAGS -Wstringop-overflow)
endif()

set(VERSION 3.0.3)
set(COPYRIGHT_YEAR "1991-2024")
string(REPLACE "." ";" VERSION_TRIPLET ${VERSION})
list(GET VERSION_TRIPLET 0 VERSION_MAJOR)
list(GET VERSION_TRIPLET 1 VERSION_MINOR)
list(GET VERSION_TRIPLET 2 VERSION_REVISION)
function(pad_number NUMBER OUTPUT_LEN)
  string(LENGTH "${${NUMBER}}" INPUT_LEN)
  if(INPUT_LEN LESS OUTPUT_LEN)
    math(EXPR ZEROES "${OUTPUT_LEN} - ${INPUT_LEN} - 1")
    set(NUM ${${NUMBER}})
    foreach(C RANGE ${ZEROES})
      set(NUM "0${NUM}")
    endforeach()
    set(${NUMBER} ${NUM} PARENT_SCOPE)
  endif()
endfunction()
pad_number(VERSION_MINOR 3)
pad_number(VERSION_REVISION 3)
set(LIBJPEG_TURBO_VERSION_NUMBER ${VERSION_MAJOR}${VERSION_MINOR}${VERSION_REVISION})

string(TIMESTAMP BUILD "opencv-${OPENCV_VERSION}-libjpeg-turbo")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(BUILD "${BUILD}-debug")
endif()

message(STATUS "libjpeg-turbo: VERSION = ${VERSION}, BUILD = ${BUILD}")

math(EXPR BITS "${CMAKE_SIZEOF_VOID_P} * 8")
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" CMAKE_SYSTEM_PROCESSOR_LC)

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

OCV_OPTION(ENABLE_LIBJPEG_TURBO_SIMD "Include SIMD extensions for libjpeg-turbo, if available for this platform" (NOT CV_DISABLE_OPTIMIZATION))
option(WITH_ARITH_ENC "Include arithmetic encoding support when emulating the libjpeg v6b API/ABI" TRUE)
option(WITH_ARITH_DEC "Include arithmetic decoding support when emulating the libjpeg v6b API/ABI" TRUE)
set(WITH_SIMD 1)
set(HAVE_LIBJPEG_TURBO_SIMD 0 PARENT_SCOPE)

include(CheckCSourceCompiles)
include(CheckIncludeFiles)
include(CheckTypeSize)

check_type_size("size_t" SIZE_T)
check_type_size("unsigned long" UNSIGNED_LONG)

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

set(JPEG_LIB_VERSION 70)

# OpenCV
set(JPEG_LIB_VERSION "${VERSION}-${JPEG_LIB_VERSION}" PARENT_SCOPE)

set(THREAD_LOCAL "")  # WITH_TURBOJPEG is not used

add_definitions(-DNO_GETENV -DNO_PUTENV)

if(MSVC)
  add_definitions(-W3 -wd4996 -wd4018)
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(JPEG16_SOURCES jcapistd.c jccolor.c jcdiffct.c jclossls.c jcmainct.c
    jcprepct.c jcsample.c jdapistd.c jdcolor.c jddiffct.c jdlossls.c jdmainct.c
    jdpostct.c jdsample.c jutils.c)

set(JPEG12_SOURCES ${JPEG16_SOURCES} jccoefct.c jcdctmgr.c jdcoefct.c
    jddctmgr.c jdmerge.c jfdctfst.c jfdctint.c jidctflt.c jidctfst.c jidctint.c
    jidctred.c jquant1.c jquant2.c)

set(JPEG_SOURCES ${JPEG12_SOURCES} jcapimin.c jchuff.c jcicc.c jcinit.c
    jclhuff.c jcmarker.c jcmaster.c jcomapi.c jcparam.c jcphuff.c jctrans.c
    jdapimin.c jdatadst.c jdatasrc.c jdhuff.c jdicc.c jdinput.c jdlhuff.c
    jdmarker.c jdmaster.c jdphuff.c jdtrans.c jerror.c jfdctflt.c jmemmgr.c
    jmemnobs.c jpeg_nbits.c)

if(WITH_ARITH_ENC OR WITH_ARITH_DEC)
  set(JPEG_SOURCES ${JPEG_SOURCES} jaricom.c)
endif()

if(WITH_ARITH_ENC)
  set(JPEG_SOURCES ${JPEG_SOURCES} jcarith.c)
endif()

if(WITH_ARITH_DEC)
  set(JPEG_SOURCES ${JPEG_SOURCES} jdarith.c)
endif()

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_C_COMPILER_ID MATCHES "Clang")
  # Use the maximum optimization level for release builds
  foreach(var CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_RELWITHDEBINFO)
    if(${var} MATCHES "-O2")
      string(REGEX REPLACE "-O2" "-O3" ${var} "${${var}}")
    endif()
  endforeach()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "SunOS")
  if(CMAKE_C_COMPILER_ID MATCHES "SunPro")
    # Use the maximum optimization level for release builds
    foreach(var CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_RELWITHDEBINFO)
      if(${var} MATCHES "-xO3")
        string(REGEX REPLACE "-xO3" "-xO5" ${var} "${${var}}")
      endif()
      if(${var} MATCHES "-xO2")
        string(REGEX REPLACE "-xO2" "-xO5" ${var} "${${var}}")
      endif()
    endforeach()
  endif()
endif()

include(CheckTypeSize)
check_type_size("size_t" SIZE_T)
check_type_size("unsigned long" UNSIGNED_LONG)

if(ENABLE_LIBJPEG_TURBO_SIMD)
  add_subdirectory(src/simd)
  if(NEON_INTRINSICS)
    add_definitions(-DNEON_INTRINSICS)
  endif()
else()
  set(WITH_SIMD 0)
endif()

if(WITH_SIMD)
  message(STATUS "SIMD extensions: ${CPU_TYPE} (WITH_SIMD = ${WITH_SIMD})")
  set(HAVE_LIBJPEG_TURBO_SIMD 1 PARENT_SCOPE)
  if(MSVC_IDE OR XCODE)
    set_source_files_properties(${SIMD_OBJS} PROPERTIES GENERATED 1)
  endif()
  set(SIMD_TARGET_OBJECTS $<TARGET_OBJECTS:simd>)
endif()

configure_file(jversion.h.in jversion.h)
configure_file(jconfig.h.in jconfig.h)
configure_file(jconfigint.h.in jconfigint.h)

ocv_list_add_prefix(JPEG16_SOURCES src/)
ocv_list_add_prefix(JPEG12_SOURCES src/)
ocv_list_add_prefix(JPEG_SOURCES src/)

set(JPEG_SOURCES ${JPEG_SOURCES} ${SIMD_OBJS})

add_library(jpeg12-static OBJECT ${JPEG12_SOURCES})
set_property(TARGET jpeg12-static PROPERTY COMPILE_FLAGS
  "-DBITS_IN_JSAMPLE=12")
add_library(jpeg16-static OBJECT ${JPEG16_SOURCES})
set_property(TARGET jpeg16-static PROPERTY COMPILE_FLAGS
  "-DBITS_IN_JSAMPLE=16")
add_library(${JPEG_LIBRARY} STATIC ${JPEG_SOURCES} ${SIMD_TARGET_OBJECTS}
  ${SIMD_OBJS} $<TARGET_OBJECTS:jpeg12-static>
  $<TARGET_OBJECTS:jpeg16-static>)

set_target_properties(${JPEG_LIBRARY}
  PROPERTIES OUTPUT_NAME ${JPEG_LIBRARY}
  DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
  COMPILE_PDB_NAME ${JPEG_LIBRARY}
  COMPILE_PDB_NAME_DEBUG "${JPEG_LIBRARY}${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
  )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${JPEG_LIBRARY} PROPERTIES FOLDER "3rdparty/jpeg")
  set_target_properties(jpeg12-static PROPERTIES FOLDER "3rdparty/jpeg")
  set_target_properties(jpeg16-static PROPERTIES FOLDER "3rdparty/jpeg")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${JPEG_LIBRARY} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev OPTIONAL)
endif()

ocv_install_3rdparty_licenses(libjpeg-turbo README.md LICENSE.md README.ijg)
