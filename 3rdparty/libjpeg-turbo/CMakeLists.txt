project(${JPEG_LIBRARY} C)

ocv_warnings_disable(CMAKE_C_FLAGS -Wunused-parameter -Wsign-compare -Wshorten-64-to-32 -Wimplicit-fallthrough)

set(VERSION_MAJOR 1)
set(VERSION_MINOR 5)
set(VERSION_REVISION 3)
set(VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_REVISION})
set(LIBJPEG_TURBO_VERSION_NUMBER 1005003)

string(TIMESTAMP BUILD "opencv-${OPENCV_VERSION}-libjpeg-turbo")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(BUILD "${BUILD}-debug")
endif()

message(STATUS "libjpeg-turbo: VERSION = ${VERSION}, BUILD = ${BUILD}")

option(WITH_ARITH_ENC "Include arithmetic encoding support when emulating the libjpeg v6b API/ABI" TRUE)
option(WITH_ARITH_DEC "Include arithmetic decoding support when emulating the libjpeg v6b API/ABI" TRUE)

if(NOT DEFINED SIZEOF_SIZE_T)
  if(IOS)  # Workaround iOS issues
    set(SIZEOF_SIZE_T "${CMAKE_SIZEOF_VOID_P}")
    message(STATUS "SIZEOF_SIZE_T = ${SIZEOF_SIZE_T}")
  else()
    include(CheckTypeSize)
    CHECK_TYPE_SIZE("size_t" SIZEOF_SIZE_T)
  endif()
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

if(MSVC)
  add_definitions(-W3 -wd4996 -wd4018)
endif()

configure_file(jconfig.h.in jconfig.h)
configure_file(jconfigint.h.in jconfigint.h)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(JPEG_SOURCES jcapimin.c jcapistd.c jccoefct.c jccolor.c jcdctmgr.c jchuff.c
  jcinit.c jcmainct.c jcmarker.c jcmaster.c jcomapi.c jcparam.c jcphuff.c
  jcprepct.c jcsample.c jctrans.c jdapimin.c jdapistd.c jdatadst.c jdatasrc.c
  jdcoefct.c jdcolor.c jddctmgr.c jdhuff.c jdinput.c jdmainct.c jdmarker.c
  jdmaster.c jdmerge.c jdphuff.c jdpostct.c jdsample.c jdtrans.c jerror.c
  jfdctflt.c jfdctfst.c jfdctint.c jidctflt.c jidctfst.c jidctint.c jidctred.c
  jquant1.c jquant2.c jutils.c jmemmgr.c jmemnobs.c)

if(WITH_ARITH_ENC OR WITH_ARITH_DEC)
  set(JPEG_SOURCES ${JPEG_SOURCES} jaricom.c)
endif()

if(WITH_ARITH_ENC)
  set(JPEG_SOURCES ${JPEG_SOURCES} jcarith.c)
endif()

if(WITH_ARITH_DEC)
  set(JPEG_SOURCES ${JPEG_SOURCES} jdarith.c)
endif()

# No SIMD
set(JPEG_SOURCES ${JPEG_SOURCES} jsimd_none.c)

ocv_list_add_prefix(JPEG_SOURCES src/)

add_library(${JPEG_LIBRARY} STATIC ${JPEG_SOURCES} ${SIMD_OBJS})

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
  ocv_install_target(${JPEG_LIBRARY} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

ocv_install_3rdparty_licenses(libjpeg-turbo README.md LICENSE.md README.ijg)
