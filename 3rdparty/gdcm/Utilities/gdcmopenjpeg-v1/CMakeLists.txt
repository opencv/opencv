# Main CMakeLists.txt to build the OpenJPEG project using CMake (www.cmake.org)
# Written by Mathieu Malaterre

# This CMake project will by default create a library called openjpeg
# But if you want to use this project within your own (CMake) project
# you will eventually like to prefix the library to avoid linking confusion
# For this purpose you can define a CMake var: OPENJPEG_NAMESPACE to whatever you like
# e.g.:
# set(OPENJPEG_NAMESPACE "GDCMOPENJPEG")
cmake_minimum_required(VERSION 2.8.9)
if(POLICY CMP0022)
  cmake_policy(SET CMP0022 OLD)
endif()

if(NOT OPENJPEG_NAMESPACE)
  set(OPENJPEG_NAMESPACE "OPENJPEG")
  set(OPENJPEG_STANDALONE 1)
endif()
# In all cases:
string(TOLOWER ${OPENJPEG_NAMESPACE} OPENJPEG_LIBRARY_NAME)

project(${OPENJPEG_NAMESPACE} C)

# Do full dependency headers.
include_regular_expression("^.*$")

#-----------------------------------------------------------------------------
# OPENJPEG version number, useful for packaging and doxygen doc:
set(OPENJPEG_VERSION_MAJOR 1)
set(OPENJPEG_VERSION_MINOR 4)
set(OPENJPEG_VERSION_BUILD 0)
set(OPENJPEG_VERSION
  "${OPENJPEG_VERSION_MAJOR}.${OPENJPEG_VERSION_MINOR}.${OPENJPEG_VERSION_BUILD}")
set(PACKAGE_VERSION
  "${OPENJPEG_VERSION_MAJOR}.${OPENJPEG_VERSION_MINOR}.${OPENJPEG_VERSION_BUILD}")
# This setting of SOVERSION assumes that any API change
# will increment either the minor or major version number of openjpeg
set(OPENJPEG_LIBRARY_PROPERTIES
  VERSION   "${OPENJPEG_VERSION_MAJOR}.${OPENJPEG_VERSION_MINOR}.${OPENJPEG_VERSION_BUILD}"
  SOVERSION "${OPENJPEG_VERSION_MAJOR}.${OPENJPEG_VERSION_MINOR}"
)
# You will also need to define a value for the following variables:
# OPENJPEG_INSTALL_BIN_DIR          - binary dir (executables)
# OPENJPEG_INSTALL_LIB_DIR          - library dir (libs)
# OPENJPEG_INSTALL_DATA_DIR         - share dir (say, examples, data, etc)
# OPENJPEG_INSTALL_INCLUDE_DIR      - include dir (headers)


# On Visual Studio 8 MS deprecated C. This removes all 1.276E1265 security
# warnings
if(WIN32)
  if(NOT BORLAND)
    if(NOT CYGWIN)
      if(NOT MINGW)
        if(NOT ITK_ENABLE_VISUAL_STUDIO_DEPRECATED_C_WARNINGS)
          add_definitions(
            -D_CRT_FAR_MAPPINGS_NO_DEPRECATE
            -D_CRT_IS_WCTYPE_NO_DEPRECATE
            -D_CRT_MANAGED_FP_NO_DEPRECATE
            -D_CRT_NONSTDC_NO_DEPRECATE
            -D_CRT_SECURE_NO_DEPRECATE
            -D_CRT_SECURE_NO_DEPRECATE_GLOBALS
            -D_CRT_SETERRORMODE_BEEP_SLEEP_NO_DEPRECATE
            -D_CRT_TIME_FUNCTIONS_NO_DEPRECATE
            -D_CRT_VCCLRIT_NO_DEPRECATE
            -D_SCL_SECURE_NO_DEPRECATE
            )
        endif()
      endif()
    endif()
  endif()
endif()


# --------------------------------------------------------------------------
# Install directories

string(TOLOWER ${PROJECT_NAME} projectname)
set(subdir "${projectname}-${OPENJPEG_VERSION_MAJOR}.${OPENJPEG_VERSION_MINOR}")

if(NOT OPENJPEG_INSTALL_BIN_DIR)
  set(OPENJPEG_INSTALL_BIN_DIR "bin")
endif()

if(NOT OPENJPEG_INSTALL_LIB_DIR)
  set(OPENJPEG_INSTALL_LIB_DIR "lib")
endif()

if(NOT OPENJPEG_INSTALL_DATA_DIR)
  set(OPENJPEG_INSTALL_DATA_DIR "share/${subdir}")
endif()

if(NOT OPENJPEG_INSTALL_INCLUDE_DIR)
  set(OPENJPEG_INSTALL_INCLUDE_DIR "include/")
endif()

if(NOT OPENJPEG_INSTALL_MAN_DIR)
  set(OPENJPEG_INSTALL_MAN_DIR "share/man/")
endif()

if(NOT OPENJPEG_INSTALL_DOC_DIR)
  set(OPENJPEG_INSTALL_DOC_DIR "share/doc/${subdir}")
endif()

if(NOT OPENJPEG_INSTALL_PACKAGE_DIR)
  set(OPENJPEG_INSTALL_PACKAGE_DIR ${OPENJPEG_INSTALL_LIB_DIR}/${subdir}
    CACHE INTERNAL "")
endif()

#-----------------------------------------------------------------------------
# Test for some required system information.
include (${CMAKE_ROOT}/Modules/CMakeBackwardCompatibilityC.cmake)

#-----------------------------------------------------------------------------
# Test for getopt being available in this system
include (${PROJECT_SOURCE_DIR}/CMake/CheckHaveGetopt.cmake )

#-----------------------------------------------------------------------------
# Setup file for setting custom ctest vars
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/CMake/CTestCustom.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CTestCustom.cmake
  @ONLY
  )

#-----------------------------------------------------------------------------
# OpenJPEG build configuration options.
#option(BUILD_SHARED_LIBS "Build OpenJPEG shared library and link executables against it." ON)

#-----------------------------------------------------------------------------


# configure name mangling to allow multiple libraries to coexist
# peacefully
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/openjpeg_mangle.h.in)
set(MANGLE_PREFIX ${OPENJPEG_LIBRARY_NAME})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/openjpeg_mangle.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/openjpeg_mangle.h
               @ONLY)
endif()

if(NOT OPENJPEG_INSTALL_NO_DEVELOPMENT)
  install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/openjpeg_mangle.h
    DESTINATION ${OPENJPEG_INSTALL_INCLUDE_DIR} COMPONENT Headers
  )
endif()

#-----------------------------------------------------------------------------
# Always build the library
include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})
subdirs(libopenjpeg)

#-----------------------------------------------------------------------------
# Build CODEC executables ?
#option(OPJ_BUILD_CODEC "Build the CODEC executables" OFF)
if(OPJ_BUILD_CODEC)
  subdirs(codec)
endif()

#-----------------------------------------------------------------------------
# Build MJ2 executables ?
#option(BUILD_MJ2 "Build the MJ2 executables." OFF)
if(BUILD_MJ2)
  subdirs(mj2)
endif()

#-----------------------------------------------------------------------------
# Build JPWL executables ?
#option(BUILD_JPWL "Build the JPWL executables" OFF)
if(BUILD_JPWL)
  subdirs(jpwl)
endif()

#-----------------------------------------------------------------------------
# Build JP3D executables ?
#option(BUILD_JP3D "Build the JP3D executables" OFF)
if(BUILD_JP3D)
  subdirs(jp3d)
endif()

#-----------------------------------------------------------------------------
# Build INDEXER_JPIP executables ?
#option(BUILD_INDEXER_JPIP "Build the INDEXER_JPIP executables" OFF)
if(BUILD_INDEXER_JPIP AND NOT UNIX)
  subdirs(indexer_JPIP)
endif()

#-----------------------------------------------------------------------------
# Build DOCUMENTATION ?
#option(BUILD_DOC "Build the doxygen documentation" OFF)
if(BUILD_DOC)
  subdirs(doc)
endif()

#-----------------------------------------------------------------------------
# For openjpeg team if they ever want CDash+CMake
option(BUILD_TESTING "Build the tests." OFF)
if(BUILD_TESTING)
  enable_testing()
  include(CTest)
endif()

# Adding test with dataset from:
# http://www.crc.ricoh.com/~gormish/jpeg2000conformance/
# -> wget http://www.crc.ricoh.com/~gormish/jpeg2000conformance/j2kp4files_v1_5.zip
# http://www.jpeg.org/jpeg2000guide/testimages/testimages.html
#-----------------------------------------------------------------------------
# Adding JPEG2000_CONFORMANCE_DATA_ROOT
find_path(JPEG2000_CONFORMANCE_DATA_ROOT testimages.html
  ${OPENJPEG_SOURCE_DIR}/../jpeg2000testimages
  $ENV{JPEG2000_CONFORMANCE_DATA_ROOT}
)
mark_as_advanced(JPEG2000_CONFORMANCE_DATA_ROOT)

#-----------------------------------------------------------------------------
# Compiler specific flags:
if(CMAKE_COMPILER_IS_GNUCC)
  # For all builds, make sure openjpeg is std99 compliant:
  # set(CMAKE_C_FLAGS "-Wall -std=c99 ${CMAKE_C_FLAGS}") # FIXME: this setting prevented us from setting a coverage build.
  # Do not use ffast-math for all build, it would produce incorrect results, only set for release:
  #set(CMAKE_C_FLAGS_RELEASE "-ffast-math ${CMAKE_C_FLAGS_RELEASE}")
endif()

# install CHANGES and LICENSE
#install(
#  FILES       CHANGES
#              LICENSE
#  DESTINATION ${OPENJPEG_INSTALL_DOC_DIR})
#
#if(UNIX OR CYGWIN)
#  set(CMAKE_INCLUDE_PATH /usr/include /usr/local/include /opt/include
#	/opt/local/include /usr/include/libpng /usr/include/libpng14
#	/usr/include/libpng12 /usr/local/include/libpng
#	/usr/local/include/libpng14 /usr/local/include/libpng12
#	/opt/include/libpng /opt/include/libpng14 /opt/include/libpng12
#	/opt/local/include/libpng /opt/local/include/libpng14
#	/opt/local/include/libpng12 )
#  set(CMAKE_LIBRARY_PATH /usr/lib /usr/local/lib /opt/lib /opt/local/lib)
#elseif(WIN32)
#  set(CMAKE_INCLUDE_PATH ${OPENJPEG_SOURCE_DIR}/libs/libtiff
#	${OPENJPEG_SOURCE_DIR}/libs/png ${OPENJPEG_SOURCE_DIR}/libs/lcms2
#	C:/WINDOWS/system32/user )
#  set(CMAKE_LIBRARY_PATH ${OPENJPEG_SOURCE_DIR}/libs/libtiff
#        ${OPENJPEG_SOURCE_DIR}/libs/png ${OPENJPEG_SOURCE_DIR}/libs/lcms2
#	C:/WINDOWS/system32/user )
#endif()
#
include (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
CHECK_INCLUDE_FILE("strings.h"       HAVE_STRINGS_H)
CHECK_INCLUDE_FILE("inttypes.h"      HAVE_INTTYPES_H)
CHECK_INCLUDE_FILE("memory.h"        HAVE_MEMORY_H)
CHECK_INCLUDE_FILE("stdint.h"        HAVE_STDINT_H)
CHECK_INCLUDE_FILE("stdlib.h"        HAVE_STDLIB_H)
CHECK_INCLUDE_FILE("string.h"        HAVE_STRING_H)
CHECK_INCLUDE_FILE("sys/stat.h"      HAVE_SYS_STAT_H)
CHECK_INCLUDE_FILE("sys/types.h"     HAVE_SYS_TYPES_H)
CHECK_INCLUDE_FILE("unistd.h"        HAVE_UNISTD_H)

#find_file(HAVE_STRINGS_H_FOUND strings.h)
#if(NOT HAVE_STRINGS_H_FOUND STREQUAL "HAVE_STRINGS_H_FOUND-NOTFOUND")
#  find_file(HAVE_STRINGS_H strings.h)
#  set(HAS_STRINGS_H 1)
#endif()
#find_file(HAVE_INTTYPES_H_FOUND inttypes.h)
#if(NOT HAVE_INTTYPES_H_FOUND STREQUAL "HAVE_INTTYPES_H_FOUND-NOTFOUND")
#  find_file(HAVE_INTTYPES_H inttypes.h)
#  set(HAS_INTTYPES_H 1)
#endif()
#find_file(HAVE_MEMORY_H_FOUND memory.h)
#if(NOT HAVE_MEMORY_H_FOUND STREQUAL "HAVE_MEMORY_H_FOUND-NOTFOUND")
#  find_file(HAVE_MEMORY_H memory.h)
#  set(HAS_MEMORY_H 1)
#endif()
#find_file(HAVE_STDINT_H_FOUND stdint.h)
#if(NOT HAVE_STDINT_H_FOUND STREQUAL "HAVE_STDINT_H_FOUND-NOTFOUND")
#  find_file(HAVE_STDINT_H stdint.h)
#  set(HAS_STDINT_H 1)
#endif()
#find_file(HAVE_STDLIB_H_FOUND stdlib.h)
#if(NOT HAVE_STDLIB_H_FOUND STREQUAL "HAVE_STDLIB_H_FOUND-NOTFOUND")
#  find_file(HAVE_STDLIB_H stdlib.h)
#  set(HAS_STDLIB_H 1)
#endif()
#find_file(HAVE_STRING_H_FOUND string.h)
#if(NOT HAVE_STRING_H_FOUND STREQUAL "HAVE_STRING_H_FOUND-NOTFOUND")
#  find_file(HAVE_STRING_H string.h)
#  set(HAS_STRING_H 1)
#endif()
#find_file(HAVE_SYS_STAT_H_FOUND sys/stat.h)
#if(NOT HAVE_SYS_STAT_H_FOUND STREQUAL "HAVE_SYS_STAT_H_FOUND-NOTFOUND")
#  find_file(HAVE_SYS_STAT_H sys/stat.h)
#  set(HAS_SYS_STAT_H 1)
#endif()
#find_file(HAVE_SYS_TYPES_H_FOUND sys/types.h)
#if(NOT HAVE_SYS_TYPES_H_FOUND STREQUAL "HAVE_SYS_TYPES_H_FOUND-NOTFOUND")
#  find_file(HAVE_SYS_TYPES_H sys/types.h)
#  set(HAS_SYS_TYPES_H 1)
#endif()
#find_file(HAVE_UNISTD_H_FOUND unistd.h)
#if(NOT HAVE_UNISTD_H_FOUND STREQUAL "HAVE_UNISTD_H_FOUND-NOTFOUND")
#  find_file(HAVE_UNISTD_H unistd.h)
#  set(HAS_UNISTD_H 1)
#endif()
#
# Does the system have png library installed ?
#
#find_package(PNG)
#
if(PNG_FOUND)
  set(HAVE_PNG_H 1)
  set(HAVE_LIBPNG 1)
endif()
#
# Does the system have tiff library installed ?
# 
#find_package(TIFF)
#
if(TIFF_FOUND)
  set(HAVE_TIFF_H 1)
  set(HAVE_LIBTIFF 1)
endif()
#
#
# Does the system have lcms library installed ?
#
set(LCMS_LIB "")
#find_file(LCMS2_HEADER_FOUND lcms2.h)
#
if(LCMS2_HEADER_FOUND STREQUAL "LCMS2_HEADER_FOUND-NOTFOUND")
 set(LCMS2_HEADER_FOUND "")
endif()
if(LCMS2_HEADER_FOUND)
 find_path(LCMS_INCLUDE_DIR  lcms2.h)
 if(UNIX OR CYGWIN)
   find_library(HAVE_LIBLCMS2 lcms2)
 else()
   find_library(HAVE_LIBLCMS2 lcms2_static.lib)
 endif() 
 if(HAVE_LIBLCMS2 STREQUAL "HAVE_LIBLCMS2-NOTFOUND")
  set(HAVE_LIBLCMS2 "")
 endif()
 if(HAVE_LIBLCMS2)
  set(LCMS_LIB "${HAVE_LIBLCMS2}")
  set(HAVE_LCMS2_LIB 1)
  set(HAVE_LCMS2_H 1)
 endif()
endif()
#if(NOT LCMS2_HEADER_FOUND)
# find_file(LCMS1_HEADER_FOUND lcms.h)
# if(LCMS1_HEADER_FOUND STREQUAL "LCMS1_HEADER_FOUND-NOTFOUND")
#  set(LCMS1_HEADER_FOUND "")
# endif()
# if(LCMS1_HEADER_FOUND)
#  find_path(LCMS_INCLUDE_DIR  lcms.h)
#  find_library(HAVE_LIBLCMS1 lcms)
#  if(HAVE_LIBLCMS1 STREQUAL "HAVE_LIBLCMS1-NOTFOUND")
#   set(HAVE_LIBLCMS1 "")
#  endif()
#  if(HAVE_LIBLCMS1)
#   set(LCMS_LIB "${HAVE_LIBLCMS1}")
#   set(HAVE_LCMS1_LIB 1)
#   set(HAVE_LCMS1_H 1)
#  endif()
# endif()
#endif()
#
# generate opj_config.h
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/opj_configh.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/opj_config.h"
  @ONLY 
)
#mark_as_advanced( HAVE_INTTYPES_H_FOUND HAVE_LIBLCMS2 HAVE_MEMORY_H
#  HAVE_MEMORY_H_FOUND     HAVE_STDINT_H_FOUND     HAVE_STDLIB_H_FOUND
#  HAVE_STRINGS_H          HAVE_STRINGS_H_FOUND    HAVE_STRING_H
#  HAVE_STRING_H_FOUND     HAVE_SYS_STAT_H         HAVE_SYS_STAT_H_FOUND
#  HAVE_SYS_TYPES_H_FOUND  HAVE_UNISTD_H_FOUND     LCMS2_HEADER_FOUND
#  LCMS_INCLUDE_DIR        OPJ_BUILD_CODEC         )
# install all targets referenced as OPENJPEGTargets
#install(EXPORT ${GDCM_TARGETS_NAME} DESTINATION ${OPENJPEG_INSTALL_PACKAGE_DIR})
#configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/CMake/OpenJPEGConfig.cmake.in
#  ${CMAKE_CURRENT_BINARY_DIR}/OpenJPEGConfig.cmake
#  @ONLY
#)
#install( FILES ${CMAKE_CURRENT_BINARY_DIR}/OpenJPEGConfig.cmake
#  DESTINATION ${OPENJPEG_INSTALL_PACKAGE_DIR}
#)
