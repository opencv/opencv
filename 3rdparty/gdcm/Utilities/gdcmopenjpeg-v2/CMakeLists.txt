# Main CMakeLists.txt to build the OpenJPEG project using CMake (www.cmake.org)
# Written by Mathieu Malaterre

# This CMake project will by default create a library called openjpeg
# But if you want to use this project within your own (CMake) project
# you will eventually like to prefix the library to avoid linking confusion
# For this purpose you can define a CMake var: OPENJPEG_NAMESPACE to whatever you like
# e.g.:
# set(OPENJPEG_NAMESPACE "GDCMOPENJPEG")
cmake_minimum_required(VERSION 2.8.9)

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
set(OPENJPEG_VERSION_MAJOR 2)
set(OPENJPEG_VERSION_MINOR 0)
set(OPENJPEG_VERSION_BUILD 0)
set(OPENJPEG_VERSION
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
  set(OPENJPEG_INSTALL_INCLUDE_DIR "include/${subdir}")
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
# Setup file for setting custom ctest vars
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/CMake/CTestCustom.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CTestCustom.cmake
  @ONLY
  )

#-----------------------------------------------------------------------------
# OpenJPEG build configuration options.
#option(BUILD_SHARED_LIBS "Build OpenJPEG with shared libraries." OFF)
#option(ENABLE_PROFILING "Enable profiling for the library" OFF)

#-----------------------------------------------------------------------------
#set (EXECUTABLE_OUTPUT_PATH ${OPENJPEG_BINARY_DIR}/bin CACHE PATH "Single output directory for building all executables.")
#set (LIBRARY_OUTPUT_PATH ${OPENJPEG_BINARY_DIR}/bin CACHE PATH "Single output directory for building all libraries.")
#mark_as_advanced(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH)


#-----------------------------------------------------------------------------
# For the codec...
#option(BUILD_EXAMPLES "Build the Examples (codec...)." OFF)


# configure name mangling to allow multiple libraries to coexist
# peacefully
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/openjpeg_mangle.h.in)
set(MANGLE_PREFIX ${OPENJPEG_LIBRARY_NAME})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/openjpeg_mangle.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/openjpeg_mangle.h
               @ONLY)
endif()

#-----------------------------------------------------------------------------
# Configure files with settings for use by the build.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/opj_configure.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/opj_configure.h)

if(NOT OPENJPEG_INSTALL_NO_DEVELOPMENT)
  install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/openjpeg_mangle.h
      ${CMAKE_CURRENT_BINARY_DIR}/opj_configure.h
    DESTINATION ${OPENJPEG_INSTALL_INCLUDE_DIR} COMPONENT Headers
  )
endif()


#-----------------------------------------------------------------------------
# Always build the library
include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})
subdirs(
  libopenjpeg
  )
#-----------------------------------------------------------------------------
# Build example only if requested
if(BUILD_EXAMPLES)
  # subdirs(codec)
endif()

#-----------------------------------------------------------------------------
# For the documentation
option(BUILD_DOCUMENTATION "Build the doxygen documentation" OFF)
mark_as_advanced(BUILD_DOCUMENTATION)
if(BUILD_DOCUMENTATION)
  subdirs(doc)
endif()

#-----------------------------------------------------------------------------
# For openjpeg team if they ever want CDash+CMake
option(BUILD_TESTING "Build the tests." OFF)
if(BUILD_TESTING)
  enable_testing()
  include(CTest)
endif()

#if(BUILD_TESTING)
#set(CMAKE_MODULE_PATH "${OPENJPEG_SOURCE_DIR}/CMake")
#find_package(FreeImage REQUIRED)
#include_directories( ${FREEIMAGE_INCLUDE_PATH} )
#
#  subdirs(
#    test_V2_tile_handling
#    test_Free_image_V2_tile_handling
#  )
#endif()

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
#if(CMAKE_COMPILER_IS_GNUCC)
#  # For all builds, make sure openjpeg is std99 compliant:
#  # set(CMAKE_C_FLAGS "-Wall -std=c99 ${CMAKE_C_FLAGS}") # FIXME: this setting prevented us from setting a coverage build.
#  # Do not use ffast-math for all build, it would produce incorrect results, only set for release:
#  set(CMAKE_C_FLAGS_RELEASE "-ffast-math ${CMAKE_C_FLAGS_RELEASE}")
#endif()

# install all targets referenced as OPENJPEGTargets
install(EXPORT ${GDCM_TARGETS_NAME} DESTINATION ${OPENJPEG_INSTALL_PACKAGE_DIR})
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/CMake/OpenJPEGConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/OpenJPEGConfig.cmake
  @ONLY
)
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/OpenJPEGConfig.cmake
  DESTINATION ${OPENJPEG_INSTALL_PACKAGE_DIR}
)

