# ----------------------------------------------------------------------------
#  OpenCVPreamble.cmake
#  The following options MUST be parsed before the
#  "project(OpenCV ...)" declaration
# ----------------------------------------------------------------------------

# Prevent in-source builds
# This does its best to prevent users from trashing the source tree
# however CMakeFiles/ and CMakeCache.txt will (unfortunately) still be created
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  # this is a hack suggested on the cmake message boards to prevent cmake
  # from writing the CMakeCache.txt file after an in-source build attempt
  # which then causes future off-tree builds to fail
  file(MAKE_DIRECTORY CMakeCache.txt)
  message(FATAL_ERROR "
  =========================================================================
  WARNING: In-source builds are not allowed. Please create a separate build
  folder, and invoke cmake from there. For example:
    $ rm -r CMakeCache.txt CMakeFiles
    $ mkdir build
    $ cd build
    $ cmake ..
  =========================================================================\n")
endif()

# Following block can break build in case of cross-compilng
# but CMAKE_CROSSCOMPILING variable will be set only on project(OpenCV) command
# so we will try to detect crosscompiling by presense of CMAKE_TOOLCHAIN_FILE
if(NOT CMAKE_TOOLCHAIN_FILE)
  # it _must_ go before project(OpenCV) in order to work
  if(WIN32)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
  else()
    set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation Directory")
  endif()

  if(MSVC)
    set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
  endif()
else(NOT CMAKE_TOOLCHAIN_FILE)
  #Android: set output folder to ${CMAKE_BINARY_DIR}
  set( LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_BINARY_DIR} CACHE PATH "root for library output, set this to change where android libs are compiled to" )
  # any crosscompiling
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif(NOT CMAKE_TOOLCHAIN_FILE)

# set the available configuration types
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(DEFINED CMAKE_BUILD_TYPE)
  set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES} )
endif()
